# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**SWAPS** (Scan-Wise Activation and Peak Selection) is an MS1-centric peptide identification and quantification framework for proteomics. It combines sparse coding on MS1 scan data, image-based peak detection (watershed + SIFT/Zernike), and ML-based FDR control (Mokapot/XGBoost) to quantify peptides across multiple LC-MS/MS runs.

Publication: https://pubs.acs.org/doi/10.1021/acs.jproteome.4c00972

## Commands

```bash
# Install (Python 3.10 required)
conda create --name swaps python==3.10.12 --no-default-packages
conda activate swaps
poetry install

# Run pipeline
sbs_runner_ims <path-to-config.yaml>

# Run tests (fast, no real data needed)
pytest
pytest -m "not slow"

# Run slow end-to-end tests (opt-in)
pytest -m slow

# Run integration tests (requires real .d files)
export SWAPS_TEST_RAW_DIR=<path-to-.d-files>
export SWAPS_TEST_FRAGPIPE_DIR=<path-to-fragpipe-output>
pytest -m integration

# Coverage
pytest --cov=swaps --cov-report=html
```

## Architecture & Data Flow

The pipeline runs in six sequential stages, each checkpointing results to disk:

### 1. Prepare Dictionary (`swaps/prepare_dict/prepare_dict.py`)
Parses search engine output (MaxQuant `evidence.txt`, FragPipe, or Sage), builds a per-peptide reference dictionary with predicted RT (PeptDeep), predicted/measured ion mobility, expected m/z with isotope envelopes. Output: `dict_ref.pkl` — a pandas DataFrame indexed by `mz_rank`.

### 2. Scan-Wise Activation (`swaps/optimization/inference.py`)
Loads each `.d` raw file via AlphaTims. For every MS1 scan frame: extracts a 2D m/z × IM matrix, builds a candidate array of peptide m/z patterns, solves a sparse coding problem (activation matrix), and writes results as parquet files sorted by `mz_rank`. Output: `activation/*.parquet` + `dict_ref_with_activation.pkl`.

### 3. Feature Matching & Quantification (`swaps/postprocessing/match_features.py`)
The largest module (~3400 lines). Loads activation parquets in contiguous `mz_rank` batches across all runs. For each peptide's 2D RT×IM image patch:
- **Reference path**: watershed segmentation on the best-quality run; template-match other runs using SIFT descriptors and Zernike moments; snap coordinates to watershed labels.
- **Consensus path** (`generate_consensus=True`): average aligned images across runs, re-segment, then re-snap all anchors to consensus labels.
Outputs: `matches_target/decoy.parquet`, `pp_*.parquet` (peak properties).

### 4. FDR Control (`swaps/postprocessing/rescore.py`)
Normalizes RT/IM shift features per run, then trains Mokapot (XGBoost) using target-decoy competition. Features include `rt_shift`, `im_shift`, `template_matching_score`, SIFT score, and Zernike score. Applies 5% FDR training / 1% FDR test threshold with per-PSM deduplication.

### 5. Label-Free Quantification (`swaps/postprocessing/direct_lfq.py`)
Adapter layer reformats SWAPS peak areas into DirectLFQ's input format, runs protein-level quantification.

### 6. Orchestration (`swaps/sbs_runner_ims.py`)
Top-level entry point. Loops over raw files, calls each stage in order, saves a timestamped config YAML for reproducibility.

## Key Design Patterns

**YACS Configuration**: All settings flow through a singleton config (`swaps/utils/singleton_swaps_optimization.py`). Configs are hierarchical YAML files. Example configs live in `swaps/utils/exp_configs/`. Load via `merge_cfg_from_file()`.

**Parallel Processing**: Three levels — `ThreadPoolExecutor` for multi-run activation building; `ProcessPoolExecutor` for within-batch peptide quantification; DuckDB row-group skipping for fast parquet I/O by `mz_rank`.

**Worker Context Caching**: `_WORKER_CONTEXT` in `match_features.py` caches immutable shared state in ProcessPool workers to avoid repeated serialization overhead.

**Sparse → Dense Representation**: SWA uses COO sparse tensors (memory efficient for high-dimensional scan data); feature matching uses dense 2D patches (fast convolutions). Conversion utilities in `swaps/postprocessing/helper.py` and `swaps/postprocessing/image_processing.py`.

**Search Engine Abstraction**: `swaps/prepare_dict/search_engine_output_parser.py` provides pluggable parsers (MaxQuant/FragPipe/Sage) that all output a canonical pandas DataFrame.

## Testing Strategy

- Default `pytest` runs fast unit tests with synthetic DataFrames — no real data needed.
- `pytest -m slow` runs full-pipeline tests (3–4 runs end-to-end).
- `pytest -m integration` requires setting `SWAPS_TEST_RAW_DIR` / `SWAPS_TEST_FRAGPIPE_DIR` env vars pointing to real `.d` files.
- Coverage excludes `peak_detection_2d` and `result_analysis` modules.

## Documentation Maintenance

`docs/sbs_runner_flow.html` is a standalone decision-flow diagram + config reference table for `swaps/sbs_runner_ims.py` (the six-stage orchestration logic and every YACS config branch it reads). After any code change touching `sbs_runner_ims.py`, the stage modules it calls, or the config defaults in `swaps/utils/singleton_swaps_optimization.py`, evaluate whether the change alters a branch, gate, default, or config key shown in that doc (new/removed config flag, changed default, new decision point, changed stage order, a knob that becomes live/dead). If so, update `docs/sbs_runner_flow.html` (the mermaid flowchart and/or the config reference tables) to match before considering the task done.

## Don't
- Don't duplicate logic — always check for existing functions/utilities before writing new code. Refactor and reuse.
- Don't write long monolithic functions — extract repeated patterns into shared helpers.
- Don't inline logic that already exists elsewhere in the codebase.
- Don't add comments that just restate what the code does, only short hint
- NEVER run `rm` (or any other deletion of files/directories) without first asking for explicit consent, even for files/directories generated earlier in the same session.