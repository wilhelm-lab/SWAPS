import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
import subprocess
import threading
from queue import Queue, Empty

# This is a simplified version of the full config for GUI purposes.
# We'll load the full config and then update it with the GUI values.
from swaps.utils.singleton_swaps_optimization import swaps_optimization_cfg


class SWAPS_GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("SWAPS GUI")

        self.config = swaps_optimization_cfg.clone()

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(pady=10, padx=10, expand=True, fill="both")

        self.create_general_tab()
        self.create_prepare_dict_tab()
        self.create_optimization_tab()

        self.log_frame = ttk.Frame(root, padding="10")
        self.log_frame.pack(expand=True, fill="both")
        self.log_text = scrolledtext.ScrolledText(
            self.log_frame, wrap=tk.WORD, height=15
        )
        self.log_text.pack(expand=True, fill="both")

        self.create_buttons()

        self.log_queue = Queue()
        self.root.after(100, self.process_log_queue)

    def process_log_queue(self):
        try:
            while True:
                line = self.log_queue.get_nowait()
                self.log_text.insert(tk.END, line)
                self.log_text.see(tk.END)
        except Empty:
            pass
        self.root.after(100, self.process_log_queue)

    def create_general_tab(self):
        general_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(general_frame, text="General")

        self.create_entry(general_frame, "DATA_PATH", "Data Path:", 0)
        self.create_entry(general_frame, "RESULT_PATH", "Result Path:", 1)
        self.create_entry(general_frame, "SEARCH_OUTPUT_PATH", "Search Output Path:", 2)
        self.create_search_engine_dropdown(
            general_frame, "PREPARE_DICT.SEARCH_ENGINE", "Search Engine:", 3
        )

    def create_prepare_dict_tab(self):
        prepare_dict_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(prepare_dict_frame, text="Prepare Dictionary")

        self.create_entry(prepare_dict_frame, "PREPARE_DICT.RT_TOL", "RT Tolerance:", 0)
        self.create_entry(prepare_dict_frame, "PREPARE_DICT.IM_LENGTH", "IM Length:", 1)
        self.create_entry(
            prepare_dict_frame, "PREPARE_DICT.PPM_TOL", "PPM Tolerance:", 2
        )

    def create_optimization_tab(self):
        optimization_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(optimization_frame, text="Optimization")

        self.create_entry(
            optimization_frame, "OPTIMIZATION.N_BATCH", "Number of Batches:", 0
        )

    def create_entry(self, parent, config_key, label_text, row):
        label = ttk.Label(parent, text=label_text)
        label.grid(row=row, column=0, sticky="w", padx=5, pady=5)

        entry = ttk.Entry(parent, width=50)
        entry.grid(row=row, column=1, padx=5, pady=5)

        browse_button = ttk.Button(
            parent, text="Browse", command=lambda: self.browse_path(entry)
        )
        browse_button.grid(row=row, column=2, padx=5, pady=5)

        keys = config_key.split(".")
        value = self.config
        for key in keys:
            value = value[key]
        entry.insert(0, str(value))

        entry.bind(
            "<FocusOut>",
            lambda event, k=config_key, e=entry: self.update_config(k, e.get()),
        )

        return entry

    def create_search_engine_dropdown(self, parent, config_key, label_text, row):
        label = ttk.Label(parent, text=label_text)
        label.grid(row=row, column=0, sticky="w", padx=5, pady=5)

        options = ["maxquant", "fragpipe", "sage"]
        variable = tk.StringVar(parent)

        keys = config_key.split(".")
        value = self.config
        for key in keys:
            value = value[key]
        variable.set(str(value))

        dropdown = ttk.OptionMenu(
            parent,
            variable,
            *options,
            command=lambda value: self.update_config(config_key, value),
        )
        dropdown.grid(row=row, column=1, padx=5, pady=5, sticky="ew")

    def browse_path(self, entry):
        path = filedialog.askdirectory()
        if path:
            entry.delete(0, tk.END)
            entry.insert(0, path)

    def update_config(self, key, value):
        keys = key.split(".")
        cfg_node = self.config
        for k in keys[:-1]:
            cfg_node = cfg_node[k]

        # Convert value to the correct type
        original_value = cfg_node[keys[-1]]
        if isinstance(original_value, bool):
            value = value.lower() in ["true", "1", "yes"]
        elif isinstance(original_value, int):
            value = int(value)
        elif isinstance(original_value, float):
            value = float(value)

        cfg_node[keys[-1]] = value

    def create_buttons(self):
        button_frame = ttk.Frame(self.root, padding="10")
        button_frame.pack(fill="x")

        save_button = ttk.Button(
            button_frame, text="Save Config", command=self.save_config
        )
        save_button.pack(side="left", padx=5)

        load_button = ttk.Button(
            button_frame, text="Load Config", command=self.load_config
        )
        load_button.pack(side="left", padx=5)

        self.run_button = ttk.Button(button_frame, text="Run", command=self.run_sbs)
        self.run_button.pack(side="right", padx=5)

    def save_config(self):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".yaml",
            filetypes=[("YAML files", "*.yaml"), ("All files", "*.*")],
        )
        if file_path:
            with open(file_path, "w") as f:
                f.write(self.config.dump())

    def load_config(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("YAML files", "*.yaml"), ("All files", "*.*")]
        )
        if file_path:
            self.config.merge_from_file(file_path)
            self.update_gui_from_config()

    def update_gui_from_config(self):
        for tab in self.notebook.tabs():
            frame = self.root.nametowidget(tab)
            for widget in frame.winfo_children():
                if isinstance(widget, ttk.Entry):
                    # This is a bit of a hack to get the config key from the widget.
                    # A better way would be to store the config key in the widget itself.
                    # For now, we'll just re-create the widgets.
                    # TODO: Find a better way to do this.
                    pass

        # For simplicity, we just destroy and recreate the tabs
        for tab in self.notebook.tabs():
            self.notebook.forget(tab)

        self.create_general_tab()
        self.create_prepare_dict_tab()
        self.create_optimization_tab()

    def run_sbs(self):
        self.run_button.config(state=tk.DISABLED)
        self.log_text.delete("1.0", tk.END)

        temp_config_path = "temp_config.yaml"
        with open(temp_config_path, "w") as f:
            f.write(self.config.dump())

        self.process_thread = threading.Thread(
            target=self.run_process_in_thread, args=(temp_config_path,)
        )
        self.process_thread.start()

    def run_process_in_thread(self, config_path):
        command = ["python", "-u", "swaps/sbs_runner_ims.py", config_path]
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                universal_newlines=True,
            )
            for line in iter(process.stdout.readline, ""):
                self.log_queue.put(line)
            process.stdout.close()
            process.wait()
        except Exception as e:
            self.log_queue.put(f"Error: {e}\n")
        finally:
            self.root.after(0, lambda: self.run_button.config(state=tk.NORMAL))


if __name__ == "__main__":
    root = tk.Tk()
    app = SWAPS_GUI(root)
    root.mainloop()
