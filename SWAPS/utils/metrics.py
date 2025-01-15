import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from utils.plot import plot_scatter
from typing import Literal


class RT_metrics:
    def __init__(self, RT_obs: pd.Series, RT_pred: pd.Series) -> None:
        self.y_true = RT_obs
        self.y_pred = RT_pred
        self.y_delta = self.y_pred - self.y_true

    def PlotDeltaRT(self):
        self.y_delta.hist()
        plt.show()

    def CalcDeltaRTwidth(
        self, coverage: int = 95, calc: Literal["abs", "real"] = "abs"
    ):
        """
        Calculate delta RT (95)

        :calc: how to calculate the metric, 'abs' use absolute error (as in DeepLC)
               and 'real' use real error and take the distance from 97.5% - 2.5%
        """
        self.coverage = coverage
        if calc == "real":
            perc = (100 - coverage) / 2
            self.p_low = np.percentile(self.y_delta, perc)
            self.p_high = np.percentile(self.y_delta, 100 - perc)
            return self.p_high - self.p_low
        elif calc == "abs":
            width = np.percentile(abs(self.y_delta), self.coverage)
            self.p_low = -width
            self.p_high = width
            return width

    def CalcPrsCorr(self):
        return pearsonr(x=self.y_true, y=self.y_pred)[0]

    def CalcMAE(self):
        return sum(abs(self.y_true - self.y_pred)) / len(self.y_true.index)

    def PlotRTScatter(self):
        print("Green line shows deltaRT", self.coverage)
        _, _, _ = plot_scatter(
            x=self.y_true,
            y=self.y_pred,
            log_x=False,
            log_y=False,
            filter_thres=0,
            interactive=False,
            show_diag=True,
            show_conf=(self.p_low, self.p_high),
            x_label="Experimental RT (min)",
            y_label="Predicted RT (min)",
            title="Corr. between Experimental and Predicted RT",
        )
