import matplotlib.pyplot as plt
import numpy as np

from matplotlib.figure import Figure


class MrFppi:

    def __init__(self, result):
        self.result = result
        self.mr = 1.0 - np.true_divide(result.tp, result.nof_gts)  # 1 - recall
        self.fppi = np.true_divide(result.fp, result.nof_imgs)  # false positives per image

    def create_plot(self, title=None, label=None):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        ax.loglog()

        ax.set_xlabel('FPPI')
        ax.set_ylabel('Miss Rate')

        ax.grid(b=True, which='major', linestyle='-')
        ax.yaxis.grid(b=True, which='minor', linestyle='--')

        ax.set_xlim([5 * 1e-5, 10])
        ax.set_ylim([8 * 1e-3, 1])

        ax.plot(self.fppi, self.mr)
        if title:
            ax.set_title(title)
        if label:
            ax.legend([label])

        return fig

    def log_avg_mr_reference_implementation(self):
        ys = np.true_divide(self.result.tp, self.result.nof_gts)  # recall
        xs = np.true_divide(self.result.fp, self.result.nof_imgs)  # false positives per image
        ref = np.power(10, np.linspace(-2, 0, 9))
        result = np.zeros(ref.shape)
        for i in range(len(ref)):
            j = np.argwhere(xs <= ref[i]).flatten()
            if j.size:
                # "[...](for curves that end before reaching a given FPPI
                #                   rate, the minimum miss rate achieved is used)"
                result[i] = ys[j[-1]]
            else:
                # if first detection is a tp the minimal fppi is zero and we never execute this else
                # case (assuming fppi ref points are > 0)
                # if first detection is a fp the minimal fppi is 1/nof_imgs, we execute this else
                # case if 1/nof_imgs > min(fppi ref points)
                # this part corresponds to xs[0] = -inf and ys[0] = 0
                result[i] = 0  # unnecessary since result[i] is 0 anyways...

        # The 1e-10 is needed if missrate of 0 (equals recall of 1 as result variable stores recall)
        # is reached.
        return np.exp(np.mean(np.log(np.maximum(1e-10, 1 - result))))


class MrFppiMultiPlot:
    def __init__(self, title, append_score_to_label=True,
                 score_method=MrFppi.log_avg_mr_reference_implementation):
        self.__helper = MultiPlotHelper(title, 'Miss Rate', 'False Positives per Image')
        self.append_score_to_label = append_score_to_label
        self.score_method = score_method
        self.__helper.ax.loglog()
        assert score_method.im_class == MrFppi, 'score_method must be a method of class MrFppi'

    @property
    def fig(self):
        return self.__helper.fig

    @property
    def ax(self):
        return self.__helper.ax

    def add_result(self, result, label, linestyle=None, color=None, linewidth=None):
        mr_fppi = MrFppi(result)
        score = self.score_method(mr_fppi) if self.append_score_to_label else None

        self.__helper.add_plot(mr_fppi.fppi, mr_fppi.mr, label=label, score=score,
                               linestyle=linestyle, color=color, linewidth=linewidth)
        return self


class MultiPlotHelper:
    def __init__(self, title, xlabel, ylabel):
        self.fig = Figure()
        self.ax = self.fig.add_subplot(1, 1, 1)

        if title:
            self.ax.set_title(title)

        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)

    def add_plot(self, x, y, label, score=None, linestyle=None, color=None, linewidth=None):
        if score is not None:
            label = '{:.3f} - {}'.format(score, label)

        self.ax.plot(x, y, linestyle=linestyle, label=label, color=color, linewidth=linewidth)
        self.ax.legend()
        return self
