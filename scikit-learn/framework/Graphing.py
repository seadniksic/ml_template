import seaborn


class Graphing:
    def __init__(self):
        pass

    def line_plot_2D(self, x, y, xlabel="x", ylabel="y"):
        ax = seaborn.lineplot(x=x, y=y)
        ax.set(xlabel=xlabel, ylabel=ylabel)
        return ax
