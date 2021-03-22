
from pathlib import Path

plots = Path("plots_new/")


def make_figure_string(name):
    base_path = Path("figures/bhusan/")
    path = base_path / name

    return r"\begin{figure}" + "\n" \
        r"\centering" + "\n" \
        r"\includegraphics[width=\linewidth]{" + str(path) + "}\n" \
        r"\end{figure}" + "\n"



with open("output.tex", "w+") as f:
    for p in plots.glob("*.png"):
        f.write(make_figure_string(p.name))
