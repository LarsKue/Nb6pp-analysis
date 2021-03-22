
import subprocess
from pathlib import Path


def main(argv: list) -> int:

    pdf_path = Path("plots")
    png_path = Path("plots_png")

    for pdf in pdf_path.glob("*.pdf"):

        png = png_path / pdf.stem

        subprocess.run(["pdftoppm", "-png", "-singlefile", str(pdf), str(png)])

    return 0


if __name__ == "__main__":
    import sys
    main(sys.argv)