"""Project folder locations, so the notebooks work wherever the repo is cloned.

Usage (from a notebook in Source/Main):
    from paths import DATA, OUTPUT, RESULTS, data_file
    video = cv2.VideoCapture(data_file("Merengues-bowl_195bar_delrin_qs_vert.avi"))
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # Source/Main/paths.py -> project root
DATA = ROOT / "Data"        # input videos and frames (downloaded, see README)
OUTPUT = ROOT / "Output"    # generated videos and images
RESULTS = ROOT / "Results"  # generated CSVs and figures

# Folders the notebooks write into; created automatically on first import.
for _folder in (
    OUTPUT,
    OUTPUT / "Images",
    RESULTS / "excel" / "frontview",
    RESULTS / "excel" / "topview",
):
    _folder.mkdir(parents=True, exist_ok=True)


def data_file(name):
    """Return the full path of an input file in Data/, or fail with a clear message.

    OpenCV does not raise an error for a missing file (VideoCapture just does not
    open, imread returns None), so check up front.
    """
    path = DATA / name
    if not path.is_file():
        raise FileNotFoundError(
            f"Input file not found: {path}\n"
            "Download the project data and place it in the Data/ folder "
            "(see 'Data setup' in README.md)."
        )
    return str(path)
