"""Download the experiment videos from Google Drive into the Data/ folder.

Usage (from the project folder, with the venv active):
    python download_data.py

Files that are already downloaded completely are skipped, so it is safe to re-run.
Drive folder (manual download):
    https://drive.google.com/drive/folders/1KivCVjxuPRbE7q0vvnRarvejXnMS4Py8
"""
import sys
from pathlib import Path

try:
    import gdown
except ImportError:
    sys.exit("gdown is not installed. Run:  pip install -r requirements.txt")

DATA = Path(__file__).resolve().parent / "Data"

# name in Data/ -> (Google Drive file ID, expected size in bytes)
VIDEOS = {
    "Merengues-bowl_195bar_delrin_qs_vert.avi": ("1vEu1Hg49jzlqGypPphKqQFhble4JVU5W", 3_687_408_044),  # front view
    "Merengues bowl1_C001S0001.avi": ("1nolrGlUEGAWjgxlfjGGoWwvrdE95sQmZ", 1_342_206_464),             # top view
}


def main():
    DATA.mkdir(exist_ok=True)
    failed = []
    for name, (file_id, size) in VIDEOS.items():
        target = DATA / name
        if target.is_file() and target.stat().st_size == size:
            print(f"[skip] {name} is already downloaded")
            continue
        print(f"[download] {name} ({size / 1e9:.1f} GB)")
        gdown.download(id=file_id, output=str(target), quiet=False)
        if not target.is_file() or target.stat().st_size != size:
            failed.append(name)
            print(f"[error] {name} is missing or incomplete")
    if failed:
        sys.exit(
            "Some downloads failed. Re-run this script, or download them manually from\n"
            "https://drive.google.com/drive/folders/1KivCVjxuPRbE7q0vvnRarvejXnMS4Py8\n"
            f"and place them in {DATA}"
        )
    print("All data is in place.")


if __name__ == "__main__":
    main()
