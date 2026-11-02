# Boulder Detection Project

Impact experiment analysis in the context of the HERA and DART missions.

## Description

- The project develops an evaluation methodology to analyse impact experiments imitating rubble-pile asteroid bodies.
- OpenCV's KNN background subtraction is used to detect and track the moving ejecta and calculate their pixel locations frame by frame. Further calibration yields the trajectories and the velocity vs. mass distribution of the detected ejecta.
- Ballistic trajectories are estimated and drawn on the extracted images using OpenCV image-processing tools.
- Pre- and post-impact boulder location analysis is performed using OpenCV.

**Data:** two high-speed recordings of a hypervelocity impact experiment, a front view at 5000 fps and a top view at 500 fps (see [Data setup](#3-data-setup)).

## Project Structure

```
Boulder-Detection-Project/
├── Data/                     # input videos (downloaded) and frame images (included)
├── Output/                   # generated videos and images
├── Results/
│   └── excel/
│       ├── frontview/        # generated CSV data, front view
│       └── topview/          # generated CSV data, top view
├── Source/
│   ├── Main/                 # main notebooks
│   │   ├── Boulder and Projectile Detections_front view.ipynb   # ejecta detection, tracking and velocities (front view)
│   │   ├── Boulder detection_Top view.ipynb                     # ejecta detection and tracking (top view)
│   │   ├── Object_detection_projectile_top_view.ipynb           # projectile detection (top view)
│   │   ├── Calibration.ipynb                                    # calibration on the front-view video
│   │   ├── Location and Trajectory.ipynb                        # boulder locations and ballistic trajectories
│   │   ├── Velocity_analysis.ipynb                              # velocity plots
│   │   ├── tracker.py / tracker.ipynb                           # Euclidean distance object tracker
│   │   └── paths.py                                             # project folder locations
│   └── Additional/           # older script versions
├── download_data.py          # downloads the videos into Data/
└── requirements.txt
```

## Installation

Tested on Python 3.9 with Jupyter notebooks.

### 1. Clone the repository

```bash
git clone https://github.com/jateenrawal1002/Boulder-Detection-Project.git
cd Boulder-Detection-Project
```

### 2. Create the virtual environment and install the requirements

```bash
# Windows (PowerShell)
py -3.9 -m venv .venv-boulder --prompt boulder-detection
.venv-boulder\Scripts\activate

# macOS/Linux
python3 -m venv .venv-boulder --prompt boulder-detection
source .venv-boulder/bin/activate
```

```bash
pip install -r requirements.txt
```

### 3. Data setup

The experiment videos are too large for GitHub (1.3 GB and 3.5 GB), so they are stored on Google Drive.

Download them automatically (about 5 GB; files you already have are skipped):

```bash
python download_data.py
```

Or download them manually from the [Google Drive folder](https://drive.google.com/drive/folders/1KivCVjxuPRbE7q0vvnRarvejXnMS4Py8?usp=sharing) and place them in `Data/` with their names unchanged. The three frame images are already included in the repository:

```
Data/
├── Merengues-bowl_195bar_delrin_qs_vert.avi   (front view, 5000 fps) - download
├── Merengues bowl1_C001S0001.avi              (top view, 500 fps)    - download
├── Frames380.png                              (included)
├── Frames1023.png                             (included)
└── Frames4000.png                             (included)
```

No paths need to be changed. All notebooks find their files through `Source/Main/paths.py`, which works out the project folder automatically. If a file is missing, the notebook stops with a message saying which file to download.

## Usage

1. Open the notebooks in `Source/Main/` (VS Code or Jupyter).
2. Select the `.venv-boulder` interpreter / kernel.
3. Run the cells. Generated files are written automatically:
   - `Output/` - processed videos and frame images
   - `Results/` - CSV data (`Results/excel/frontview`, `Results/excel/topview`) and figures

While a video is processing, press `z` in the video window to stop early.
