**Project Name**
Impact Expoeriment Analysis in context to the HERA and DART Mission.

**Description**
- The project aims at the evaluation methodolody to analyse the impact experiments imitating the rubber pile asteroid bodies. 
- The Open CV module with KNN Background Substraction algorithm is equipped to track the moving ejecta and calculating its pixel loactions on real time. The further calibrations can yeild us the trajectroies and the velocity vs mass distrubution of the detected ejecta.
- The Ballistic Trajectorie are estimated and modified in the extracted images using the Image processing tools of the open CV module.
- The Pre and Post impact location analysis is performed using the Open CV module.

**Badges**
- The hypervleocity impact experiment recorded videos are stored in the ../Data. The two videos of front view and Top view with Frames per second of 5000 and 500 respectively.

**Installation**
- The code works on the python libraries of version 3.9.0 and above with integration of Jupyter notebook. 



**Peoject Sturcture**

*Clone the Repository*

```bash
#Clone the repository and access the file
git clone https://github.com/jateenrawal1002/Boulder-Detection-Project.git
cd Boulder-Detection-Project
```
*Create and activate the virtual environment*
```bash 
#Windows (PowerShell) - tested on Python 3.9
py -3.9 -m venv .venv-boulder --prompt boulder-detection
.venv-boulder\Scripts\activate

#macOS/Linux
python3 -m venv .venv-boulder --prompt boulder-detection
source .venv-boulder/bin/activate
```
*Install the requirements*

```bash
pip install -r requirements.txt
```

*Data setup*

The experiment videos are too large for GitHub (1.3 GB and 3.5 GB), so they are stored separately.

1. Download the videos from: **<!-- TODO: add download link -->**
2. Place them in the `Data/` folder of the project. The three frame images are already included in the repository:
```
Boulder-Detection-Project/
└── Data/
    ├── Merengues-bowl_195bar_delrin_qs_vert.avi   (front view, 5000 fps) - download
    ├── Merengues bowl1_C001S0001.avi              (top view, 500 fps)    - download
    ├── Frames380.png                              (included)
    ├── Frames1023.png                             (included)
    └── Frames4000.png                             (included)
```
3. That's it, no paths need to be changed. All notebooks find their files through `Source/Main/paths.py`, which works out the project folder automatically. If a file is missing, the notebook stops with a message saying which file to download.

**Usage**

- Open the notebooks in `Source/Main/` and select the `.venv-boulder` interpreter / kernel.
- Generated files are written automatically and do not need to be downloaded:
  - `Output/` - processed videos and frame images
  - `Results/` - CSV data (`Results/excel/frontview`, `Results/excel/topview`) and figures











