# AI CD Metrology - Team Local Setup

## Repository

GitHub: <https://github.com/junseo0im/ai-cd-metrology>

Dashboard sharing branch: `phase-a-skeleton`

```powershell
git clone https://github.com/junseo0im/ai-cd-metrology.git
cd ai-cd-metrology
git fetch
git checkout phase-a-skeleton
```

For an existing clone:

```powershell
git fetch
git checkout phase-a-skeleton
git pull
```

## Dataset

The GitHub repository and raw dataset are distributed separately. Copy the shared
Drive dataset into the repository without changing file names or folder hierarchy.

```text
ai-cd-metrology/
├─ data/
│  ├─ raw/
│  │  └─ ...
│  └─ derived/
│     └─ dataset_audit_v2/
│        └─ dataset_inventory.csv
├─ dashboard/
├─ src/
├─ configs/
├─ docs/
└─ tests/
```

The Dashboard reads images below `data/raw/` and the inventory at
`data/derived/dataset_audit_v2/dataset_inventory.csv`. Dataset and derived files
are excluded from Git by `.gitignore`; obtain them through the team data-sharing
channel. Do not rename or reorganize them.

## Python environment (Windows PowerShell)

Python 3.11 or newer is required.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

This installs the project, test dependency, and Streamlit. A separate
`pip install streamlit` is not required.

## Test

```powershell
python -m pytest -q
```

At this checkpoint the expected result is `68 passed`; the durable requirement is
that all tests pass.

## Dashboard

```powershell
python -m streamlit run dashboard/app.py
```

Open <http://localhost:8501>. If Streamlit shows its first-run email prompt, leave
the field blank and press Enter to skip it.

Available: dataset overview, selected-image analysis, frozen Method 2
Gradient/Edge CD metrology, Outer/Inner/Gap pixel results, separate dataset and
measurement statuses, ROI/E1-E6 traceability overlay, reliability/local-band
diagnostics, common structured results, method-comparison skeleton, defect
metadata view, and wafer hierarchy view.

Pending: official ruler calibration, µm results, manual-reference accuracy,
Method 1 and Method 3 adapters, defect detection/classification, geometric wafer
map, What-if simulation, and a Process Hypothesis model. The Dashboard does not
generate fake measurements, calibration values, references, or predictions.

## Method 2 baseline

The frozen Method 2 production implementation is the current comparison baseline;
do not change it casually. Add future Method 1/Method 3 results through adapters
and the common evaluation/comparison contract rather than coupling them to Method 2.
