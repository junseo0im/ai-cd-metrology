# Dashboard MVP

## Run locally

From the repository root:

```powershell
.venv\Scripts\python.exe -m streamlit run dashboard/app.py
```

The dashboard reads the existing dataset inventory and curated images. It does not run batch metrology. Frozen Method 2 runs only when the user selects an image and presses the analysis button.

## Available in the MVP

- Dataset and system overview from existing audit metadata
- Cascading curated-image filters
- Original image display
- Frozen Method 2 pixel measurement for a selected image
- Position-specific ROI, representative E1–E6 x markers and five-band overlay
- Whole-image CD result, status, failure reason and runtime
- Reliability and local-band diagnostic tables
- Common CD evaluation-contract view
- Method 1/2/3 comparison structure with Method 2 actual results only
- Defect and wafer metadata skeleton views

## Pending or not available

- Official calibration and µm results
- Manual reference and accuracy results
- Method 1 and Method 3 adapters
- Defect detection and classification algorithms
- Physical wafer coordinates and geometric wafer map
- What-if simulation algorithm
- Process-hypothesis or root-cause model

Pending and unavailable fields are displayed explicitly. The MVP does not create placeholder measurements, calibration values, predictions or operational decisions.
