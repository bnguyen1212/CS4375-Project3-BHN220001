# Reproducibility Instructions

This project can be reproduced in either:
- Local VS Code/Jupyter (recommended for grading)
- Google Colab (requires Drive setup or repo clone)

## Option A: Local VS Code/Jupyter (Recommended)

1. Open this project folder in VS Code.
2. Create/activate a Python environment (if not already active).
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Open `project.ipynb`.
5. Select a local Python kernel from this project environment.
6. Run notebook cells in order from top to bottom.

Expected output artifacts (created by the last cell):
- `results/all_tuning_results.csv`
- `results/best_validation_configs.csv`
- `results/best_models_test_results.csv`

## Option B: Google Colab

Use this only if you are running the notebook with a Colab kernel.

### B1. Put the project files where Colab can access them
Choose one method:
1. Upload project folder to Google Drive under `MyDrive`, or
2. Clone the repository in Colab runtime.

### B2. Open and run the notebook
1. Open `project.ipynb` in Colab.
2. Run Cell 2 first (Colab/Drive setup cell).
   - It mounts Drive and tries to locate `dataset_loaders.py`.
   - Confirm it prints `Using project root: ...`.
3. Run Cell 3 next (imports).
4. Continue running remaining cells in order.