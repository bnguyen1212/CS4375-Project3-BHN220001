# Reproducibility Instructions

Run this project either locally (VS Code/Jupyter) or in Google Colab.

## Local (VS Code/Jupyter)

1. Open this folder in VS Code.
2. Create and activate a virtual environment.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Open project.ipynb.
5. Select the notebook kernel from your project virtual environment.
6. Run all cells from top to bottom.

Notes:
- MNIST/CIFAR-10 are downloaded automatically by torchvision if missing.
- Results are saved under results/.

Expected CSV outputs:
- results/mlp_all_tuning_results.csv
- results/mlp_best_validation_configs.csv
- results/mlp_best_models_test_results.csv
- results/cnn_all_tuning_results.csv
- results/cnn_best_validation_configs.csv
- results/cnn_best_models_test_results.csv

## Google Colab

1. Open project.ipynb in Colab.
2. Run the setup cell that clones the repository and changes into the project directory.
3. Install dependencies in a Colab cell:

```python
%pip install -r requirements.txt
```

4. Run the remaining cells in order.

Colab notes:
- Runtime type: GPU is recommended (Runtime -> Change runtime type -> GPU).
- If the runtime restarts after package install, rerun cells from the top.