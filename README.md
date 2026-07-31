# Forecasting Starlink Delay

A project for the LEO Satellites Seminar focusing on forecasting network latency over Starlink connections using a Temporal Convolutional Network (TCN). 

## Overview
This repository implements a Temporal Convolutional Network (TCN) as a direct forecaster to predict future Starlink delay sequences. It also includes a baseline persistence model and comprehensive evaluation/visualization tools.

## Structure
- **`config/`**: Configuration files (e.g., `config.yml`) for setting up hyperparameters, model types, and datasets.
- **`data/`** (not included in repo): Processed data, train/val/test splits, and scalers.
- **`notebooks/`**: Jupyter notebooks for evaluation plots (predictions vs. ground truth, per-step boxplots, etc.).
- **`src/`**: Source code for the project:
  - `data/`: Preprocessing and PyTorch dataset.
  - `models/`: Model architectures (TCN, Persistence).
  - `utils/`: Utilities for evaluation and visualization.

## Tech Stack
- **PyTorch & Lightning**: For building and training models.
- **Weights & Biases (wandb)**: For experiment tracking and artifact management.
- **Matplotlib & Seaborn**: For generating visualizations.

## Setup Environment
This project uses [`uv`](https://github.com/astral-sh/uv) for fast and reproducible Python dependency management.

1. Ensure `uv` is installed on your system.
2. Install the project dependencies and create a virtual environment by running:
   ```bash
   uv sync
   ```
3. Activate the virtual environment:
   ```bash
   source .venv/bin/activate
   ```

## Usage
To train a TCN, first configure your hyperparameters in `config/config.yml`. 

Then, run the training script:
```bash
python src/train.py
```
For evaluation and visualizing the results (e.g., ground truth vs. predictions, per-step errors), explore the Jupyter notebooks inside the `notebooks/` directory.

## License
This project is open source and available under the [MIT License](LICENSE).
