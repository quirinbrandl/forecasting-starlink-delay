import tempfile
from pathlib import Path

import joblib
import lightning as L
import pandas as pd
import torch
from torch.utils.data import DataLoader

import wandb
from data.dataset import DelayDataset
from models.models import PersistenceModel, TCNDirectForecaster


def get_feature_selection_results(wandb_project):
    api = wandb.Api()
    runs = api.runs(f"{wandb_project}")

    data = []
    for run in runs:
        if "feature" in run.name:
            config = run.config
            summary = run.summary

            data.append(
                {
                    "run_id": run.id,
                    "run_name": run.name,
                    "covariates": config["covariates"],
                    "rmse": summary["RMSE"],
                }
            )

    df = pd.DataFrame(data)

    return df


def get_lookback_window_results(wandb_project):
    api = wandb.Api()
    runs = api.runs(wandb_project)

    data = []
    for run in runs:
        if "lbw" in run.name:
            config = run.config
            summary = run.summary

            data.append(
                {
                    "run_id": run.id,
                    "run_name": run.name,
                    "look_back_window_size": config["look_back_window_size"],
                    "kernel_size": config["kernel_size"],
                    "hidden_channels": config["hidden_channels"],
                    "dropout": config["dropout"],
                    "rmse": summary.get("RMSE"),
                }
            )

    df = pd.DataFrame(data)

    if not df.empty:
        best_indices = df.groupby("look_back_window_size")["rmse"].idxmin()
        df = (
            df.loc[best_indices]
            .sort_values(by="look_back_window_size")
            .reset_index(drop=True)
        )

    return df


def test_model(
    wandb_project,
    model_artifact_ref,
    test_dataset_path,
    test_indices_path,
    look_back_window_size,
    covariate_set,
    num_workers=6,
):
    api = wandb.Api()
    full_artifact_path = f"{wandb_project}/{model_artifact_ref}:best"
    artifact = api.artifact(full_artifact_path, type="model")

    with tempfile.TemporaryDirectory() as temp_dir:
        artifact_dir = artifact.download(root=temp_dir)
        ckpt_path = Path(artifact_dir) / "model.ckpt"
        model = TCNDirectForecaster.load_from_checkpoint(ckpt_path)

    test_set = DelayDataset(
        dataset_path=test_dataset_path,
        indices_path=test_indices_path,
        look_back_window_size=look_back_window_size,
        forecast_horizon_size=15,
        feature_set=covariate_set + ["delay-rtt"],
    )
    test_loader = DataLoader(
        test_set, batch_size=1, num_workers=num_workers, shuffle=False
    )

    trainer = L.Trainer(logger=False, enable_checkpointing=False)
    trainer.test(model=model, dataloaders=test_loader)


def test_persistence_model(
    test_dataset_path,
    test_indices_path,
    scaler_path,
    num_workers=6,
):
    test_set = DelayDataset(
        dataset_path=test_dataset_path,
        indices_path=test_indices_path,
        look_back_window_size=1,
        forecast_horizon_size=15,
        feature_set=["delay-rtt"],
    )
    test_loader = DataLoader(
        test_set, batch_size=1, num_workers=num_workers, shuffle=False
    )

    scaler_dict = joblib.load(scaler_path)
    scaler = scaler_dict["delay-rtt"]
    mean = scaler.mean_[0]
    std = scaler.scale_[0]
    model = PersistenceModel(forecast_horizon=15, target_mean=mean, target_std=std)

    trainer = L.Trainer(logger=False, enable_checkpointing=False)
    trainer.test(model=model, dataloaders=test_loader)


def find_extreme_rmse_indices(
    wandb_project,
    model_artifact_ref,
    test_dataset_path,
    test_indices_path,
    look_back_window_size,
    covariate_set,
    n=5,
    find_largest=True,
    metric_type="mean",
    scaler_path="data/processed/scalers.pkl",
    num_workers=6,
):
    api = wandb.Api()
    full_artifact_path = f"{wandb_project}/{model_artifact_ref}:best"
    artifact = api.artifact(full_artifact_path, type="model")

    with tempfile.TemporaryDirectory() as temp_dir:
        artifact_dir = artifact.download(root=temp_dir)
        ckpt_path = Path(artifact_dir) / "model.ckpt"
        model = TCNDirectForecaster.load_from_checkpoint(ckpt_path)
    model.eval()

    test_set = DelayDataset(
        dataset_path=test_dataset_path,
        indices_path=test_indices_path,
        look_back_window_size=look_back_window_size,
        forecast_horizon_size=15,
        feature_set=covariate_set + ["delay-rtt"],
    )

    test_loader = DataLoader(
        test_set, batch_size=128, num_workers=num_workers, shuffle=False
    )

    scaler_dict = joblib.load(scaler_path)
    scaler = scaler_dict["delay-rtt"]
    target_mean = scaler.mean_[0]
    target_std = scaler.scale_[0]

    results = []
    current_idx = 0

    for x_batch, y_batch in test_loader:
        with torch.no_grad():
            preds = model(x_batch)

        preds_unnorm = preds * target_std + target_mean
        y_unnorm = y_batch * target_std + target_mean

        abs_err = torch.abs(preds_unnorm - y_unnorm)

        if metric_type == "mean":
            metric = torch.sqrt(torch.mean(abs_err**2, dim=1))
        elif metric_type == "max":
            metric = torch.max(abs_err, dim=1)[0]
        elif metric_type == "min":
            metric = torch.min(abs_err, dim=1)[0]
        else:
            raise ValueError("metric_type must be 'mean', 'max', or 'min'")

        for val in metric:
            results.append((current_idx, val.item()))
            current_idx += 1

    results.sort(key=lambda x: x[1], reverse=find_largest)

    return [idx for idx, val in results[:n]]


def get_per_step_absolute_errors(
    wandb_project,
    model_artifact_ref,
    test_dataset_path,
    test_indices_path,
    look_back_window_size,
    covariate_set,
    scaler_path="data/processed/scalers.pkl",
    num_workers=6,
):
    """
    Calculates the absolute error for every step of the forecasting horizon.
    Returns a list of 1D numpy arrays, where each array corresponds to the absolute errors across the test set for that specific horizon step.
    This format is ideal for passing to plt.boxplot().
    """
    api = wandb.Api()
    full_artifact_path = f"{wandb_project}/{model_artifact_ref}:best"
    artifact = api.artifact(full_artifact_path, type="model")

    with tempfile.TemporaryDirectory() as temp_dir:
        artifact_dir = artifact.download(root=temp_dir)
        ckpt_path = Path(artifact_dir) / "model.ckpt"
        model = TCNDirectForecaster.load_from_checkpoint(ckpt_path)
    model.eval()

    test_set = DelayDataset(
        dataset_path=test_dataset_path,
        indices_path=test_indices_path,
        look_back_window_size=look_back_window_size,
        forecast_horizon_size=15,
        feature_set=covariate_set + ["delay-rtt"],
    )

    test_loader = DataLoader(
        test_set, batch_size=128, num_workers=num_workers, shuffle=False
    )

    scaler_dict = joblib.load(scaler_path)
    scaler = scaler_dict["delay-rtt"]
    target_mean = scaler.mean_[0]
    target_std = scaler.scale_[0]

    all_abs_errors = []

    for x_batch, y_batch in test_loader:
        with torch.no_grad():
            preds = model(x_batch)

        preds_unnorm = preds * target_std + target_mean
        y_unnorm = y_batch * target_std + target_mean

        abs_err = torch.abs(preds_unnorm - y_unnorm)
        all_abs_errors.append(abs_err)

    all_errs = torch.cat(all_abs_errors, dim=0).numpy()

    return [all_errs[:, i] for i in range(all_errs.shape[1])]
