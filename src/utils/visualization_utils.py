import tempfile
from pathlib import Path

import joblib
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import cm, ticker

import wandb
from data.dataset import DelayDataset
from models.models import TCNDirectForecaster
from utils.evaluation_utils import get_per_step_absolute_errors


def set_visualization_style():
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": "Times New Roman",
            "axes.labelsize": 16,
            "axes.titlesize": 18,
            "font.size": 16,
            "legend.fontsize": 16,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def plot_feature_selection():
    set_visualization_style()

    data = [
        {"Step": 0, "Candidate": "Baseline", "RMSE": 6.825969, "Selected": True},
        {"Step": 1, "Candidate": "+ 15s Cycle", "RMSE": 6.769010, "Selected": True},
        {"Step": 1, "Candidate": "+ delay-send", "RMSE": 6.804429, "Selected": False},
        {"Step": 1, "Candidate": "+ 24h Cycle", "RMSE": 6.809479, "Selected": False},
        {"Step": 1, "Candidate": "+ lost", "RMSE": 6.823690, "Selected": False},
        {"Step": 2, "Candidate": "+ 24h Cycle", "RMSE": 6.750632, "Selected": True},
        {"Step": 2, "Candidate": "+ delay-send", "RMSE": 6.759192, "Selected": False},
        {"Step": 2, "Candidate": "+ lost", "RMSE": 6.759411, "Selected": False},
        {"Step": 3, "Candidate": "+ lost", "RMSE": 6.764220, "Selected": False},
        {"Step": 3, "Candidate": "+ delay-send", "RMSE": 6.765065, "Selected": False},
    ]

    df = pd.DataFrame(data)

    plt.figure(figsize=(8.5, 5.5))

    selected = df[df["Selected"]]
    sns.lineplot(
        data=selected,
        x="Step",
        y="RMSE",
        marker="o",
        markersize=9,
        color="#1f77b4",
        linewidth=2.5,
        label="Selected Path",
        zorder=3,
    )

    rejected = df[~df["Selected"]]
    sns.scatterplot(
        data=rejected,
        x="Step",
        y="RMSE",
        color="#7f7f7f",
        marker="o",
        s=70,
        alpha=0.7,
        label="Rejected",
        zorder=2,
    )

    for index, row in rejected.iterrows():
        x_offset = 8
        y_offset = 0
        va = "center"

        if row["Step"] == 1 and "lost" in row["Candidate"]:
            y_offset = 4
            va = "bottom"
        elif row["Step"] == 1 and "delay-send" in row["Candidate"]:
            y_offset = -4
            va = "top"

        elif row["Step"] == 2 and "lost" in row["Candidate"]:
            y_offset = 4
            va = "bottom"
        elif row["Step"] == 2 and "delay-send" in row["Candidate"]:
            y_offset = -4
            va = "top"

        elif row["Step"] == 3 and "delay-send" in row["Candidate"]:
            y_offset = 4
            va = "bottom"
        elif row["Step"] == 3 and "lost" in row["Candidate"]:
            y_offset = -4
            va = "top"

        plt.annotate(
            row["Candidate"],
            xy=(row["Step"], row["RMSE"]),
            xytext=(x_offset, y_offset),
            textcoords="offset points",
            color="#555555",
            fontsize=14,
            va=va,
            ha="left",
        )

    for index, row in selected.iterrows():
        x_offset = 8
        y_offset = 0
        va = "center"

        if row["Step"] == 1:
            y_offset = 8
            va = "bottom"
        elif row["Step"] == 2:
            y_offset = -8
            va = "top"
        else:
            y_offset = 6
            va = "bottom"

        plt.annotate(
            row["Candidate"],
            xy=(row["Step"], row["RMSE"]),
            xytext=(x_offset, y_offset),
            textcoords="offset points",
            color="#1f77b4",
            fontsize=14,
            fontweight="bold",
            va=va,
            ha="left",
            zorder=6,
        )

    optimal_step = 2
    optimal_rmse = selected[selected["Step"] == optimal_step]["RMSE"].values[0]
    plt.plot(
        optimal_step, optimal_rmse, marker="o", color="#d62728", markersize=13, zorder=4
    )

    plt.annotate(
        f"Optimal:\n{optimal_rmse:.4f}",
        xy=(optimal_step, optimal_rmse),
        xytext=(0, -48),
        textcoords="offset points",
        ha="center",
        va="top",
        fontsize=14,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9),
        arrowprops=dict(arrowstyle="-|>", color="gray", lw=1.2),
        zorder=5,
    )

    plt.title("Forward Feature Selection Progression")
    plt.ylabel("Validation RMSE (ms)")

    plt.xticks(
        ticks=[0, 1, 2, 3],
        labels=[
            "Step 0:\nBaseline",
            "Step 1:\n1 Feature",
            "Step 2:\n2 Features",
            "Step 3:\n3 Features",
        ],
    )

    plt.legend(loc="upper right", frameon=True)

    plt.xlim(-0.2, 3.5)
    plt.ylim(6.720, 6.835)

    plt.tight_layout()
    plt.savefig("notebooks/plots/feature_selection_progression.pdf")
    plt.show()


def plot_lookback_analysis():
    set_visualization_style()

    lbw_data = pd.DataFrame(
        {
            "Look-Back Window": [30, 60, 120, 300],
            "RMSE": [6.763783, 6.732739, 6.736427, 6.726470],
        }
    )

    plt.figure(figsize=(7, 4.5))

    ax = sns.lineplot(
        data=lbw_data,
        x="Look-Back Window",
        y="RMSE",
        marker="o",
        markersize=8,
        color="#2ca02c",
        linewidth=2,
    )

    plt.xticks(lbw_data["Look-Back Window"].values)
    plt.title("Impact of Look-Back Window Size on Forecasting Accuracy")
    plt.xlabel("Look-Back Window Size k (seconds)")
    plt.ylabel("Validation RMSE (ms)")

    best_idx = lbw_data["RMSE"].idxmin()
    best_x = lbw_data.loc[best_idx, "Look-Back Window"]
    best_y = lbw_data.loc[best_idx, "RMSE"]

    plt.plot(best_x, best_y, marker="o", color="#d62728", markersize=12, zorder=5)

    plt.annotate(
        f"Optimal:\n{best_y:.4f}",
        xy=(best_x, best_y),
        xytext=(0, 20),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=14,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9),
    )

    plt.ylim(6.72, 6.775)

    plt.tight_layout()
    plt.savefig("notebooks/plots/lookback_window_analysis.pdf")
    plt.show()


def plot_final_evaluation():
    set_visualization_style()

    eval_data = pd.DataFrame(
        {"Model": ["Persistence Baseline", "Optimized TCN"], "RMSE": [9.2514, 7.5927]}
    )

    baseline_val = eval_data.loc[0, "RMSE"]
    tcn_val = eval_data.loc[1, "RMSE"]
    improvement = ((baseline_val - tcn_val) / baseline_val) * 100

    plt.figure(figsize=(6, 4.5))

    colors = ["#7f7f7f", "#1f77b4"]
    ax = sns.barplot(
        data=eval_data,
        x="Model",
        y="RMSE",
        hue="Model",
        palette=colors,
        edgecolor="black",
        linewidth=1.2,
        legend=False,
    )

    plt.title("Final Model Evaluation on Test Set")
    plt.ylabel("Test RMSE (ms)")
    plt.xlabel("")

    for i, p in enumerate(ax.patches):
        height = p.get_height()

        ax.annotate(
            f"{height:.4f} ms",
            xy=(p.get_x() + p.get_width() / 2.0, height),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="bold",
        )

        if i == 1:
            ax.annotate(
                f"(-{improvement:.2f}%)",
                xy=(p.get_x() + p.get_width() / 2.0, height),
                xytext=(0, 20),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=14,
                fontweight="bold",
                color="#d62728",
            )

    plt.ylim(0, 10.8)

    plt.tight_layout()
    plt.savefig("notebooks/plots/final_model_evaluation.pdf")
    plt.show()


def visualize_predictions(
    wandb_project,
    model_artifact_ref,
    test_dataset_path,
    test_indices_path,
    look_back_window_size,
    covariate_set,
    start_idx,
    num_horizons,
    scaler_path="data/processed/scalers.pkl",
    save_path="notebooks/plots/predictions_visualization.pdf",
    plot_overlapping_shifts=False,
    show_markers=True,
):
    set_visualization_style()

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

    scaler_dict = joblib.load(scaler_path)
    scaler = scaler_dict["delay-rtt"]
    target_mean = scaler.mean_[0]
    target_std = scaler.scale_[0]

    plt.figure(figsize=(10, 5))

    df = pd.read_csv(test_dataset_path)
    timestamps = pd.to_datetime(df["timestamp"])

    available_indices = {test_set.indices[i].item(): i for i in range(len(test_set))}
    current_df_idx = test_set.indices[start_idx].item()

    if plot_overlapping_shifts:
        max_total_steps = 15 * num_horizons + 15
        target_df_indices = list(
            range(current_df_idx + 1, current_df_idx + 1 + max_total_steps)
        )

        all_y_true = []
        all_time_steps = []
        for t in target_df_indices:
            if t < len(df):
                y_val = df["delay-rtt"].iloc[t]
                y_unnorm = y_val * target_std + target_mean
                all_y_true.append(y_unnorm)
                all_time_steps.append(timestamps.iloc[t])

        plt.close()
        fig, axes = plt.subplots(5, 3, figsize=(8.5, 12), sharex=True, sharey=True)
        axes = axes.flatten()
        start_time_str = (
            str(all_time_steps[0]).split("+")[0].split(".")[0].replace("T", " ")
        )
        start_timestamp = all_time_steps[0]
        all_time_steps = [
            (ts - start_timestamp).total_seconds() for ts in all_time_steps
        ]

        for shift in range(15):
            ax = axes[shift]
            color = cm.viridis(shift / 14.0)

            ax.plot(
                all_time_steps,
                all_y_true,
                color="red",
                linestyle="--",
                alpha=0.5,
                linewidth=1,
                label="Ground Truth",
            )

            for h in range(num_horizons):
                target_df_idx = current_df_idx + shift + h * 15

                if target_df_idx not in available_indices:
                    continue

                test_idx = available_indices[target_df_idx]
                x, _ = test_set[test_idx]

                with torch.no_grad():
                    pred = model(x.unsqueeze(0)).squeeze(0)
                pred_unnorm = (pred * target_std + target_mean).numpy()

                block_ts = timestamps.iloc[target_df_idx + 1 : target_df_idx + 16]
                block_ts = [(ts - start_timestamp).total_seconds() for ts in block_ts]

                ax.plot(
                    block_ts,
                    pred_unnorm,
                    color=color,
                    marker="o" if show_markers else None,
                    markersize=1.5,
                    label="Prediction" if h == 0 else "",
                    alpha=0.7,
                )

                ax.axvline(x=block_ts[0], color="gray", linestyle=":", alpha=0.4)

            ax.set_title(f"Starting Point Offset {shift}", fontsize=14)

            if shift >= 12:
                ax.set_xlabel("Time Step (s)")
            if shift % 3 == 0:
                ax.set_ylabel("Delay RTT (ms)")

        fig.suptitle(
            f"Forecasts for All Starting Points vs Ground Truth\n(Start: {start_time_str})"
        )

        for ax in axes[-3:]:
            ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4))

        handles, labels = axes[0].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if "Ground Truth" in by_label and "Prediction" in by_label:
            fig.legend(
                [by_label["Ground Truth"], by_label["Prediction"]],
                ["Ground Truth", "Prediction"],
                loc="lower center",
                ncol=2,
                fontsize="medium",
                bbox_to_anchor=(0.5, 0.0),
            )

        plt.tight_layout(rect=[0, 0.05, 1, 0.96])

        if save_path:
            plt.savefig(save_path)
        plt.show()
        return

    all_y_true = []
    all_time_steps = []
    actual_horizons_plotted = 0
    start_timestamp = timestamps.iloc[current_df_idx + 1]

    for h in range(num_horizons):
        target_df_idx = current_df_idx + h * 15

        if target_df_idx not in available_indices:
            print(
                f"Stopping at horizon {h}: next contiguous window (df row {target_df_idx}) was skipped due to NaNs."
            )
            break

        test_idx = available_indices[target_df_idx]

        x, y = test_set[test_idx]
        x_batch = x.unsqueeze(0)

        with torch.no_grad():
            pred = model(x_batch).squeeze(0)

        pred_unnorm = pred * target_std + target_mean
        y_unnorm = y * target_std + target_mean

        horizon_timestamps = timestamps.iloc[target_df_idx + 1 : target_df_idx + 16]
        horizon_ts_sec = [
            (ts - start_timestamp).total_seconds() for ts in horizon_timestamps
        ]

        if h == 0:
            plt.plot(
                horizon_ts_sec,
                pred_unnorm.numpy(),
                color="#d62728",
                marker="o" if show_markers else None,
                markersize=2,
                label="Predicted Horizon",
                alpha=0.7,
            )
        else:
            plt.plot(
                horizon_ts_sec,
                pred_unnorm.numpy(),
                color="#d62728",
                marker="o" if show_markers else None,
                markersize=2,
                alpha=0.7,
            )

        plt.axvline(x=horizon_ts_sec[0], color="gray", linestyle="--", alpha=0.5)

        all_y_true.extend(y_unnorm.numpy())
        all_time_steps.extend(horizon_ts_sec)
        actual_horizons_plotted += 1

    if actual_horizons_plotted > 0:
        plt.plot(
            all_time_steps,
            all_y_true,
            color="#1f77b4",
            linestyle="-",
            marker="o" if show_markers else None,
            markersize=2,
            label="Ground Truth",
            zorder=1,
        )

        start_time_str = (
            str(start_timestamp).split("+")[0].split(".")[0].replace("T", " ")
        )
        plt.title(f"Forecasting Predictions vs Ground Truth\n(Start: {start_time_str})")
        plt.xlabel("Time Step (s)")
        plt.ylabel("Delay RTT (ms)")
        plt.legend(loc="upper left")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()
    else:
        print("No horizons could be plotted.")


def visualize_ground_truth(
    dataset_path,
    start_idx,
    num_steps,
    scaler_path="data/processed/scalers.pkl",
    save_path="notebooks/plots/ground_truth_visualization.pdf",
):
    set_visualization_style()

    scaler_dict = joblib.load(scaler_path)
    scaler = scaler_dict["delay-rtt"]
    target_mean = scaler.mean_[0]
    target_std = scaler.scale_[0]

    plt.figure(figsize=(10, 5))

    df = pd.read_csv(dataset_path)
    timestamps = pd.to_datetime(df["timestamp"])

    if start_idx + num_steps > len(df):
        print("Warning: num_steps exceeds the dataset length. Truncating.")
        num_steps = len(df) - start_idx

    y_scaled = df["delay-rtt"].iloc[start_idx : start_idx + num_steps].values
    y_unnorm = y_scaled * target_std + target_mean

    horizon_timestamps = timestamps.iloc[start_idx : start_idx + num_steps]
    start_timestamp = horizon_timestamps.iloc[0]
    horizon_ts_sec = [
        (ts - start_timestamp).total_seconds() for ts in horizon_timestamps
    ]

    plt.plot(
        horizon_ts_sec,
        y_unnorm,
        color="#1f77b4",
        linestyle="-",
        marker="o",
        markersize=2,
        label="Ground Truth",
        zorder=1,
    )

    start_time_str = str(start_timestamp).split("+")[0].split(".")[0].replace("T", " ")
    plt.title(f"Ground Truth Data\n(Start: {start_time_str})")
    plt.xlabel("Time Step (s)")
    plt.ylabel("Delay RTT (ms)")
    plt.legend(loc="upper left")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def visualize_per_step_boxplot(
    wandb_project,
    model_artifact_ref,
    test_dataset_path,
    test_indices_path,
    look_back_window_size,
    covariate_set,
    scaler_path="data/processed/scalers.pkl",
    save_path="notebooks/plots/per_step_boxplot.pdf",
    num_workers=6,
    show_max_text=True,
):
    set_visualization_style()

    box_data = get_per_step_absolute_errors(
        wandb_project=wandb_project,
        model_artifact_ref=model_artifact_ref,
        test_dataset_path=test_dataset_path,
        test_indices_path=test_indices_path,
        look_back_window_size=look_back_window_size,
        covariate_set=covariate_set,
        scaler_path=scaler_path,
        num_workers=num_workers,
    )

    plt.figure(figsize=(12, 6))

    plt.boxplot(
        box_data,
        patch_artist=True,
        showmeans=True,
        meanline=True,
        showfliers=not show_max_text,
        boxprops=dict(facecolor="#1f77b4", color="black"),
        capprops=dict(color="black"),
        whiskerprops=dict(color="black"),
        flierprops=dict(marker="o", color="black", markersize=2, alpha=0.3),
        medianprops=dict(color="red", linewidth=1.5),
        meanprops=dict(color="orange", linewidth=1.5, linestyle="--"),
    )

    mean_line = mlines.Line2D(
        [], [], color="orange", linewidth=2.5, linestyle="--", label="Mean"
    )
    median_line = mlines.Line2D([], [], color="red", linewidth=1.5, label="Median")
    plt.legend(
        handles=[mean_line, median_line], loc="upper right", bbox_to_anchor=(0.99, 0.75)
    )

    plt.title("Absolute Error Distribution per Forecasting Step")
    plt.xlabel("Forecasting Step (Horizon)")
    plt.ylabel("Absolute Error (ms)")

    plt.xticks(range(1, len(box_data) + 1), [i for i in range(1, len(box_data) + 1)])

    if show_max_text:
        y_min, y_max = plt.ylim()
        plt.ylim(y_min, y_max + (y_max - y_min) * 0.15)

        for i, data in enumerate(box_data):
            max_err = np.max(data)
            plt.text(
                i + 1,
                y_max + (y_max - y_min) * 0.02,
                f"Max:\n{max_err:.1f}",
                horizontalalignment="center",
                size="x-small",
                color="#d62728",
            )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
