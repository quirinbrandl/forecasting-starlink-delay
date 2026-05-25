from pathlib import Path

import joblib
import pandas as pd
import numpy as np
import argparse
from sklearn.preprocessing import StandardScaler


def load_data(path):
    df = pd.read_csv(
        path,
        parse_dates=[
            "timestamp-client-send",
            "timestamp-client-receive",
            "timestamp-server-send",
            "timestamp-server-receive",
        ],
    )
    return df


def index_dataset(df):
    timeline = pd.date_range(
        start=df["timestamp-client-send"].min(),
        end=df["timestamp-client-send"].max(),
        freq="1s",
    )

    df_indexed = pd.DataFrame({"timestamp": timeline})
    df_indexed = pd.merge(
        df_indexed,
        df,
        left_on="timestamp",
        right_on="timestamp-client-send",
        how="outer",
    )
    return df_indexed.set_index("timestamp")


def find_valid_indices(df, max_look_back_window_size, forecasting_steps):
    look_back_clean = (
        df.isna()
        .any(axis=1)
        .rolling(max_look_back_window_size, min_periods=max_look_back_window_size)
        .sum()
        .eq(0)
    )

    horizon_clean = (
        df["delay-rtt"]
        .isna()
        .rolling(forecasting_steps, min_periods=forecasting_steps)
        .sum()
        .eq(0)
        .shift(-forecasting_steps, fill_value=False)
    )
    valid_mask = look_back_clean & horizon_clean

    return np.flatnonzero(valid_mask)


def add_temporal_features(
    df,
    features=[
        {"feature_name": "15s", "period": 15},
        {"feature_name": "24h", "period": 86400},
    ],
):
    df_temporal = df.copy()

    time_steps = np.array([i for i in range(len(df))])

    for feature in features:
        df_temporal[f"{feature['feature_name']}-sin"] = np.sin(
            2 * np.pi * time_steps / feature["period"]
        )
        df_temporal[f"{feature['feature_name']}-cos"] = np.cos(
            2 * np.pi * time_steps / feature["period"]
        )

    return df_temporal


def split_indices(
    idcs, train_ratio, val_ratio, max_look_back_steps, max_forecasting_steps
):
    num_samples = len(idcs)
    full_window_length = max_look_back_steps + max_forecasting_steps
    num_possibly_overlapping_samples = full_window_length - 1

    # possibly overlapping samples are not used to prevent data leakage
    num_used_samples = num_samples - (2 * num_possibly_overlapping_samples)

    train_size = int(train_ratio * num_used_samples)
    val_size = int(val_ratio * num_used_samples)

    train_idcs = idcs[:train_size]

    val_start = train_size + num_possibly_overlapping_samples
    val_end = val_start + val_size
    val_idcs = idcs[val_start:val_end]

    test_idcs = idcs[val_end + num_possibly_overlapping_samples :]

    return train_idcs, val_idcs, test_idcs


def save_indices(train_idcs, val_idcs, test_idcs, output_dir):
    output_dir = Path(output_dir)

    pd.Series(train_idcs).to_csv(
        output_dir / "train_indices.csv", index=False, header=["valid_indices"]
    )
    pd.Series(val_idcs).to_csv(
        output_dir / "val_indices.csv", index=False, header=["valid_indices"]
    )
    pd.Series(test_idcs).to_csv(
        output_dir / "test_indices.csv", index=False, header=["valid_indices"]
    )


def normalize_features(df, features, val_start_idx):
    df_normalized = df.copy()
    training_data = df_normalized.iloc[:val_start_idx]
    scalers = {}

    for feature in features:
        scaler = StandardScaler()
        scaler.fit(training_data[[feature]])
        df_normalized[feature] = scaler.transform(df_normalized[[feature]])
        scalers[feature] = scaler

    return df_normalized, scalers


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_data_path",
        default="data/raw/data-rtt-stars-validation-month.2025-06.csv",
    )
    parser.add_argument("--output_dir", default="data/processed")
    parser.add_argument("--num_forecasting_steps", type=int, default=15)
    parser.add_argument("--max_look_back_window_size", type=int, default=600)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    print("Loading raw data...")
    df = load_data(args.raw_data_path)

    print("Indexing dataset...")
    df = index_dataset(df)

    df = df[["delay-rtt", "delay-send", "lost"]]

    print("Adding temporal features...")
    df = add_temporal_features(df)

    print("Identifying valid indices...")
    valid_idcs = find_valid_indices(
        df, args.max_look_back_window_size, args.num_forecasting_steps
    )

    print("Splitting dataset...")
    train_idcs, val_idcs, test_idcs = split_indices(
        valid_idcs,
        max_look_back_steps=args.max_look_back_window_size,
        max_forecasting_steps=args.num_forecasting_steps,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )

    print("Normalizing features...")
    df, scaler_dict = normalize_features(
        df, features=["delay-rtt", "delay-send"], val_start_idx=val_idcs[0]
    )

    print("Saving dataset, indices and scalers...")
    df.to_csv(output_dir / "processed_data.csv")
    save_indices(train_idcs, val_idcs, test_idcs, args.output_dir)
    joblib.dump(scaler_dict, output_dir / "scalers.pkl")
