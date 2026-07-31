import math
import subprocess


def run_training(
    run_name, look_back_window, hidden_channels, dropout, kernel_size, num_blocks
):
    cmd = [
        "uv",
        "run",
        "src/train.py",
        "--run_name",
        str(run_name),
        "--look_back_window_size",
        str(look_back_window),
        "--hidden_channels",
        str(hidden_channels),
        "--dropout",
        str(dropout),
        "--kernel_size",
        str(kernel_size),
        "--num_blocks",
        str(num_blocks),
    ]

    subprocess.run(cmd)


def calculate_min_required_blocks(look_back_window_size, kernel_size):
    return math.ceil(
        math.log2((look_back_window_size - 1) / (2 * (kernel_size - 1)) + 1)
    )


if __name__ == "__main__":
    look_back_window_sizes = [60]
    hidden_channels = [32, 64, 128]
    dropouts = [0.0, 0.1, 0.2]
    kernel_sizes = [3, 5, 7]

    completed_runs = {
        (32, 0.2, 5),
        (32, 0.1, 5),
        (32, 0.0, 5),
        (32, 0.1, 7),
        (64, 0.1, 3),
        (32, 0.2, 7),
        (32, 0.1, 3),
        (32, 0.0, 7),
        (64, 0.0, 5),
        (32, 0.2, 3),
        (64, 0.1, 5),
        (32, 0.0, 3),
        (64, 0.0, 3),
        (64, 0.0, 7),
    }

    run_idx = 0 

    for look_back_window_size in look_back_window_sizes:
        for hidden_channel in hidden_channels:
            for dropout in dropouts:
                for kernel_size in kernel_sizes:
                    
                    current_combo = (hidden_channel, dropout, kernel_size)
                    if current_combo in completed_runs:
                        print(f"Skipping completed run: HC={hidden_channel}, Drop={dropout}, KS={kernel_size}")
                        continue
                    
                    print(f"Starting run {run_idx}: HC={hidden_channel}, Drop={dropout}, KS={kernel_size}")
                    run_training(
                        run_name=f"lbw_analysis_run_{run_idx}",
                        look_back_window=look_back_window_size,
                        hidden_channels=hidden_channel,
                        dropout=dropout,
                        kernel_size=kernel_size,
                        num_blocks=calculate_min_required_blocks(
                            look_back_window_size, kernel_size
                        ),
                    )
                    run_idx += 1