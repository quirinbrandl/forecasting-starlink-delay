import argparse
import subprocess

import wandb
import yaml


def run_training_and_get_rmse(wandb_project, run_name, covariate_set):
    covariates_list = [
        covariate_element
        for covariate in covariate_set
        for covariate_element in covariate
    ]

    cmd = [
        "uv",
        "run",
        "src/train.py",
        "--run_name",
        run_name,
        "--covariates",
    ] + covariates_list

    subprocess.run(cmd)

    api = wandb.Api()
    runs = api.runs(wandb_project, filters={"display_name": run_name})
    rmse = runs[0].summary.get("RMSE")

    return rmse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="config/config.yml", help="Path to config file"
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    covariates_to_consider = {
        ("delay-send",),
        ("15s-cos", "15s-sin"),
        ("24h-sin", "24h-cos"),
        ("lost",),
    }

    wandb_project = config["wandb_project"]

    last_round_best_rmse = float("inf")
    curr_round_best_rmse = run_training_and_get_rmse(
        wandb_project=wandb_project,
        run_name="feature_selection_baseline",
        covariate_set=[],
    )
    curr_covariate_set = set()
    run_idx = 1

    while (
        curr_covariate_set != covariates_to_consider
        and curr_round_best_rmse < last_round_best_rmse
    ):
        last_round_best_rmse = curr_round_best_rmse
        curr_round_best_rmse = float("inf")
        remaining_covariates = covariates_to_consider - curr_covariate_set

        for covariate in remaining_covariates:
            rmse = run_training_and_get_rmse(
                wandb_project=wandb_project,
                run_name=f"feature_selection_run_{run_idx}",
                covariate_set=(curr_covariate_set | {covariate}),
            )
            if rmse < curr_round_best_rmse:
                curr_round_best_rmse = rmse
                best_covariate = covariate

            run_idx += 1

        if curr_round_best_rmse < last_round_best_rmse:
            curr_covariate_set.add(best_covariate)
