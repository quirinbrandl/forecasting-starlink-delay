import argparse

import joblib
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
import wandb
import yaml

from data.dataset import DelayDataset
from models.models import PersistenceModel, TCNDirectForecaster


def parse_config():
    conf_parser = argparse.ArgumentParser()
    conf_parser.add_argument(
        "--config", default="config/config.yml", help="Path to config file"
    )

    args_config, remaining_argv = conf_parser.parse_known_args()

    with open(args_config.config, "r") as f:
        config_dict = yaml.safe_load(f)

    parser = argparse.ArgumentParser(description="TCN Training Script")

    parser.add_argument("--wandb_project")
    parser.add_argument("--run_name")
    parser.add_argument("--processed_data_path")
    parser.add_argument("--train_indices_path")
    parser.add_argument("--val_indices_path")
    parser.add_argument("--scaler_path")

    parser.add_argument("--look_back_window_size", type=int)
    parser.add_argument("--forecast_horizon", type=int)
    parser.add_argument("--covariates", nargs="*")

    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--max_epochs", type=int)
    parser.add_argument("--early_stopping_patience", type=int)
    parser.add_argument("--num_workers", type=int)

    parser.add_argument("--model")

    parser.add_argument("--hidden_channels", type=int)
    parser.add_argument("--kernel_size", type=int)
    parser.add_argument("--num_blocks", type=int)
    parser.add_argument("--dilation_base", type=int)
    parser.add_argument("--dropout", type=float)

    parser.set_defaults(**config_dict)
    args = parser.parse_args(remaining_argv)

    return args


def main():
    args = parse_config()
    
    L.seed_everything(42, workers=True)

    feature_set = args.covariates + ["delay-rtt"]
    train_set = DelayDataset(
        dataset_path=args.processed_data_path,
        indices_path=args.train_indices_path,
        look_back_window_size=args.look_back_window_size,
        forecast_horizon_size=args.forecast_horizon,
        feature_set=feature_set,
    )
    val_set = DelayDataset(
        dataset_path=args.processed_data_path,
        indices_path=args.val_indices_path,
        look_back_window_size=args.look_back_window_size,
        forecast_horizon_size=args.forecast_horizon,
        feature_set=feature_set,
    )

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        dataset=val_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        persistent_workers=True,
    )

    scaler_dict = joblib.load(args.scaler_path)
    scaler = scaler_dict["delay-rtt"]
    mean = scaler.mean_[0]
    std = scaler.scale_[0]

    model_name = args.model.lower()
    if model_name == "tcn":
        model = TCNDirectForecaster(
            num_features=len(feature_set),
            hidden_channels=args.hidden_channels,
            kernel_size=args.kernel_size,
            num_blocks=args.num_blocks,
            dilation_base=args.dilation_base,
            dropout=args.dropout,
            forecast_horizon=args.forecast_horizon,
            learning_rate=args.learning_rate,
            target_mean=float(mean),
            target_std=float(std),
        )
    elif model_name == "persistence":
        model = PersistenceModel(
            forecast_horizon=args.forecast_horizon,
            target_mean=mean,
            target_std=std,
        )
    else:
        raise ValueError(f"{model_name} is not a supported model")

    wandb_logger = WandbLogger(
        project=args.wandb_project,
        name=args.run_name,
        log_model="all",
        config=vars(args),
    )
    wandb_logger.watch(model, log="all")

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss", mode="min", save_top_k=1, filename="best_model"
    )
    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        patience=args.early_stopping_patience,
    )

    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stopping_callback],
    )

    if not isinstance(model, PersistenceModel):
        trainer.fit(model, train_loader, val_loader)
        trainer.test(model, dataloaders=val_loader, ckpt_path="best")
    else:
        trainer.test(model, dataloaders=val_loader)

    wandb.finish()


if __name__ == "__main__":
    main()
