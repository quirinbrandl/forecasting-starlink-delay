import lightning as L
from torch import optim
import torch
import torch.nn as nn


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, : -self.chomp_size].contiguous()


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()

        self.padding = (kernel_size - 1) * dilation

        self.conv = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=self.padding,
                dilation=dilation,
            )
        )

        self.chomp = Chomp1d(chomp_size=self.padding)

    def forward(self, x):
        return self.chomp(self.conv(x))


class TemporalBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, dilation_base, block_idx, dropout
    ):
        super().__init__()

        layers = []

        for i in range(2):
            layers.append(
                CausalConv1d(
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation_base**block_idx,
                )
            )
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        self.net = nn.Sequential(*layers)

        self.residual_connection = (
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
            if in_channels != out_channels
            else None
        )

    def forward(self, x):
        return self.net(x) + (
            self.residual_connection(x) if self.residual_connection else x
        )


class TCN(nn.Module):
    def __init__(
        self,
        num_features,
        hidden_channels,
        kernel_size,
        num_blocks,
        dilation_base,
        dropout,
    ):
        super().__init__()

        layers = []

        for i in range(num_blocks):
            layers.append(
                TemporalBlock(
                    in_channels=num_features if i == 0 else hidden_channels,
                    out_channels=hidden_channels,
                    kernel_size=kernel_size,
                    dilation_base=dilation_base,
                    dropout=dropout,
                    block_idx=i,
                )
            )

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class TCNDirectForecaster(L.LightningModule):
    def __init__(
        self,
        num_features,
        hidden_channels,
        kernel_size,
        num_blocks,
        dilation_base,
        dropout,
        forecast_horizon,
        learning_rate,
        target_mean,
        target_std,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.tcn = TCN(
            num_features=num_features,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            num_blocks=num_blocks,
            dilation_base=dilation_base,
            dropout=dropout,
        )
        self.linear_predictor = nn.Linear(hidden_channels, forecast_horizon)
        self.loss_fn = nn.MSELoss()
        self.learning_rate = learning_rate

        self.register_buffer(
            "target_mean", torch.tensor(target_mean, dtype=torch.float32)
        )
        self.register_buffer(
            "target_std", torch.tensor(target_std, dtype=torch.float32)
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        last_hidden_state = self.tcn(x)[:, :, -1]
        return self.linear_predictor(last_hidden_state)

    def training_step(self, batch, batch_idx):
        x, y = batch
        predictions = self(x)
        loss = self.loss_fn(predictions, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        predictions = self(x)
        loss = self.loss_fn(predictions, y)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        predictions = self(x)

        predictions_unnormalized = predictions * self.target_std + self.target_mean
        y_unnormalized = y * self.target_std + self.target_mean

        mse_unnormalized = torch.nn.functional.mse_loss(
            predictions_unnormalized, y_unnormalized
        )
        rmse_unnormalized = torch.sqrt(mse_unnormalized)

        self.log("MSE", mse_unnormalized)
        self.log("RMSE", rmse_unnormalized)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)


class PersistenceModel(L.LightningModule):
    def __init__(self, forecast_horizon, target_mean, target_std):
        super().__init__()
        self.forecast_horizon = forecast_horizon

        self.register_buffer(
            "target_mean", torch.tensor(target_mean, dtype=torch.float32)
        )
        self.register_buffer(
            "target_std", torch.tensor(target_std, dtype=torch.float32)
        )

    def forward(self, x):
        if x.size(dim=2) != 1:
            raise ValueError("PersistenceModel can not handle covariates.")

        last_known_value = x[:, -1, 0]
        predictions = last_known_value.unsqueeze(1).repeat(1, self.forecast_horizon)

        return predictions

    def test_step(self, batch, batch_idx):
        x, y = batch
        predictions = self(x)

        predictions_unnormalized = predictions * self.target_std + self.target_mean
        y_unnormalized = y * self.target_std + self.target_mean

        mse_unnormalized = torch.nn.functional.mse_loss(
            predictions_unnormalized, y_unnormalized
        )
        rmse_unnormalized = torch.sqrt(mse_unnormalized)

        self.log("MSE", mse_unnormalized)
        self.log("RMSE", rmse_unnormalized)
