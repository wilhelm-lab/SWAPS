import logging
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    average_precision_score,
)
import torch.nn.functional as F

Logger = logging.getLogger(__name__)


class CNNEncoder(nn.Module):
    def __init__(self, embed_dim=128):
        super(CNNEncoder, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=5, padding=2)
        self.fc = nn.Linear(256, embed_dim)  # Project to embedding space

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.mean(x, dim=-1)  # Global average pooling
        x = self.fc(x)
        return F.normalize(x, dim=-1)  # L2 normalization


class CNN1DModel(nn.Module):
    def __init__(self):
        super(CNN1DModel, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels=1, out_channels=16, kernel_size=25, stride=3, padding=2
        )
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(
            in_channels=16, out_channels=32, kernel_size=25, stride=2, padding=1
        )
        self.bn2 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(
            in_channels=32, out_channels=64, kernel_size=25, stride=1, padding=1
        )
        self.bn3 = nn.BatchNorm1d(64)
        self.pool = nn.AdaptiveAvgPool1d(1)  # Global Average Pooling
        self.fc = nn.Linear(32, 2)  # Output 2 units for binary classification (size 2)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension for Conv1D (1 channel)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool(x).squeeze(-1)  # Global Average Pooling, squeeze last dimension
        x = self.fc(x)  # Output logits for both classes (size 2)
        return x  # Return logits (no sigmoid needed)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=kernel_size // 2
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size, padding=kernel_size // 2
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # Residual connection
        return torch.relu(out)


class ResNet1D(nn.Module):
    def __init__(
        self, num_classes, num_blocks=3, initial_channels=32, dropout_rate=0.3
    ):
        super(ResNet1D, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout_rate = dropout_rate  # Store dropout rate
        in_channels = 1
        out_channels = initial_channels

        for _ in range(num_blocks):
            self.layers.append(ResBlock(in_channels, out_channels))
            in_channels = out_channels
            out_channels *= 2

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(initial_channels * (2 ** (num_blocks - 1)), num_classes)
        self.dropout = nn.Dropout(dropout_rate)  # Dropout before FC layer

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = F.dropout(
                x, p=self.dropout_rate, training=self.training
            )  # Apply dropout

        x = self.global_pool(x).squeeze(-1)
        x = self.dropout(x)  # Dropout before final FC layer
        return self.fc(x)


class TCN(nn.Module):
    def __init__(self, num_classes):
        super(TCN, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels=1, out_channels=32, kernel_size=5, dilation=1, padding=2
        )
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, dilation=2, padding=4)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=5, dilation=4, padding=8)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.global_pool(x).squeeze(-1)
        return self.fc(x)


class ConvAutoencoder1D(nn.Module):
    def __init__(self, input_channels=1, base_channels=16, num_layers=3, kernel_size=3):
        """
        A flexible 1D convolutional autoencoder that ensures input and output sizes match.

        Args:
            input_channels (int): Number of input channels (default: 1).
            base_channels (int): Number of filters in the first layer (default: 16).
            num_layers (int): Number of convolutional layers in the encoder (default: 3).
            kernel_size (int): Kernel size for convolutions (default: 3).
        """
        super(ConvAutoencoder1D, self).__init__()

        self.encoder_layers = nn.ModuleList()
        self.decoder_layers = nn.ModuleList()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, return_indices=True)
        self.unpool = nn.MaxUnpool1d(kernel_size=2, stride=2)

        in_channels = input_channels
        out_channels = base_channels

        # Encoder
        self.enc_indices = []  # Store indices for unpooling
        for _ in range(num_layers):
            self.encoder_layers.append(
                nn.Conv1d(
                    in_channels, out_channels, kernel_size, padding=kernel_size // 2
                )
            )
            in_channels = out_channels
            out_channels *= 2  # Double channels at each layer

        # Decoder (reverse process)
        for _ in range(num_layers):
            out_channels = in_channels // 2
            self.decoder_layers.append(
                nn.ConvTranspose1d(
                    in_channels, out_channels, kernel_size, padding=kernel_size // 2
                )
            )
            in_channels = out_channels

        # Final layer to reconstruct exact input shape
        self.final_layer = nn.Conv1d(
            out_channels, input_channels, kernel_size, padding=kernel_size // 2
        )

    def forward(self, x):
        # Encoder
        encodings = []
        self.enc_indices = []
        sizes = []  # Store sizes for unpooling

        for layer in self.encoder_layers:
            x = F.leaky_relu(layer(x))
            encodings.append(x)
            sizes.append(x.shape[-1])  # Store sizes before pooling
            x, indices = self.pool(x)
            self.enc_indices.append(indices)

        # Decoder
        for i, layer in enumerate(self.decoder_layers):
            x = self.unpool(
                x, self.enc_indices[-(i + 1)], output_size=[sizes[-(i + 1)]]
            )
            x = F.leaky_relu(layer(x))

        # Final reconstruction layer
        x = self.final_layer(x)
        # x = torch.sigmoid(self.final_layer(x))

        return x


def train_model(
    model,
    train_dataloader,
    val_dataloader=None,
    epochs=10,
    lr=0.001,
    device="cpu",
    metric="val_loss",  # Metric to monitor for early stopping
    patience=3,  # Number of epochs to wait before stopping if no improvement
    pos_weight=None,  # Tensor of size (2,) for handling class imbalance
    alpha=0.25,  # Focal Loss alpha parameter
    gamma=2.0,  # Focal Loss gamma parameter
):
    # Ensure pos_weight is a tensor of shape (2,) if provided
    if pos_weight is not None:
        pos_weight = torch.tensor(pos_weight, device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    # criterion = FocalLoss(alpha=alpha, gamma=gamma)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_metric = (
        float("inf") if metric == "val_loss" else 0
    )  # Best val_loss (lower) or val_auc (higher)
    patience_counter = 0  # Tracks epochs without improvement
    best_model_state = None  # Stores best model weights

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        all_train_labels = []
        all_train_preds = []
        all_train_probs = []
        # Training Loop
        for features, labels in train_dataloader:
            features, labels = features.to(device), labels.to(device)
            labels_one_hot = torch.stack(
                [(1 - labels), labels], dim=1
            ).float()  # Convert 0 to [1, 0] and 1 to [0, 1]
            optimizer.zero_grad()
            # Logger.debug("Features shape: %s", features.shape)
            outputs = model(features)  # Output shape: (batch_size, 2)
            # Logger.debug("Output shape: %s", outputs.shape)
            loss = criterion(outputs, labels_one_hot)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            probs = outputs[:, 1].float().cpu().detach().numpy()
            preds = torch.argmax(outputs, dim=1).float().cpu().numpy()
            labels_np = labels.cpu().numpy()

            all_train_labels.extend(labels_np)
            all_train_preds.extend(preds)
            all_train_probs.extend(probs)

        train_loss = total_train_loss / len(train_dataloader)
        train_auc = roc_auc_score(all_train_labels, all_train_probs)

        # Validation Loop (if validation data is provided)
        val_loss, val_auc = None, None
        if val_dataloader:
            model.eval()
            total_val_loss = 0
            all_val_labels = []
            all_val_preds = []
            all_val_probs = []
            with torch.no_grad():
                for features, labels in val_dataloader:
                    features, labels = features.to(device), labels.to(device)
                    labels_one_hot = torch.stack(
                        [(1 - labels), labels], dim=1
                    ).float()  # Convert 0 to [1, 0] and 1 to [0, 1]
                    outputs = model(features)  # Output shape: (batch_size, 2)
                    loss = criterion(outputs, labels_one_hot)

                    total_val_loss += loss.item()
                    preds = torch.argmax(outputs, dim=1).float().cpu().numpy()
                    probs = outputs[:, 1].float().cpu().detach().numpy()
                    labels_np = labels.cpu().numpy()

                    all_val_labels.extend(labels_np)
                    all_val_preds.extend(preds)
                    all_val_probs.extend(probs)

            val_loss = total_val_loss / len(val_dataloader)
            val_auc = roc_auc_score(all_val_labels, all_val_probs)
            val_f1 = f1_score(all_val_labels, all_val_preds, average="micro")
            val_ap = average_precision_score(
                all_val_labels, all_val_probs, average="micro"
            )
            # Print epoch results
            print(
                f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Train AUC: {train_auc: .4f} | "
                f"Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f} | Val F1: {val_f1:.4f} | Val AP: {val_ap:.4f}"
            )

            # Early Stopping Logic
            match metric:
                case "val_loss":
                    current_metric = val_loss
                case "val_auc":
                    current_metric = val_auc
                case "val_f1":
                    current_metric = val_f1
                case "val_ap":
                    current_metric = val_ap
                case _:
                    raise ValueError(f"Invalid metric: {metric}")
            if (metric == "val_loss" and current_metric < best_metric) or (
                metric in ["val_auc", "val_f1", "val_ap"]
                and current_metric > best_metric
            ):
                best_metric = current_metric
                patience_counter = 0
                best_model_state = model.state_dict()  # Save best model state
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(
                        f"Early stopping triggered at epoch {epoch+1}. Restoring best model..."
                    )
                    model.load_state_dict(best_model_state)
                    break
        else:
            print(
                f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Train AUC: {train_auc:.4f}"
            )

    return model


def predict(model, dataloader, device="cpu"):
    """
    Predicts labels and probabilities using a trained model.

    Args:
        model (torch.nn.Module): Trained PyTorch model.
        dataloader (DataLoader): DataLoader containing test samples.
        device (str): "cpu" or "cuda" to specify where to run inference.

    Returns:
        probs (list): List of predicted probabilities.
        preds (list): List of binary predictions (0 or 1).
    """
    model.eval()  # Set model to evaluation mode
    model.to(device)

    probs = []
    preds = []

    with torch.no_grad():  # Disable gradient computation
        for features, _ in dataloader:  # Ignore labels if available
            features = features.to(device)

            outputs = model(features).squeeze()

            # probabilities = torch.sigmoid(outputs)  # Apply sigmoid for probabilities
            predictions = torch.argmax(
                outputs, dim=1
            ).float()  # Convert to binary labels
            probs.extend(
                torch.sigmoid(outputs[:, 1]).cpu().numpy()
            )  # Get sigmoid of second output
            preds.extend(predictions.cpu().numpy())

    return probs, preds


def reconstruct(
    model,
    dataloader_with_ids,
    device,
    scaling_factor: float = 1.0,
    error_cal: str = "MAE",
):
    model.to(device)
    with torch.no_grad():
        error = []
        ids_list = []
        for features, _, ids in dataloader_with_ids:
            features = features.to(device)
            reconstructed = model(features) * scaling_factor
            match error_cal:
                case "MAE":
                    reconstruction_error = (
                        torch.mean(abs(features - reconstructed), dim=2)
                        .flatten()
                        .cpu()
                        .detach()
                        .numpy()
                    )
                case "MSE":
                    reconstruction_error = (
                        torch.mean((features - reconstructed) ** 2, dim=2)
                        .flatten()
                        .cpu()
                        .detach()
                        .numpy()
                    )
                case "MAPE":
                    reconstruction_error = (
                        torch.mean(
                            abs((features - reconstructed) / (features + 1e-6)), dim=2
                        )
                        .flatten()
                        .cpu()
                        .detach()
                        .numpy()
                    )
                case _:
                    raise ValueError(f"Invalid error calculation: {error_cal}")
            error.extend(reconstruction_error)
            ids_list.extend(ids.flatten().detach().numpy())
            Logger.info("Reconstruction error length %s", len(error))
            Logger.info("IDs length %s", len(ids_list))
    result = pd.DataFrame({"id": ids_list, "error": error})
    return result
