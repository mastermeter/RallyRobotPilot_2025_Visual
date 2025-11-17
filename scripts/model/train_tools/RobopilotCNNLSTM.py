import torch.nn as nn

class RobopilotCNNLSTM(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        seq_len: int = 8,
        cnn_feature_dim: int = 256,
        lstm_hidden_dim: int = 128,
        lstm_num_layers: int = 1,
        output_size: int = 3,
        dropout_rate: float = 0.3,
    ):
        super().__init__()
        self.seq_len = seq_len

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.cnn_head = nn.Sequential(
            nn.Flatten(),                          
            nn.Dropout(dropout_rate),
            nn.Linear(128 * 8 * 8, cnn_feature_dim),
            nn.ReLU(),                 
        )

        self.lstm = nn.LSTM(
            input_size=cnn_feature_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            batch_first=True,   
            bidirectional=False
        )

        self.fc_out = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(lstm_hidden_dim, output_size)
        )

    def forward(self, x):
        """
        x : (B, T, C, H, W)
        """
        B, T, C, H, W = x.shape
        # fusionner B et T -> (B*T, C, H, W)
        x = x.reshape(B * T, C, H, W)
        x = self.conv(x)
        x = self.cnn_head(x)          # (B*T, F)

        # remettre en séquence -> (B, T, F)
        F = x.shape[-1]
        x = x.reshape(B, T, F)

        # LSTM
        lstm_out, _ = self.lstm(x)    # (B, T, H)
        last_h = lstm_out[:, -1, :]   # (B, H)

        logits = self.fc_out(last_h)  # (B, 3)
        return logits
