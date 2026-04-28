import torch.nn as nn

class CNNLSTM(nn.Module):
    def __init__(self, F):
        super().__init__()

        self.conv1 = nn.Conv1d(F, 32, 3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, 3, padding=1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

        self.lstm = nn.LSTM(64, 64, batch_first=True)
        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.dropout(self.relu(self.conv1(x)))
        x = self.dropout(self.relu(self.conv2(x)))
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        return self.fc(x)