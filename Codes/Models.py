import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_classes,
                 embed_matrix=None, trainable=True):

        super(LSTM, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        if embed_matrix is not None:
            self.embedding.weight = nn.Parameter(
                torch.tensor(embed_matrix, dtype=torch.float32),
                requires_grad=trainable
            )

        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=hidden_size, batch_first=True)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x = (batch, seq_len)

        x = self.embedding(x)           # (batch, seq_len, embed_dim)
        out, _ = self.lstm(x)           # (batch, seq_len, hidden)

        out = out[:, -1, :]             # last timestep
        out = self.bn(out)              # batch norm on features
        out = self.fc(out)              # logits

        return out


class Parallel_CNN_LSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_classes,
                 embed_matrix=None, trainable=True,
                 num_filters=64, kernel_sizes=[2,3,5]):

        super(Parallel_CNN_LSTM, self).__init__()

        # Embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        if embed_matrix is not None:
            self.embedding.weight = nn.Parameter(
                torch.tensor(embed_matrix, dtype=torch.float32),
                requires_grad=trainable
            )

        # LSTM
        self.lstm = nn.LSTM(embed_dim, hidden_size, batch_first=True)

        # CNN
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim,
                      out_channels=num_filters,
                      kernel_size=k)
            for k in kernel_sizes
        ])

        # BatchNorm
        self.bn_lstm = nn.BatchNorm1d(hidden_size)
        self.bn_cnn = nn.BatchNorm1d(num_filters * len(kernel_sizes))

        # Final classifier
        self.fc = nn.Linear(hidden_size + num_filters * len(kernel_sizes), num_classes)

    def forward(self, x):
        # x: (batch, seq_len)

        x = self.embedding(x)  # (batch, seq_len, embed_dim)

        # LSTM
        lstm_out, _ = self.lstm(x)
        # lstm_out = lstm_out[:, -1, :]  # last timestep
        lstm_out = torch.mean(lstm_out, dim=1)
        lstm_out = self.bn_lstm(lstm_out)

        # CNN
        cnn_in = x.permute(0, 2, 1)  # (batch, embed_dim, seq_len)

        conv_outs = []
        for conv in self.convs:
            c = F.relu(conv(cnn_in))             # (batch, num_filters, L')
            c = F.max_pool1d(c, kernel_size=c.size(2))  # global max pooling
            conv_outs.append(c.squeeze(2))       # (batch, num_filters)

        cnn_out = torch.cat(conv_outs, dim=1)    # (batch, num_filters * len(kernels))
        cnn_out = self.bn_cnn(cnn_out)

        # Concat
        combined = torch.cat([lstm_out, cnn_out], dim=1)

        out = self.fc(combined)

        return out