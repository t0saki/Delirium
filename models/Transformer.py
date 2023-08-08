import torch
import torch.nn as nn


class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, num_heads, dropout_rate):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim)
        self.transformer_layers = nn.ModuleList(
            [TransformerLayer(hidden_dim, num_heads, dropout_rate)
             for _ in range(num_layers)]
        )
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.embedding(x)
        x = self.positional_encoding(x)
        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x)
        x = torch.mean(x, dim=1)  # 取平均作为全局特征
        x = self.output_layer(x)
        x = self.sigmoid(x).squeeze()
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float(
        ) * (-torch.log(torch.tensor(10000.0)) / hidden_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x


class TransformerLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout_rate):
        super(TransformerLayer, self).__init__()
        self.self_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout_rate)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x, _ = self.self_attention(x, x, x)
        x = self.dropout1(x)
        x = x + residual

        residual = x
        x = self.norm2(x)
        x = self.feed_forward(x)
        x = self.dropout2(x)
        x = x + residual

        return x


if __name__ == '__main__':
    seq_len = 128
    batch_size = 32
    hidden_dim = 64
    output_dim = 1
    num_layers = 2
    num_heads = 64
    dropout_rate = 0.1

    model = TransformerModel(input_dim=4, output_dim=output_dim, hidden_dim=hidden_dim,
                             num_layers=num_layers, num_heads=num_heads, dropout_rate=dropout_rate)

    # Test the model
    x = torch.randn(32, 4, 128)

    y = model(x)
    print(y.shape)
