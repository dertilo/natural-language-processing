import torch.nn as nn
import torch.nn.functional as F


class EmbeddingBagClfModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super().__init__()
        self.embedding = nn.EmbeddingBag(
            vocab_size, embed_dim, sparse=False
        )  # TODO(tilo): sparse=True leads to error in SGD gradient momentum calculation
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, x):
        text, offsets = x
        return self.fc(self.embedding(text, offsets))


class ConvNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, num_channels):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.convnet = nn.Sequential(
            nn.Conv1d(in_channels=embed_dim, out_channels=num_channels, kernel_size=3),
            nn.ELU(),
            nn.Conv1d(
                in_channels=num_channels,
                out_channels=num_channels,
                kernel_size=3,
                stride=2,
            ),
            nn.ELU(),
            nn.Conv1d(
                in_channels=num_channels,
                out_channels=num_channels,
                kernel_size=3,
                stride=2,
            ),
            nn.ELU(),
            nn.Conv1d(
                in_channels=num_channels, out_channels=num_channels, kernel_size=3
            ),
            nn.ELU(),
        )
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(num_channels, num_classes)

    def forward(self, text):
        x = self.embedding(text)
        x = x.transpose(2,1)
        features = self.convnet(x)
        features_pooled = self.pooling(features).squeeze()
        return self.fc(features_pooled)
