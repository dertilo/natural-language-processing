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

        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
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
        self.fc = nn.Linear(num_channels, num_classes)

    def forward(self, text, offsets):
        x = self.embedding(text, offsets)
        features = self.convnet(x).squeeze(dim=2)
        return self.fc(features)
