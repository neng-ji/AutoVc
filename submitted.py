from typing import Any, Tuple

from torch import Tensor, matmul, norm, randn, zeros, zeros_like
from torch.nn import BatchNorm1d, Conv1d, Module, ModuleList, Parameter, ReLU, Sequential, Sigmoid, Tanh
from torch.nn import LSTM, GRU, Linear
import torch.nn.functional as F


from torchtyping import TensorType

################################################################################



from typing import Tuple

import torch
from torch import nn
import random

class Encoder(nn.Module):
    def __init__(self, dim_neck: int, dim_emb: int):
        super(Encoder, self).__init__()
        self.dim_neck = dim_neck

        self.convolutions = nn.ModuleList()
        self.convolutions.append(
            nn.Conv1d(in_channels=80 + dim_emb, out_channels=512, kernel_size=5, stride=1, padding=2))
        self.convolutions.append(nn.BatchNorm1d(512))
        self.convolutions.append(nn.ReLU())

        for _ in range(2):
            self.convolutions.append(nn.Conv1d(in_channels=512, out_channels=512, kernel_size=5, stride=1, padding=2))
            self.convolutions.append(nn.BatchNorm1d(512))
            self.convolutions.append(nn.ReLU())

        self.recurrent = nn.LSTM(input_size=512, hidden_size=dim_neck, num_layers=2, batch_first=True,
                                 bidirectional=True)

    def forward(self, x):
        for module in self.convolutions:
            x = module(x)

        x = x.transpose(1, 2)

        lstm_out, _ = self.recurrent(x)
        out_forward = lstm_out[:, :, :self.dim_neck]
        out_backward = lstm_out[:, :, self.dim_neck:]

        return torch.cat((out_forward, out_backward), dim=2)


class SpeakerEmbedder(Module):
    """
    Style Encoder (Figure 3(b) of AutoVC paper, but simplified).
    """

    def __init__(self, n_hid: int, n_mels: int, n_layers: int, fc_dim: int, hidden_p: float) -> None:
        """
        Sets up the following:
            self.rnn_stack - an n_layers-layer GRU with n_mels input features,
                n_hid hidden features, and a dropout of hidden_p, with input and output tensors in
                batch_first=True ordering.
            self.projection - a Linear layer with an input size of n_hid
                and an output size of fc_dim.
        """
        super(SpeakerEmbedder, self).__init__()
        self.rnn_stack = GRU(n_mels, n_hid, n_layers, batch_first=True, dropout=hidden_p if n_layers > 1 else 0.)
        self.projection = Linear(n_hid, fc_dim)

    def forward(self, x: TensorType["batch", "frames", "n_mels"]) -> TensorType["batch", "fc_dim"]:
        """
        Performs the forward propagation of the SpeakerEmbedder.
            After passing the input through the RNN, the last frame of the output
            should be taken and passed through the fully connected layer.
            Each of the frames should then be normalized so that its Euclidean norm is 1.
        """
        output, _ = self.rnn_stack(x)
        last_frame = output[:, -1, :]
        embedded = self.projection(last_frame)

        normalized_embeddings = F.normalize(embedded, p=2, dim=1)
        return normalized_embeddings


class Decoder(nn.Module):
    def __init__(self, dim_neck: int, dim_emb: int, dim_pre: int) -> None:
        super(Decoder, self).__init__()
        self.recurrent1 = nn.LSTM(input_size=2 * dim_neck + dim_emb, hidden_size=dim_pre, num_layers=1, batch_first=True)

        self.convolutions = nn.ModuleList()
        for _ in range(3):
            self.convolutions.append(nn.Conv1d(in_channels=dim_pre, out_channels=dim_pre, kernel_size=5, stride=1, padding=2, dilation=1))
            self.convolutions.append(nn.BatchNorm1d(dim_pre))
            self.convolutions.append(nn.ReLU())

        self.recurrent2 = nn.LSTM(input_size=dim_pre, hidden_size=1024, num_layers=2, batch_first=True)
        self.fc_projection = nn.Linear(in_features=1024, out_features=80)

    def forward(self, x):
        x, _ = self.recurrent1(x)

        x = x.transpose(1, 2)
        for layer in self.convolutions:
            x = layer(x)

        x = x.transpose(1, 2)
        x, _ = self.recurrent2(x)
        x = self.fc_projection(x)

        return x



from torch import Tensor
from torch.nn import Module, Conv1d, BatchNorm1d, Sequential, Tanh
from torchtyping import TensorType


class Postnet(Module):
    def __init__(self) -> None:
        super(Postnet, self).__init__()

        self.convolutions = Sequential(
            Conv1d(80, 512, kernel_size=5, stride=1, padding=2, dilation=1),
            BatchNorm1d(512),
            Tanh(),

            Conv1d(512, 512, kernel_size=5, stride=1, padding=2, dilation=1),
            BatchNorm1d(512),
            Tanh(),

            Conv1d(512, 512, kernel_size=5, stride=1, padding=2, dilation=1),
            BatchNorm1d(512),
            Tanh(),

            Conv1d(512, 512, kernel_size=5, stride=1, padding=2, dilation=1),
            BatchNorm1d(512),
            Tanh(),

            Conv1d(512, 80, kernel_size=5, stride=1, padding=2, dilation=1),
            BatchNorm1d(80)
        )

    def forward(self, x: TensorType["batch", "input_channels", "n_mels"]) -> TensorType[
        "batch", "input_channels", "n_mels"]:
        return self.convolutions(x)