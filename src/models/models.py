from typing import overload

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import wfdb


class RNNLayer(nn.Module):
    def __init__(self, in_ax, aa_channels, out_ya, g1_activation='tanh', g2_activation='tanh'):
        """
        :param in_ax: The input size of the linear layer corresponds to the Wax matrix.
        :param aa_channels: Number of channels in the linear layer corresponds to the Waa matrix.
        :param out_ya: Output size of the linear layer corresponds to the Wya matrix.
        :param g1_activation: First activation type.
        :param g2_activation: Second activation type.
        """
        super().__init__()
        activations = {'sigmoid': nn.Sigmoid(), 'tanh': nn.Tanh(), 'relu': nn.ReLU(),
                       'regression': nn.Identity()}  # you can use 'regression' for no activation

        # ------Your code------#
        # Define the three linear layers, each corresponds to a different weight matrix.
        # When you need to add layers together (first equation), make sure they have the same output size.
        # Set the activations according to their input types, use the given "activations" dictionary.

        self.aa = nn.Linear(aa_channels, aa_channels, bias=True)
        self.ax = nn.Linear(in_ax, aa_channels, bias=False)
        self.ya = nn.Linear(aa_channels, out_ya, bias=True)

        self.g1 = activations[g1_activation]
        self.g2 = activations[g2_activation]
        # ------^^^^^^^^^------#

    def forward(self, x, a_in=None):
        # Start with zeros if no second input was given:
        if a_in is None:
            a_in = torch.zeros_like(self.ax(x))  # Makes sure it matches the input dimensions.

        # ------Your code------#
        # Complete according to the equations and return the outputs: y_pred and a_out.
        a_out = self.g1(self.aa(a_in) + self.ax(x))
        y_pred = self.g2(self.ya(a_out))
        # ------^^^^^^^^^------#
        return y_pred, a_out


class BRNNModel(nn.Module):
    def __init__(self,
                 in_ax=1,
                 aa_channels=128,
                 out_ya=1,
                 g_activation='regression',
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super().__init__()

        activations = {'sigmoid': nn.Sigmoid(), 'tanh': nn.Tanh(), 'relu': nn.ReLU(), 'regression': nn.Identity()}

        self.aa_channels = aa_channels
        self.device = device

        # Here we have two RNNs - forward and backward:
        self.rnn_forward = RNNLayer(in_ax=in_ax, aa_channels=aa_channels, out_ya=1, g1_activation='relu',
                                    g2_activation='regression')
        self.rnn_backward = RNNLayer(in_ax=in_ax, aa_channels=aa_channels, out_ya=1, g1_activation='relu',
                                     g2_activation='regression')

        self.linear = nn.Linear(aa_channels * 2, out_ya, bias=True)  # channels are doubled in the BRNN case.
        self.g = activations[g_activation]

    def forward(self, X, a=None):

        if a is None:
            a_f = None
            a_b = None
        else:
            # According to this specific implementation, this is how we select the last state in both ways:
            a_f = a[:, -1, :self.aa_channels]
            a_b = a[:, 0, self.aa_channels:]

        a_fs = torch.zeros((X.shape[0], X.shape[1], self.aa_channels), device=self.device)
        X_forward = torch.transpose(X, 0,
                                    1)  # The transpose is required to iterate over the first dim, which needs to be the sequence.
        for i, xt in enumerate(X_forward):
            _, a_f = self.rnn_forward(xt, a_f)
            a_fs[:, i, :] = a_f

        a_bs = torch.zeros((X.shape[0], X.shape[1], self.aa_channels), device=self.device)
        X_backward = torch.flip(X_forward, [0])  # after transpose the sequence dimension is the first.
        for i, xt in enumerate(X_backward):
            _, a_b = self.rnn_backward(xt, a_b)
            a_bs[:, i, :] = a_b

        a_bs = torch.flip(a_bs, [1])
        a = torch.cat((a_fs, a_bs), dim=2)  # concatenate in the feature dim.

        y_pred = self.g(self.linear(a))

        return y_pred, a


class GRUModel(nn.Module):
    def __init__(self, input_shape, aa_channels=128, out_ya=1, num_layers=2, batch_first=True, bidirectional=True):
        super().__init__()
        # ------Your code------#
        # Add GRU and linear layers.
        # Use the num_layer attribute to concatenate several GRU layers.
        # Use aa_channels to set the linear layer size according to the bidirectional input.
        _, input_channels, in_ax = input_shape
        self.GRU = nn.GRU(input_size=in_ax, hidden_size=aa_channels, num_layers=num_layers, batch_first=batch_first,
                          bidirectional=bidirectional)

        if bidirectional:
            in_features = aa_channels * 2
        else:
            in_features = aa_channels

        self.linear = nn.Linear(in_features=in_features, out_features=out_ya)

    # ------^^^^^^^^^------#

    def forward(self, x, a_in=None):
        # ------Your code------#
        # Set a condition to use only x as input if a_in=None.
        if a_in is None:
            GRU_output, a_out = self.GRU(x)  # GRU output is: (output features for each t, final hidden state)
        else:
            GRU_output, a_out = self.GRU(x, a_in)

        # Select last hidden state of last layer from each direction - for many-to-one classification task:
        # a_out is (D∗num_layers, batch, features)
        last_forward = a_out[-2]
        last_backward = a_out[-1]
        lasts = torch.cat((last_forward, last_backward), dim=1)  # dim is features

        y_pred = self.linear(lasts)
        # ------^^^^^^^^^------#
        return y_pred, a_out


class CNN_BN_P(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=11, stride=1, p=0.5):
        """
        A simple CNN layer which is also followed by batch normalization and dropout layers.
        """
        super().__init__()

        self.layer = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=0, dilation=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(p=p),
        )

    def forward(self, x):
        return self.layer(x)


class SequentialNet(nn.Module):
    def __init__(self, input_shape, fs):
        """
        :param input_shape: input tensor shape (B, C_in=1, S*L=5*60*fs), assuming 5-minute segments.
        :param fs: the sampling frequency of the signals is used for reshaping transformation of the inputs.
        """
        super().__init__()

        self.fs = fs

        # A feature extraction (CNN) block.
        # Will compute each short (10-second) segment individually:
        self.CNN = nn.Sequential(
            CNN_BN_P(in_channels=input_shape[1], out_channels=64, stride=1),
            CNN_BN_P(in_channels=64, out_channels=128, stride=2),
            # We use stride to reduce the number of features created.
            CNN_BN_P(in_channels=128, out_channels=128, stride=2),
            CNN_BN_P(in_channels=128, out_channels=128, stride=2),
        )

        # Forward to get the CNN output:
        CNN_forward = self.CNN(self.reshape_input2CNN(torch.zeros(input_shape)))
        # A linear layer to reduce the final feature dim:
        self.linear_post_CNN = nn.Linear(CNN_forward.shape[1] * CNN_forward.shape[2], 256)

        # Forward to get the linear layer output:
        CNN_forward_flat = CNN_forward.view(CNN_forward.size(0), -1)  # flatten
        linear_forward = self.linear_post_CNN(CNN_forward_flat)

        linear_forward_reshaped = self.reshape_encoder2GRU(linear_forward)  # (B,L,H)

        # Sequential (GRU) block:
        # Will compute the long (5-minute) segment classifcation using all the short-segment features as a sequence:
        self.GRU = GRUModel(input_shape=linear_forward_reshaped.shape,
                            aa_channels=256,
                            out_ya=1, num_layers=1, batch_first=True, bidirectional=True)

    def reshape_input2CNN(self, signals):
        """
        (B,C,S*L) --> (B*S,C,L);
        where:
        - B is the batch size,
        - C is the channel dim.
        - S is the samples dim of the 10-second window segments,
        - L is the sequence dim (whole 5 minutes),
        """
        S = 5
        # --> (B,C,S,L)
        unfolded = signals.unfold(dimension=2, size=S * self.fs, step=S * self.fs)  # 10 seconds
        # --> (B,S,C,L)
        permuted = unfolded.permute(0, 2, 1, 3)
        # --> (B*S,C,L)
        reshaped = permuted.reshape(-1, signals.shape[1], S * self.fs)
        return reshaped

    def reshape_encoder2GRU(self, signals):
        """
        (B*L,H) --> (B,L,H);
        where H is the features computed from each 10-second window segment.
        """
        # --> (B,H,L)
        w = 5  # windows
        t = 15  # time in seconds
        unfolded = signals.unfold(dimension=0, size=t // w, step=t // w)
        # --> (B,L,H)
        permuted = unfolded.permute(0, 2, 1)
        return permuted

    def forward(self, x, print_shapes=False):
        # Reshape input:
        x_reshaped = self.reshape_input2CNN(x)
        # Forward CNN:
        features = self.CNN(x_reshaped)
        # Flatten:
        features_flat = features.view(features.size(0), -1)
        # Forward linear layer:
        features_reduced = self.linear_post_CNN(features_flat)
        # Reshape before sequential:
        features_reshaped = self.reshape_encoder2GRU(features_reduced)
        # Forward Sequential and predict:
        gru_pred, _ = self.GRU(features_reshaped)

        # --- Only for demonstarting the dimension changing --- #
        if print_shapes:
            print('x:\n ', x.shape)
            print('x_reshaped:\n ', x_reshaped.shape)
            print('features:\n ', features.shape)
            print('features_flat:\n ', features_flat.shape)
            print('features_reduced:\n ', features_reduced.shape)
            print('features_reshaped:\n ', features_reshaped.shape)
            print('gru_pred:\n ', gru_pred.shape)
        ##########################################################

        return torch.squeeze(gru_pred)
