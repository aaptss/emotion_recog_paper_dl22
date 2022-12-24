import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import torchaudio
import numpy as np

from torch.autograd import Variable


class CNN_for_audio(nn.Module):
    def __init__(self, n_input=1, num_classes=10, stride=16, n_channel=13):
        super().__init__()
        self.n_input=n_input
        self.num_classes=num_classes
        self.stride=n_output
        self.n_channel=n_channel

    def block(self,in_features,out_features,kernel):
        block=nn.Sequential(nn.Conv1d(in_features, out_features, kernel, 4),
                            nn.BatchNorm1d(out_features),
                            nn.Relu,
                            nn.MaxPool1d(4))
        return block

        self.final_layer = nn.Linear(2 * n_channel, num_classes)
  

    def forward(self, x):
        x=self.block(self.n_input,self.n_channel,80,self.stride)(x)
        x=self.block(self.n_channel,self.n_channel,3)(x)
        x=self.block(self.n_channel,2*self.n_channel,3)(x)
        x=self.block(2*self.n_channel,2*self.n_channel,3)(x)
        x=self.final_layer(x)
        output=F.log_softmax(x, dim=2)

        return output