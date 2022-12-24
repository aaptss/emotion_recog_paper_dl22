import json 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
import numpy as np

from torch.autograd import Variable
from .model_arguments import modelArgs
from data_csgo_tournament.csgo_tournament_metadata import DATASET_ROOT

class RNN(nn.Module):
    def __init__(
        self, 
        hidden_size, 
        num_layers, 
        input_size, 
        num_classes, 
        device, 
        classes=None
    ):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.device = device
        self.classes = classes

"""
принимает запись и вектор состояния 
кидаю запись в рнн 
(беру на выходие с лстм ) -- это вытащенные фичи 
фичи конкат с вектором 

конкаченное в fc
"""
    def forward(self, x):
        # Set initial hidden and cell states
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device) 
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device) 
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # shape = (batch_size, seq_length, hidden_size)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

    def predict(self, x):
        '''Predict one label from one sample's features'''
        # x: feature from a sample, LxN
        #   L is length of sequency
        #   N is feature dimension
        x = torch.tensor(x[np.newaxis, :], dtype=torch.float32)
        x = x.to(self.device)
        outputs = self.forward(x)
        _, predicted = torch.max(outputs.data, 1)
        predicted_index = predicted.item()
        return predicted_index
    
    def set_classes(self, classes):
        self.classes = classes 
    
    def predict_audio_label(self, audio):
        idx = self.predict_audio_label_index(audio)
        assert self.classes, "Classes names are not set. Don't know what audio label is"
        label = self.classes[idx]
        return label

    def predict_audio_label_index(self, audio):
        audio.compute_mfcc()
        x = audio.mfcc.T # (time_len, feature_dimension)
        idx = self.predict(x)
        return idx

def load_weights(model, weights, PRINT=False):    
    for i, (name, param) in enumerate(weights.items()):
        model_state = model.state_dict()
        if name not in model_state:
            print("-"*80)
            print("weights name:", name) 
            print("RNN states names:", model_state.keys()) 
            assert 0, "Wrong weights file"
            
        model_shape = model_state[name].shape
        if model_shape != param.shape:
            print(f"\nWarning: Size of {name} layer is different between model and weights. Not copy parameters.")
            print(f"\tModel shape = {model_shape}, weights' shape = {param.shape}.")
        else:
            model_state[name].copy_(param)
        

def create_RNN(args: modelArgs):
    save_log_to = args.save_model_to + "log.txt"
    save_fig_to = args.save_model_to + "fig.jpg"
    
    # Create model
    model = RNN(
        args.input_size, 
        args.hidden_size, 
        args.num_layers, 
        args.num_classes, 
        args.device).to(device)

    if args.load_weights_from:
        print(f"Load weights from: {args.load_weights_from}")
        weights = torch.load(args.load_weights_from)
        load_weights(model, weights)
    return model


if __name__ == ("__main__"):
    args = modelArgs()

    args.hidden_size = 64
    args.num_layers = 3
    args.input_size = 12  # == n_mfcc
    args.batch_size = 1

    # training params
    args.num_epochs = 100
    args.learning_rate = 0.0001
    args.learning_rate_decay_interval = 5 # decay for every 5 epochs
    args.learning_rate_decay_rate = 0.5 # lr = lr * rate
    args.weight_decay = 0.00
    args.gradient_accumulations = 16 # number of gradient accums before step
    
    # training params2
    args.load_weights_from = None
    args.finetune_model = False # If true, fix all parameters except the fc layer
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # data
    args.data_folder = f'./{DATASET_ROOT}/LCB_audio'
    args.splitsize=0.8
    args.do_data_augment = True

    # labels
    args.classname_filename = f'{DATASET_ROOT}/classes.names' 
    with open(args.classname_filename) as f:
        _classes = json.load(f)
    args.num_classes = len(_classes) # should be added with a value somewhere, like this:
    #                = len(lib_io.read_list(args.classes_txt))

    # log setting
    args.plot_accu = True # if true, plot accuracy for every epoch
    args.show_plotted_accu = False # if false, not calling plt.show(), so drawing figure in background
    args.save_model_to = 'checkpoints/' # Save model and log file
        #e.g: model_001.ckpt, log.txt, log.jpg
    