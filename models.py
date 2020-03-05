import torch
import torch.nn.functional as F
from torch import nn
import argparse
from itertools import chain


class DeepInfoMax(nn.Module):

    def __init__(self, nb_samples=1, layers_size = [25,10], out_channels = 5, emb_size = 10,data_dir ='.'):
        super(DeepInfoMax, self).__init__()

        self.sample = nb_samples
        self.out_channels = out_channels
        self.emb_size = emb_size
        self.conv1 = nn.Conv1d(in_channels = 20,out_channels = 10,kernel_size = 15,stride = 1)
        self.conv2 = nn.Conv1d(in_channels = 10,out_channels = self.out_channels, kernel_size=12,stride = 1)

        #self.conv1 = nn.Conv1d(in_channels = 20,out_channels = 10,kernel_size = 5,stride = 1)
        #self.conv2 = nn.Conv1d(in_channels = 10,out_channels = self.out_channels, kernel_size=14,stride = 1)
        layers = []
        #dim = [(10*(27-5+1))+self.out_channels*self.emb_size] + layers_size # Adding the emb size*out_channels
        dim = [(10*(27-15+1))+self.emb_size] + layers_size # Adding the emb size*out_channels

        for size_in, size_out in zip(dim[:-1], dim[1:]):
            layer = nn.Linear(size_in, size_out)
            layers.append(layer)

        self.mlp_layers = nn.ModuleList(layers)

        # Last layer
        self.last_layer = nn.Linear(dim[-1], 2)
        self.softmax = nn.Softmax(dim=1)


    def get_feature_map(self, x1):

        fm = self.conv1(x1)
        fm = fm.reshape((fm.shape[0], fm.shape[1]*fm.shape[2]))
        return fm

    def get_feature_vector(self, x1):

        fv = self.conv1(x1)
        fm = self.conv2(fv)
        #import pdb; pdb.set_trace()

        fm = fm.reshape((fm.shape[0], fm.shape[1]*fm.shape[2]))
        return fm

    def forward(self,  x1, x2):

        # Get the feature maps
        fm1, fm2 = self.get_feature_map(x1), self.get_feature_map(x2)
        fv1, fv2 = self.get_feature_vector(x1), self.get_feature_vector(x1)
        self.fv = fv1


        #emb_1 = emb_1.permute(1,0,2)
        #emb_1 = emb_1.squeeze()
        #emb_2 = emb_2.squeeze()
        #emb_2 = emb_2.view(-1,2)
        #if not emb_1.shape == emb_2.shape:
        #    import pdb; pdb.set_trace()
        # Forward pass for real
        mlp_input_1 = torch.cat([fm1, fv1], 1)
        for layer in self.mlp_layers:
            mlp_input_1 = layer(mlp_input_1)
            mlp_input_1 = F.tanh(mlp_input_1)
        mlp_output_1 = self.last_layer(mlp_input_1)
        mlp_output_1 = self.softmax(mlp_output_1)

        # Forward pass for fake
        mlp_input_2 = torch.cat([fm2, fv1], 1)
        for layer in self.mlp_layers:
            mlp_input_2 = layer(mlp_input_2)
            mlp_input_2 = F.tanh(mlp_input_2)
        mlp_output_2 = self.last_layer(mlp_input_2)
        mlp_output_2 = self.softmax(mlp_output_2)


        return mlp_output_1, mlp_output_2

class Benchmark(nn.Module):

    def __init__(self, nb_samples=1, layers_size = [25,10], out_channels = 5, emb_size = 10,data_dir ='.'):
        super(DeepInfoMax, self).__init__()

        self.sample = nb_samples
        self.out_channels = out_channels
        self.emb_size = emb_size
        #self.conv1 = nn.Conv1d(in_channels = 20,out_channels = 10,kernel_size = 15,stride = 1)
        #self.conv2 = nn.Conv1d(in_channels = 10,out_channels = self.out_channels, kernel_size=12,stride = 1)

        self.conv1 = nn.Conv1d(in_channels = 20,out_channels = 10,kernel_size = 5,stride = 1)
        self.conv2 = nn.Conv1d(in_channels = 10,out_channels = self.out_channels, kernel_size=14,stride = 1)
        layers = []
        dim = [(10*(27-5+1))+self.out_channels*self.emb_size] + layers_size # Adding the emb size*out_channels
        #dim = [(10*(27-15+1))+self.emb_size] + layers_size # Adding the emb size*out_channels

        for size_in, size_out in zip(dim[:-1], dim[1:]):
            layer = nn.Linear(size_in, size_out)
            layers.append(layer)

        self.mlp_layers = nn.ModuleList(layers)

        # Last layer
        self.last_layer = nn.Linear(dim[-1], 2)
        self.softmax = nn.Softmax(dim=14)


 
    def get_feature_vector(self, x1):

        fv = self.conv1(x1)
        fm = self.conv2(fv)
        #import pdb; pdb.set_trace()

        fm = fm.reshape((fm.shape[0], fm.shape[1]*fm.shape[2]))
        return fm

    def forward(self,  x1):

        
        mlp_input = self.get_feature_vector(x1)
        
        
        for layer in self.mlp_layers:
            mlp_input = layer(mlp_input)
            mlp_input = F.tanh(mlp_input)
        mlp_output = self.last_layer(mlp_input)
        mlp_output = self.softmax(mlp_output)



        return mlp_output

def get_model(opt, inputs_size, model_state=None):

    if opt.model == 'CNN':
        model_class = DeepInfoMax
        model = model_class(layers_size=opt.layers_size, out_channels=opt.out_channels, emb_size=opt.emb_size, data_dir = opt.data_dir)
    elif opt.model == 'benchmark':
        model_class = Benchmark
        model = model_class(layers_size=opt.layers_size, out_channels=opt.out_channels, emb_size=opt.emb_size, data_dir = opt.data_dir)
    else:
        raise NotImplementedError()

    if model_state is not None:
        model.load_state_dict(model_state)

    return model
