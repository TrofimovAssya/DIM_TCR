import torch
import torch.nn.functional as F
from torch import nn
import argparse
from itertools import chain


class DeepInfoMax(nn.Module):

    def __init__(self, nb_samples=1, conv_layers_sizes = [20,10,5,10,5,14],mlp_layers_size = [25,10], emb_size = 10,data_dir ='.', seqlength = 27):
        super(DeepInfoMax, self).__init__()

        self.emb_size = emb_size
        self.seqlength = seqlength
        self.class_number = class_number

        ## adding on the conv layers
        layers = []
        outsize = 27 

        for i in range(0,len(conv_layers_sizes),3):
            layer = nn.Conv1d(in_channels = conv_layers_sizes[i+0],out_channels = conv_layers_sizes[i+1],kernel_size = conv_layers_sizes[i+2],stride = 1)
            outsize = outsize-conv_layers_sizes[i+2]+1
            layers.append(layer)
            dim1 = [(conv_layers_sizes[i+1]*(outsize))]

        self.conv_stack = nn.ModuleList(layers)
        dim1 = dim1[0]
        if not dim1==emb_size:
            import pdb; pdb.set_trace()

        dim = dim + mlp_layers_size # Adding the emb size*out_channels

        for size_in, size_out in zip(dim[:-1], dim[1:]):
            layer = nn.Linear(size_in, size_out)
            layers.append(layer)

        self.mlp_layers = nn.ModuleList(layers)

        # Last layer
        self.last_layer = nn.Linear(dim[-1], self.class_number)
        #self.softmax = nn.Softmax(dim=1)

    def get_feature_map(self, x1):
        fm = self.tcr_conv_stack[0](x1)

        fm = fm.reshape((fm.shape[0], fm.shape[1]*fm.shape[2]))
        return fm

    def get_feature_vector(self, tcr):

        for layer in self.tcr_conv_stack:
            tcr = layer(tcr)

        tcr = tcr.reshape((tcr.shape[0], tcr.shape[1]*tcr.shape[2]))
        return tcr


    def forward(self,  x1, x2):

        # Get the feature maps
        fm1, fm2 = self.get_feature_map(x1), self.get_feature_map(x2)
        fv1, fv2 = self.get_feature_vector(x1), self.get_feature_vector(x1)
        self.fv = fv1


        mlp_input_1 = torch.cat([fm1, fv1], 1)
        for layer in self.mlp_layers:
            mlp_input_1 = layer(mlp_input_1)
            mlp_input_1 = F.tanh(mlp_input_1)
        mlp_output_1 = self.last_layer(mlp_input_1)
        mlp_output_1 = F.log_softmax(mlp_output_1)

        # Forward pass for fake
        mlp_input_2 = torch.cat([fm2, fv1], 1)
        for layer in self.mlp_layers:
            mlp_input_2 = layer(mlp_input_2)
            mlp_input_2 = F.tanh(mlp_input_2)
        mlp_output_2 = self.last_layer(mlp_input_2)
        mlp_output_2 = F.log_softmax(mlp_output_2)


        return mlp_output_1, mlp_output_2


class CNNClassifier(nn.Module):

    def __init__(self, nb_samples=1, conv_layers_sizes = [20,10,15,10,5,12],mlp_layers_size = [25,10], emb_size = 10, class_number = 14, data_dir ='.'):
        super(CNNClassifier, self).__init__()

        self.sample = nb_samples
        self.emb_size = emb_size
        self.class_number = class_number

        ## adding on the conv layers
        layers = []
        outsize = 27 

        for i in range(0,len(conv_layers_sizes),3):
            layer = nn.Conv1d(in_channels = conv_layers_sizes[i+0],out_channels = conv_layers_sizes[i+1],kernel_size = conv_layers_sizes[i+2],stride = 1)
            outsize = outsize-conv_layers_sizes[i+2]+1
            layers.append(layer)
            dim1 = [(conv_layers_sizes[i+1]*(outsize))]

        self.conv_stack = nn.ModuleList(layers)
        dim1 = dim1[0]
        if not dim1==emb_size:
            import pdb; pdb.set_trace()

        dim = dim + mlp_layers_size # Adding the emb size*out_channels

        for size_in, size_out in zip(dim[:-1], dim[1:]):
            layer = nn.Linear(size_in, size_out)
            layers.append(layer)

        self.mlp_layers = nn.ModuleList(layers)

        # Last layer
        self.last_layer = nn.Linear(dim[-1], self.class_number)
        #self.softmax = nn.Softmax(dim=1)


    def get_feature_vector(self, tcr):

        for layer in self.tcr_conv_stack:
            tcr = layer(tcr)

        tcr = tcr.reshape((tcr.shape[0], tcr.shape[1]*tcr.shape[2]))
        return tcr



    def forward(self,  x1):

        # Get the feature maps

        fv = self.get_feature_vector(x1)
        self.fv = fv


        mlp_input = fv
        #import pdb; pdb.set_trace()
        for layer in self.mlp_layers:
            mlp_input = layer(mlp_input)
            mlp_input = F.tanh(mlp_input)
        mlp_output = self.last_layer(mlp_input)
        mlp_output = F.log_softmax(mlp_output)

        return mlp_output



class CNNAutoEncoder(nn.Module):


    def __init__(self, conv_layers_sizes = [20,10,15,10,5,12], emb_size = 10, data_dir ='.'):
        super(CNNAutoEncoder, self).__init__()

        layers_conv = []
        layers_deconv = []
        outsize = 27 

        for i in range(0,len(conv_layers_sizes),3):
            layer = nn.Conv1d(in_channels = conv_layers_sizes[i+0],
                              out_channels = conv_layers_sizes[i+1],
                              kernel_size = conv_layers_sizes[i+2],stride = 1)
            outsize = outsize-conv_layers_sizes[i+2]+1
            layers_conv.append(layer)
            dim1 = [(conv_layers_sizes[i+1]*(outsize))]

            layer = nn.ConvTranspose1d(in_channels =
                                       conv_layers_sizes[i+1],
                                       out_channels = conv_layers_sizes[i+0],
                                       kernel_size = conv_layers_sizes[i+2],stride = 1)
            layers_deconv.append(layer)

        self.conv_stack = nn.ModuleList(layers)
        dim1 = dim1[0]
        if not dim1==emb_size:
            import pdb; pdb.set_trace()

        self.conv_stack = nn.ModuleList(layers_conv)
        self.deconv_stack = nn.ModuleList(layers_deconv)



    def encode(self, tcr):

        for layer in self.conv_stack:
            tcr = layer(tcr)

        return tcr


    def dencode(self, tcr):

        for layer in self.conv_stack[::-1]:
            tcr = layer(tcr)

        return tcr

    def forward(self,  x1):


        fv = self.encode(x1)
        self.fv = fv
        output = self.decode(fv)

        return output



class FullDIM(nn.Module):

    def __init__(self, conv_layers_sizes1 = [20,10,5,10,5,14],
                 conv_layers_sizes2 = [20,10,5,10,5,14],
                 mlp_layers_size1 = [25,10], mlp_layers_size2 = [25,10],
                 mlp_layers_size3 = [25,10], seqlength1 = 27, seqlength2 = 12):
        super(FullDIM, self).__init__()

        self.emb_size = emb_size
        self.seqlength1 = seqlength1
        self.seqlength2 = seqlength2

        ### getting the cnn for the tcr-side deepinfomax

        self.cnn_stack1, out_size1 = get_cnn_stack(conv_layers_sizes1, self.seqlength1)
        self.tcr_dim = nn.ModuleList(self.cnn_stack1)

        ### getting the cnn for the pep-side deepinfomax

        self.cnn_stack2, out_size2 = get_cnn_stack(conv_layers_sizes2, self.seqlength2)
        self.pep_dim = nn.ModuleList(self.cnn_stack2)


        self.mlp_layers1, self.last_layer1 = get_mlp_stack(out_size1, mlp_layers_size1)
        self.mlp_layers2, self.last_layer2 = get_mlp_stack(out_size2, mlp_layers_size2)

        self.mlp_layers3, self.last_layer3 = get_mlp_stack([out_size1[0]+out_size2[0]], mlp_layers_size3)
        self.softmax = nn.Softmax(dim=1)


    def get_cnn_stack(self, conv_layers_sizes, seqlength):
        layers = []
        stack_size = 0
        outsize = seqlength
        ## adding on the conv layers
        for i in range(0,len(conv_layers_sizes),3):
            layer = nn.Conv1d(in_channels = conv_layers_sizes[i+0],
                              out_channels = conv_layers_sizes[i+1],
                              kernel_size = conv_layers_sizes[i+2],stride = 1)
            outsize = outsize-conv_layers_sizes[i+2]+1
            layers.append(layer)
            dim2 = [(conv_layers_sizes[i+1]*(outsize))]

        cnn_stack = nn.ModuleList(layers)


        return cnn_stack, outsize

    def get_mlp_stack(self, out_size, mlp_layers_size):

        layers = []
        dim = out_size + mlp_layers_size

        for size_in, size_out in zip(dim[:-1], dim[1:]):
            layer = nn.Linear(size_in, size_out)
            layers.append(layer)

        mlp_layers = nn.ModuleList(layers)
        # Last layer
        last_layer = nn.Linear(dim[-1], 2)

        return mlp_layers, last_layer


    def get_feature_map(self, x1, stack):

        fm = stack[0](x1)
        fm = fm.reshape((fm.shape[0], fm.shape[1]*fm.shape[2]))
        return fm

    def get_feature_vector(self, x1, stack):

        for layer in stack:
            x1 = layer(x1)
        fv = x1.reshape((x1.shape[0], x1.shape[1]*x1.shape[2]))

        return fv


    def one_dim_forward(self, fm, fv, layers, last_layer):

        mlp_input = torch.cat([fm, fv], 1)
        for layer in layers:
            mlp_input = layer(mlp_input)
            mlp_input = F.tanh(mlp_input)
        mlp_output = last_layer(mlp_input)
        mlp_output = self.softmax(mlp_output)
        return mlp_output


    def forward(self,  x11, x12, x21, x22):

        # Get the TCR feature maps
        fm11, fm12 = self.get_feature_map(x11), self.get_feature_map(x12)
        fv11, fv12 = self.get_feature_vector(x11), self.get_feature_vector(x11)
        self.fv_tcr = fv1
        self.fv_pep = fv2
        mlp_output_11 = self.one_dim_forward(fm11, fv11, self.mlp_layers1, self.last_layer1)
        mlp_output_12 = self.one_dim_forward(fm12, fv11, self.mlp_layers1, self.last_layer1)
        mlp_output_21 = self.one_dim_forward(fm21, fv21, self.mlp_layers2, self.last_layer2)
        mlp_output_22 = self.one_dim_forward(fm22, fv21, self.mlp_layers2, self.last_layer2)

        immu_pred = self.one_dim_forward(self.fv_tcr, self.fv_pep, self.mlp_layers3, self.last_layer3)

        return mlp_output_11, mlp_output_12, mlp_output_21, mlp_output_22, immu_pred




def get_model(opt, inputs_size, model_state=None):

    if opt.model == 'CNN':
        model_class = DeepInfoMax
        model = model_class(conv_layers_sizes = opt.cnn_layers, mlp_layers_size=opt.layers_size, emb_size=opt.emb_size, 
                            data_dir = opt.data_dir, seqlength = opt.seqlength)
    elif opt.model == 'classifier':
        model_class = CNNClassifier
        model = model_class(conv_layers_sizes = opt.cnn_layers, mlp_layers_size=opt.layers_size, emb_size=opt.emb_size, class_number = opt.nbclasses, data_dir = opt.data_dir)
    elif opt.model == 'ae':
        model_class = CNNAutoEncoder
        model = model_class(conv_layers_sizes = opt.cnn_layers, emb_size=opt.emb_size, data_dir = opt.data_dir)
    elif opt.model == 'fullDIM':
        model_class = FullDIM
        model = model_class(conv_layers_sizes1 = opt.conv_layers_sizes1,
                            conv_layers_sizes2 = opt.conv_layers_sizes2,
                            mlp_layers_size1 = opt.mlp_layers_size1,
                            mlp_layers_size2 = opt.mlp_layers_size2,
                            mlp_layers_size3 = opt.mlp_layers_size3,
                            seqlength1 = opt.seqlength1,
                            seqlength2 = opt.seqlength2)

    else:
        raise NotImplementedError()

    if model_state is not None:
        model.load_state_dict(model_state)

    return model
