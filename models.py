import torch
import torch.nn.functional as F
from torch import nn
import argparse
from itertools import chain


class DeepInfoMax(nn.Module):

    def __init__(self, nb_samples=1, conv_layers_sizes = [20,10,5,10,5,14],mlp_layers_size = [25,10], emb_size = 10,data_dir ='.', seqlength = 27):
        super(DeepInfoMax, self).__init__()

        #self.sample = nb_samples
        #self.emb_size = emb_size

        ## adding on the conv layers
        #if len(conv_layers_sizes)>=3:
        #    self.conv1 = nn.Conv1d(in_channels = conv_layers_sizes[0],out_channels = conv_layers_sizes[1],kernel_size = conv_layers_sizes[2],stride = 1)
        #    self.cnn_stack = 1
        #if len(conv_layers_sizes)>=6:
        #    self.conv2 = nn.Conv1d(in_channels = conv_layers_sizes[3],out_channels = conv_layers_sizes[4], kernel_size=conv_layers_sizes[5],stride = 1)
        #    self.cnn_stack = 2
        #if len(conv_layers_sizes)>=9:
        #    self.conv3 = nn.Conv1d(in_channels = conv_layers_sizes[6],out_channels = conv_layers_sizes[7], kernel_size=conv_layers_sizes[8],stride = 1)
        #    self.cnn_stack = 3
        #if len(conv_layers_sizes)>=12:
        #    self.conv4 = nn.Conv1d(in_channels = conv_layers_sizes[9],out_channels = conv_layers_sizes[10], kernel_size=conv_layers_sizes[11],stride = 1)
        #    self.cnn_stack = 4
        #if len(conv_layers_sizes)>=15:
        #    self.conv5 = nn.Conv1d(in_channels = conv_layers_sizes[12],out_channels = conv_layers_sizes[13], kernel_size=conv_layers_sizes[14],stride = 1)
        #    self.cnn_stack = 5

        #layers = []
        #diself.sample = nb_samples
        self.emb_size = emb_size
        self.seqlength = seqlength

        ## adding on the conv layers
        if len(conv_layers_sizes)>=3:
            self.conv1 = nn.Conv1d(in_channels = conv_layers_sizes[0],out_channels = conv_layers_sizes[1],kernel_size = conv_layers_sizes[2],stride = 1)
            self.cnn_stack = 1
            outsize = self.seqlength-conv_layers_sizes[2]+1
            dim1 = [(conv_layers_sizes[1]*(outsize))]
            print (dim1)
        if len(conv_layers_sizes)>=6:
            self.conv2 = nn.Conv1d(in_channels = conv_layers_sizes[3],out_channels = conv_layers_sizes[4], kernel_size=conv_layers_sizes[5],stride = 1)
            self.cnn_stack = 2
            outsize = outsize-conv_layers_sizes[5]+1
            dim = [conv_layers_sizes[4]*(outsize)+dim1[0]]
            print (dim)
        if len(conv_layers_sizes)>=9:
            self.conv3 = nn.Conv1d(in_channels = conv_layers_sizes[6],out_channels = conv_layers_sizes[7], kernel_size=conv_layers_sizes[8],stride = 1)
            self.cnn_stack = 3
            outsize = outsize-conv_layers_sizes[8]+1
            dim = [(conv_layers_sizes[7]*outsize)+dim1[0]]
            print (dim)
        if len(conv_layers_sizes)>=12:
            self.conv4 = nn.Conv1d(in_channels = conv_layers_sizes[9],out_channels = conv_layers_sizes[10], kernel_size=conv_layers_sizes[11],stride = 1)
            self.cnn_stack = 4
            outsize = outsize-conv_layers_sizes[11]+1
            dim = [(conv_layers_sizes[10]*outsize)+dim1[0]]
            print (dim)
        if len(conv_layers_sizes)>=15:
            self.conv5 = nn.Conv1d(in_channels = conv_layers_sizes[12],out_channels = conv_layers_sizes[13], kernel_size=conv_layers_sizes[14],stride = 1)
            self.cnn_stack = 5
            outsize = outsize-conv_layers_sizes[14]+1
            dim = [(conv_layers_sizes[13]*outsize)+dim1[0]]
            print (dim)

        #self.conv1 = nn.Conv1d(in_channels = 20,out_channels = 10,kernel_size = 5,stride = 1)
        #self.conv2 = nn.Conv1d(in_channels = 10,out_channels = self.out_channels, kernel_size=14,stride = 1)
        layers = []
        self.dim = dim
        #dim = [(10*(27-5+1))+self.out_channels*self.emb_size] + layers_size # Adding the emb size*out_channels
        dim = dim + mlp_layers_size # Adding the emb size*out_channels

        #m = [(conv_layers_sizes[1]*(27-conv_layers_sizes[2]+1))+self.emb_size] + mlp_layers_size # Adding the emb size*out_channels

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
        if self.cnn_stack >1:
            fv = self.conv2(fv)
        if self.cnn_stack >2:
            fv = self.conv3(fv)
        if self.cnn_stack >3:
            fv = self.conv4(fv)
        if self.cnn_stack >4:
            fv = self.conv5(fv)

        #import pdb; pdb.set_trace()

        fv = fv.reshape((fv.shape[0], fv.shape[1]*fv.shape[2]))
        return fv

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
        mlp_output_1 = self.softmax(mlp_output_1)

        # Forward pass for fake
        mlp_input_2 = torch.cat([fm2, fv1], 1)
        for layer in self.mlp_layers:
            mlp_input_2 = layer(mlp_input_2)
            mlp_input_2 = F.tanh(mlp_input_2)
        mlp_output_2 = self.last_layer(mlp_input_2)
        mlp_output_2 = self.softmax(mlp_output_2)


        return mlp_output_1, mlp_output_2


class CNNClassifier(nn.Module):

    def __init__(self, nb_samples=1, conv_layers_sizes = [20,10,15,10,5,12],mlp_layers_size = [25,10], emb_size = 10, class_number = 14, data_dir ='.'):
        super(CNNClassifier, self).__init__()

        self.sample = nb_samples
        self.emb_size = emb_size
        self.class_number = class_number

        ## adding on the conv layers
        if len(conv_layers_sizes)>=3:
            self.conv1 = nn.Conv1d(in_channels = conv_layers_sizes[0],out_channels = conv_layers_sizes[1],kernel_size = conv_layers_sizes[2],stride = 1)
            self.cnn_stack = 1
            outsize = 27-conv_layers_sizes[2]+1
            dim = [(conv_layers_sizes[1]*(outsize))]
            print (dim)
        if len(conv_layers_sizes)>=6:
            self.conv2 = nn.Conv1d(in_channels = conv_layers_sizes[3],out_channels = conv_layers_sizes[4], kernel_size=conv_layers_sizes[5],stride = 1)
            self.cnn_stack = 2
            outsize = outsize-conv_layers_sizes[5]+1
            dim = [conv_layers_sizes[4]*(outsize)]
            print (dim)
        if len(conv_layers_sizes)>=9:
            self.conv3 = nn.Conv1d(in_channels = conv_layers_sizes[6],out_channels = conv_layers_sizes[7], kernel_size=conv_layers_sizes[8],stride = 1)
            self.cnn_stack = 3
            outsize = outsize-conv_layers_sizes[8]+1
            dim = [(conv_layers_sizes[7]*outsize)]
            print (dim)
        if len(conv_layers_sizes)>=12:
            self.conv4 = nn.Conv1d(in_channels = conv_layers_sizes[9],out_channels = conv_layers_sizes[10], kernel_size=conv_layers_sizes[11],stride = 1)
            self.cnn_stack = 4
            outsize = outsize-conv_layers_sizes[11]+1
            dim = [(conv_layers_sizes[10]*outsize)]
            print (dim)
        if len(conv_layers_sizes)>=15:
            self.conv5 = nn.Conv1d(in_channels = conv_layers_sizes[12],out_channels = conv_layers_sizes[13], kernel_size=conv_layers_sizes[14],stride = 1)
            self.cnn_stack = 5
            outsize = outsize-conv_layers_sizes[14]+1
            dim = [(conv_layers_sizes[13]*outsize)]
            print (dim)

        #self.conv1 = nn.Conv1d(in_channels = 20,out_channels = 10,kernel_size = 5,stride = 1)
        #self.conv2 = nn.Conv1d(in_channels = 10,out_channels = self.out_channels, kernel_size=14,stride = 1)
        layers = []
        self.dim = dim
        #dim = [(10*(27-5+1))+self.out_channels*self.emb_size] + layers_size # Adding the emb size*out_channels
        dim = dim + mlp_layers_size # Adding the emb size*out_channels

        for size_in, size_out in zip(dim[:-1], dim[1:]):
            layer = nn.Linear(size_in, size_out)
            layers.append(layer)

        self.mlp_layers = nn.ModuleList(layers)

        # Last layer
        self.last_layer = nn.Linear(dim[-1], self.class_number)
        # self.softmax = nn.Softmax(dim=1)


    def get_feature_vector(self, x1):

        fv = self.conv1(x1)
        if self.cnn_stack >1:
            fv = self.conv2(fv)
        if self.cnn_stack >2:
            fv = self.conv3(fv)
        if self.cnn_stack >3:
            fv = self.conv4(fv)
        if self.cnn_stack >4:
            fv = self.conv5(fv)


        fv = fv.reshape((fv.shape[0], fv.shape[1]*fv.shape[2]))
        return fv

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


        ## adding on the conv layers
        if len(conv_layers_sizes)>=3:
            self.conv1 = nn.Conv1d(in_channels = conv_layers_sizes[0],out_channels = conv_layers_sizes[1],kernel_size = conv_layers_sizes[2],stride = 1)
            self.deconv1 = nn.ConvTranspose1d(in_channels = conv_layers_sizes[1],out_channels = conv_layers_sizes[0],kernel_size = conv_layers_sizes[2],stride = 1)
            self.cnn_stack = 1

        if len(conv_layers_sizes)>=6:
            self.conv2 = nn.Conv1d(in_channels = conv_layers_sizes[3],out_channels = conv_layers_sizes[4], kernel_size=conv_layers_sizes[5],stride = 1)
            self.deconv2 = nn.ConvTranspose1d(in_channels = conv_layers_sizes[4],out_channels = conv_layers_sizes[3],kernel_size = conv_layers_sizes[5],stride = 1)
            self.cnn_stack = 2
        if len(conv_layers_sizes)>=9:
            self.conv3 = nn.Conv1d(in_channels = conv_layers_sizes[6],out_channels = conv_layers_sizes[7], kernel_size=conv_layers_sizes[8],stride = 1)
            self.deconv3 = nn.ConvTranspose1d(in_channels = conv_layers_sizes[7],out_channels = conv_layers_sizes[6],kernel_size = conv_layers_sizes[8],stride = 1)
            self.cnn_stack = 3
        if len(conv_layers_sizes)>=12:
            self.conv4 = nn.Conv1d(in_channels = conv_layers_sizes[9],out_channels = conv_layers_sizes[10], kernel_size=conv_layers_sizes[11],stride = 1)
            self.deconv4 = nn.ConvTranspose1d(in_channels = conv_layers_sizes[10],out_channels = conv_layers_sizes[9],kernel_size = conv_layers_sizes[11],stride = 1)
            self.cnn_stack = 4
        if len(conv_layers_sizes)>=15:
            self.conv5 = nn.Conv1d(in_channels = conv_layers_sizes[12],out_channels = conv_layers_sizes[13], kernel_size=conv_layers_sizes[14],stride = 1)
            self.deconv5 = nn.ConvTranspose1d(in_channels = conv_layers_sizes[13],out_channels = conv_layers_sizes[12],kernel_size = conv_layers_sizes[14],stride = 1)
            self.cnn_stack = 5



    def encode(self, x1):

        fv = self.conv1(x1)
        if self.cnn_stack>1:
            fv = self.conv2(fv)
        if self.cnn_stack >2:
            fv = self.conv3(fv)
        if self.cnn_stack >3:
            fv = self.conv4(fv)
        if self.cnn_stack >4:
            fv = self.conv5(fv)

        #import pdb; pdb.set_trace()

        #fv = fv.reshape((fv.shape[0], fv.shape[1]*fv.shape[2]))
        return fv

    def decode(self, fv):

        if self.cnn_stack >4:
            fv = self.deconv5(fv)
        if self.cnn_stack >3:
            fv = self.deconv4(fv)
        if self.cnn_stack >2:
            fv = self.deconv3(fv)
        if self.cnn_stack >1:
            fv = self.deconv2(fv)

        fv = self.deconv1(fv)
        output = fv
        #output = fv.reshape((fv.shape[0], fv.shape[1]*fv.shape[2]))
        return output

    def forward(self,  x1):

        # Get the feature maps

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
        cnn_stack = []
        stack_size = 0
        ## adding on the conv layers
        if len(conv_layers_sizes)>=3:
            conv1 = nn.Conv1d(in_channels = conv_layers_sizes[0],out_channels = conv_layers_sizes[1],kernel_size = conv_layers_sizes[2],stride = 1)
            cnn_stack.append(conv1)
            stack_size = 1
            outsize = seqlength-conv_layers_sizes[2]+1
            dim = [(conv_layers_sizes[1]*(outsize))]
            print (dim)
        if len(conv_layers_sizes)>=6:
            conv2 = nn.Conv1d(in_channels = conv_layers_sizes[3],out_channels = conv_layers_sizes[4], kernel_size=conv_layers_sizes[5],stride = 1)
            cnn_stack.append(conv2)
            stack_size = 2
            outsize = outsize-conv_layers_sizes[5]+1
            dim = [conv_layers_sizes[4]*(outsize)+dim1[0]]
            print (dim)
        if len(conv_layers_sizes)>=9:
            conv3 = nn.Conv1d(in_channels = conv_layers_sizes[6],out_channels = conv_layers_sizes[7], kernel_size=conv_layers_sizes[8],stride = 1)
            cnn_stack.append(conv3)
            stack_size = 3
            outsize = outsize-conv_layers_sizes[8]+1
            dim = [(conv_layers_sizes[7]*outsize)+dim1[0]]
            print (dim)
        if len(conv_layers_sizes)>=12:
            conv4 = nn.Conv1d(in_channels = conv_layers_sizes[9],out_channels = conv_layers_sizes[10], kernel_size=conv_layers_sizes[11],stride = 1)
            cnn_stack.append(conv4)
            stack_size = 4
            outsize = outsize-conv_layers_sizes[11]+1
            dim = [(conv_layers_sizes[10]*outsize)+dim1[0]]
            print (dim)
        if len(conv_layers_sizes)>=15:
            conv5 = nn.Conv1d(in_channels = conv_layers_sizes[12],out_channels = conv_layers_sizes[13], kernel_size=conv_layers_sizes[14],stride = 1)
            cnn_stack.append(conv5)
            stack_size = 5
            outsize = outsize-conv_layers_sizes[14]+1
            dim = [(conv_layers_sizes[13]*outsize)+dim1[0]]
            print (dim)

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
    elif opt.model == 'full':
        model_class = Contrastive
        model = model_class(mlp_layers_sizes = opt.cnn_layers, emb_size=opt.emb_size, data_dir = opt.data_dir)
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
