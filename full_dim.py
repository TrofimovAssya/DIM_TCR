#!/usr/bin/env python
import torch
import pdb
import numpy as np
from torch.autograd import Variable
import os
import argparse
import datasets
from torch import nn
import models
import pickle
import time
import monitoring
#
def build_parser():
    parser = argparse.ArgumentParser(description="")

    ### Hyperparameter options
    parser.add_argument('--epoch', default=10, type=int, help='The number of epochs we want ot train the network.')
    parser.add_argument('--seed', default=260389, type=int, help='Seed for random initialization and stuff.')
    parser.add_argument('--batch-size', default=1, type=int, help="The batch size.")
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')

    ### Dataset specific options
    parser.add_argument('--data-dir', default='./data/', help='The folder contaning the dataset.')
    parser.add_argument('--data-file', default='.', help='The data file with the dataset.')
    parser.add_argument('--dataset', choices=['contrastive'], default='contrastive', help='Which dataset to use.')
    parser.add_argument('--suffix', type=str, default='_gd', help='Which dataset suffix to use')
    parser.add_argument('--seqtup', type=int, default=[27,11], help='Sequence lengths')

    # Model specific options
    parser.add_argument('--cnn-layers', default=[20,10,5,10,5,14], type=int, nargs='+', help='Number of layers to use.')
    parser.add_argument('--cnn-layers1', default=[20,10,5,10,5,14], type=int, nargs='+', help='Number of layers to use.')
    parser.add_argument('--cnn-layers2', default=[20,10,5,10,10,3], type=int, nargs='+', help='Number of layers to use.')
    parser.add_argument('--layers-size', default=[25, 10], type=int, nargs='+', help='Number of layers to use.')
    parser.add_argument('--emb-size', default=10, type=int, help='The size of the feature vector')
    parser.add_argument('--out-channels', default=5, type=int, help='The number of kernels on the last layer')
    parser.add_argument('--loss', choices=['BCE', 'contrastive'], default = 'BCE', help='The cost function to use')
    parser.add_argument('--weight-decay', default=1e-5, type=float, help='The size of the embeddings.')
    parser.add_argument('--model', choices=['full','contrastive','CNN'],default='CNN', help='Which sequence model to use.')
    parser.add_argument('--cpu', action='store_true', help='If we want to run on cpu.')
    parser.add_argument('--name', type=str, default=None, help="If we want to add a random str to the folder.")
    parser.add_argument('--gpu-selection', type=int, default=0, help="selectgpu")


    # Monitoring options
    parser.add_argument('--load-folder', help='The folder where to load and restart the training.')
    parser.add_argument('--load-folder1', help='The folder where to load the TCR dim.')
    parser.add_argument('--load-folder2', help='The folder where to load the peptide dim')
    parser.add_argument('--save-dir', default='./testing123/', help='The folder where everything will be saved.')

    return parser

def parse_args(argv):

    if type(argv) == list or argv is None:
        opt = build_parser().parse_args(argv)
    else:
        opt = argv

    return opt

def main(argv=None):

    opt = parse_args(argv)
    # TODO: set the seed
    seed = opt.seed
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)

    exp_dir = opt.load_folder
    if exp_dir is None: # we create a new folder if we don't load.
        exp_dir = monitoring.create_experiment_folder(opt)

    #Getting the TCR dim model
    print ("Getting the model #1...")
    filename = os.path.join(opt.load_folder1, 'checkpoint.pth.tar')
    print (f"=> loading model # 1 from checkpoint '{filename}'")
    model1 = torch.load(filename)

    # Getting the peptide dim model
    print ("Getting the model #2...")
    filename = os.path.join(opt.load_folder2, 'checkpoint.pth.tar')
    print (f"=> loading model # 2 from checkpoint '{filename}'")
    model2 = torch.load(filename)


    class FullDim(nn.Module):

        def __init__(self, model1, model2, opt):
            super(FullDim, self).__init__()

            self.checkpoint1 = model1
            model1_state = self.checkpoint1['state_dict']
            optimizer1_state = self.checkpoint1['optimizer']
            opt.seqlength = opt.seqtup[0]
            opt.cnn_layers = opt.cnn_layers1
            self.model1 = models.get_model(opt, model1_state)

            self.checkpoint2 = model2
            model2_state = self.checkpoint2['state_dict']
            optimizer2_state = self.checkpoint2['optimizer']
            opt.seqlength = opt.seqtup[1]
            opt.cnn_layers = opt.cnn_layers2
            self.model2 = models.get_model(opt, model2_state)

            self.cos = nn.CosineSimilarity(dim=1, eps=1e-6) 


        def forward(self,  x1, x2):

            # Get the feature maps

            fv1 = self.model1.get_feature_vector(x1)
            fv2 = self.model2.get_feature_vector(x2)

            ### let's try using a cosine similarity as a measure of distance
            output = self.cos(fv1, fv2)
            return output

    #Making the final model
    my_model = FullDim(model1, model2, opt)
    optimizer = torch.optim.RMSprop(my_model.parameters(), lr=opt.lr,
                                    weight_decay=opt.weight_decay)

    # creating the dataset
    print ("Getting the dataset...")
    dataset = datasets.get_dataset(opt,exp_dir)

    criterion = torch.nn.MSELoss()



    if not opt.cpu:
        print ("Putting the model on gpu...")
        my_model.cuda(opt.gpu_selection)

    # The training.
    print ("Start training.")
    loss_monitoring_train = []
    loss_monitoring_valid = []
    #monitoring and predictions
    epoch = 0
    for t in range(epoch, opt.epoch):

        start_timer = time.time()
        loss_epoch_train = []
        loss_epoch_valid = []
        for no_b, mini in enumerate(dataset):

            optimizer.zero_grad()
            #import pdb;pdb.set_trace()
            inputs_tcr, inputs_pep, targets = mini[0], mini[1], mini[2]

            inputs_tcr = Variable(inputs_tcr, requires_grad=False).float()
            inputs_pep = Variable(inputs_pep, requires_grad=False).float()
            targets = Variable(targets, requires_grad=False).float()

            if not opt.cpu:
                inputs_tcr = inputs_tcr.cuda(opt.gpu_selection)
                inputs_pep = inputs_pep.cuda(opt.gpu_selection)
                targets = targets.cuda(opt.gpu_selection)
            inputs_tcr = inputs_tcr.squeeze().permute(0, 2, 1)
            inputs_pep = inputs_pep.squeeze().permute(0, 2, 1)
            targets = targets.squeeze()

            ### Passing feature vectors to classification
            y_pred = my_model(inputs_tcr,inputs_pep)
            #import pdb; pdb.set_trace()

            loss = criterion(y_pred, targets)
            losstemp = loss.cpu().data.reshape(1,).numpy()[0]
            if len(dataset)-1==no_b:
                loss_epoch_valid.append(losstemp)
                print (f"**** Validation loss: {losstemp} ****")
            else:
                loss_epoch_train.append(losstemp)
                loss.backward()
                optimizer.step()


            if no_b % 39 == 0:
                print (f"Doing epoch {t},examples{no_b}/{len(dataset)}.Loss:{loss.data.cpu().numpy().reshape(1,)[0]}")



        #print ("Saving the model...")
        loss_monitoring_train.append(np.mean(loss_epoch_train))
        loss_monitoring_valid.append(np.mean(loss_epoch_valid))
        monitoring.save_checkpoint(my_model, optimizer, t, opt, exp_dir)
        np.save(f'{exp_dir}/train_loss',np.array(loss_monitoring_train))
        np.save(f'{exp_dir}/valid_loss',np.array(loss_monitoring_valid))



if __name__ == '__main__':
    main()
