#!/usr/bin/env python
import torch
import pdb
import numpy as np
from torch.autograd import Variable
import os
import argparse
import datasets
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
    parser.add_argument('--dataset', choices=['tcr'], default='tcr', help='Which dataset to use.')
    parser.add_argument('--suffix', type=str, default='_gd', help='Which dataset suffix to use')
    parser.add_argument('--datatype', type=str, default='_tcr', help='Which biological sequence to use')
    parser.add_argument('--seqlength', type=int, default=27, help='The initial length of the biological sequence')


    # Model specific options
    parser.add_argument('--cnn-layers', default=[20,10,5,10,10,3], type=int, nargs='+', help='Number of layers to use.')
    parser.add_argument('--layers-size', default=[25, 10], type=int, nargs='+', help='Number of layers to use.')
    parser.add_argument('--emb-size', default=10, type=int, help='The size of the feature vector')
    parser.add_argument('--out-channels', default=5, type=int, help='The number of kernels on the last layer')
    parser.add_argument('--loss', choices=['NLL'], default = 'NLL', help='The cost function to use')
    parser.add_argument('--weight-decay', default=1e-5, type=float, help='The size of the embeddings.')
    parser.add_argument('--model', choices=['CNN'], default='CNN', help='Which sequence model to use.')
    parser.add_argument('--cpu', action='store_true', help='If we want to run on cpu.')
    parser.add_argument('--name', type=str, default=None, help="If we want to add a random str to the folder.")
    parser.add_argument('--gpu-selection', type=int, default=0, help="selectgpu")


    # Monitoring options
    parser.add_argument('--load-folder', help='The folder where to load and restart the training.')
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

    # creating the dataset
    print ("Getting the dataset...")
    dataset = datasets.get_dataset(opt,exp_dir)

    # Creating a model
    print ("Getting the model...")
    my_model, optimizer, epoch, opt = monitoring.load_checkpoint(exp_dir, opt )
    print (my_model)

    criterion = torch.nn.NLLLoss()
    criterion = torch.nn.BCELoss()


    os.mkdir(f'{exp_dir}/kmer_embs/') #storing the representation

    if not opt.cpu:
        print ("Putting the model on gpu...")
        my_model.cuda(opt.gpu_selection)

    # The training.
    print ("Start training.")
    loss_monitoring = []
    #monitoring and predictions
    for t in range(epoch, opt.epoch):

        start_timer = time.time()
        loss_epoch = []
        for no_b, mini in enumerate(dataset):

            inputs_tcr = mini[0]

            inputs_tcr_pos = Variable(inputs_tcr, requires_grad=False).float()
            p = np.random.permutation(np.arange(inputs_tcr.shape[0]))
            inputs_tcr_neg = Variable(inputs_tcr[p], requires_grad=False).float()
            targets_pos = np.zeros((inputs_tcr_pos.shape[0],2))
            targets_neg = np.zeros((inputs_tcr_neg.shape[0],2))
            targets_pos[:,1]+=1
            targets_neg[:,0]+=1
            targets_pos = torch.FloatTensor(targets_pos)
            targets_pos = Variable(targets_pos, requires_grad=False).float()
            targets_neg = torch.FloatTensor(targets_neg)
            targets_neg = Variable(targets_neg, requires_grad=False).float()


            if not opt.cpu:
                inputs_tcr_pos = inputs_tcr_pos.cuda(opt.gpu_selection)
                inputs_tcr_neg = inputs_tcr_neg.cuda(opt.gpu_selection)
                targets_pos = targets_pos.cuda(opt.gpu_selection)
                targets_neg = targets_neg.cuda(opt.gpu_selection)

            # Forward pass: Compute predicted y by passing x to the model
            #import pdb; pdb.set_trace()
            inputs_tcr_pos = inputs_tcr_pos.squeeze().permute(0, 2, 1)
            inputs_tcr_neg = inputs_tcr_neg.squeeze().permute(0, 2, 1)
            y_pred1, y_pred2 = my_model(inputs_tcr_pos,inputs_tcr_neg)#transform to float?
            #y_pred1 = y_pred1.float()
            #y_pred2 = y_pred2.float()

            #import pdb; pdb.set_trace()
            #y_pred1 = y_pred1.permute(1,0)
            #y_pred2 = y_pred2.permute(1,0)

            loss = criterion(y_pred1, targets_pos)
            losstemp = loss.cpu().data.reshape(1,).numpy()[0]
            loss_epoch.append(losstemp)

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            loss = criterion(y_pred2, targets_neg)
            losstemp = loss.cpu().data.reshape(1,).numpy()[0]
            loss_epoch.append(losstemp)


            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            if no_b % 5 == 0:
                print (f"Doing epoch {t},examples{no_b}/{len(dataset)}.Loss:{loss.data.cpu().numpy().reshape(1,)[0]}")


            kmerembs = my_model.fv
            np.save(f'{exp_dir}/kmer_embs/kmer_embs_batch_{no_b}',kmerembs.cpu().data.numpy())



        #print ("Saving the model...")
        loss_monitoring.append(np.mean(loss_epoch))
        monitoring.save_checkpoint(my_model, optimizer, t, opt, exp_dir)
        np.save(f'{exp_dir}/train_loss',np.array(loss_monitoring))



if __name__ == '__main__':
    main()
