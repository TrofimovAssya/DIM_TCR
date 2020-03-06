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
    parser.add_argument('--dataset', choices=['classify','ae'], default='classify', help='Which dataset to use.')

    # Model specific options
    parser.add_argument('--layers-size', default=[25, 10], type=int, nargs='+', help='Number of layers to use.')
    parser.add_argument('--emb-size', default=10, type=int, help='The size of the feature vector')
    parser.add_argument('--out-channels', default=5, type=int, help='The number of kernels on the last layer')
    parser.add_argument('--loss', choices=['NLL','MSE'], default = 'NLL', help='The cost function to use')
    parser.add_argument('--weight-decay', default=1e-5, type=float, help='The size of the embeddings.')
    parser.add_argument('--model', choices=['classifier', 'ae'], default='classifier', help='Which sequence model to use.')
    parser.add_argument('--cpu', action='store_true', help='If we want to run on cpu.')
    parser.add_argument('--name', type=str, default=None, help="If we want to add a random str to the folder.")
    parser.add_argument('--gpu-selection', type=int, default=0, help="selectgpu")


    # Monitoring options
    parser.add_argument('--load-folder', help='The folder where to load and restart the training.')
    parser.add_argument('--save-dir', default='./benchmark123/', help='The folder where everything will be saved.')

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

    if opt.loss == 'NLL':
        criterion = torch.nn.NLLLoss()
    elif opt.loss == 'BCE':
        criterion = torch.nn.BCELoss()
    elif opt.loss == 'MSE':
        criterion = torch.nn.MSELoss()
    


    os.mkdir(f'{exp_dir}/tcr_embs/') #storing the representation

    if not opt.cpu:
        print ("Putting the model on gpu...")
        my_model.cuda(opt.gpu_selection)

    # The training.
    print ("Start training.")
    loss_monitoring_train = []
    loss_monitoring_valid = []
    #monitoring and predictions
    for t in range(epoch, opt.epoch):

        start_timer = time.time()
        loss_epoch_train = []
        loss_epoch_valid = []

        for no_b, mini in enumerate(dataset):
            if opt.model == 'classifier':
                inputs_tcr, targets = mini[0], mini[1]
                targets = Variable(targets, requires_grad=False).float()

            elif opt.model == 'ae':
                inputs_tcr = mini[0]

            inputs_tcr = Variable(inputs_tcr, requires_grad=False).float()
        

            if not opt.cpu: #putting the inputs and targets on gpu
                inputs_tcr = inputs_tcr.cuda(opt.gpu_selection)
                if opt.model == 'classifier':
                    targets = targets.cuda(opt.gpu_selection)


            # Forward pass: Compute predicted y by passing x to the model
            #import pdb; pdb.set_trace()
            inputs_tcr = inputs_tcr.squeeze().permute(0, 2, 1)

            y_pred = my_model(inputs_tcr)

            if opt.model=='classifier':
                loss = criterion(y_pred, targets)
            elif opt.model == 'ae':
                loss = criterion(y_pred, inputs_tcr)


            losstemp = loss.cpu().data.reshape(1,).numpy()[0]
            if no_b<99:
                loss_epoch_train.append(losstemp)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                loss_epoch_valid.append(losstemp)


            if no_b % 5 == 0:
                print (f"Doing epoch {t},examples{no_b}/{len(dataset)}.Loss:{loss.data.cpu().numpy().reshape(1,)[0]}")
            if no_b >98 :
                print (f"Doing epoch {t},examples{no_b}/{len(dataset)}.Loss:{loss.data.cpu().numpy().reshape(1,)[0]}")

            #### TO DO: saving of the representation?
            #np.save(f'{exp_dir}/tcr_embs/tcr_embs_batch_{no_b}',tcrembs.cpu().data.numpy())



        #print ("Saving the model...")
        loss_monitoring_train.append(np.mean(loss_epoch_train))
        loss_monitoring_valid.append(np.mean(loss_epoch_valid))
        monitoring.save_checkpoint(my_model, optimizer, t, opt, exp_dir)
        np.save(f'{exp_dir}/train_loss',np.array(loss_monitoring_train))
        np.save(f'{exp_dir}/valid_loss',np.array(loss_monitoring_valid))



if __name__ == '__main__':
    main()
