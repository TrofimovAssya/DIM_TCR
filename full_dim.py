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
    parser.add_argument('--1cnn-layers', default=[20,10,5,10,5,14], type=int, nargs='+', help='Number of layers to use.')
    parser.add_argument('--2cnn-layers', default=[20,10,5,10,10,13], type=int, nargs='+', help='Number of layers to use.')
    parser.add_argument('--layers-size', default=[25, 10], type=int, nargs='+', help='Number of layers to use.')
    parser.add_argument('--emb-size', default=10, type=int, help='The size of the feature vector')
    parser.add_argument('--out-channels', default=5, type=int, help='The number of kernels on the last layer')
    parser.add_argument('--loss', choices=['NLL'], default = 'NLL', help='The cost function to use')
    parser.add_argument('--weight-decay', default=1e-5, type=float, help='The size of the embeddings.')
    parser.add_argument('--model', choices=['fullDIM'], default='fullDIM', help='Which sequence model to use.')
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
    seed = opt.seed
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)

    exp_dir = opt.load_folder
    if exp_dir is None:
        exp_dir = monitoring.create_experiment_folder(opt)

    print ("Getting the dataset...")
    dataset = datasets.get_dataset(opt,exp_dir)

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
            ### TODO: make an optional immunogen prediction so that it's
            ### possible to train without the immunogen for some examples

            inputs_tcr, inputs_pep, immunogen = mini[0], mini[1], mini[2]

            ### creating the positive and negative sets for the DIM-tcr part

            inputs_tcr_pos = Variable(inputs_tcr, requires_grad=False).float()
            p = np.random.permutation(np.arange(inputs_tcr.shape[0]))
            inputs_tcr_neg = Variable(inputs_tcr[p], requires_grad=False).float()
            targets_pos_tcr = np.zeros((inputs_tcr_pos.shape[0],2))
            targets_neg_tcr = np.zeros((inputs_tcr_neg.shape[0],2))
            targets_pos_tcr[:,1]+=1
            targets_neg_tcr[:,0]+=1
            targets_pos_tcr = torch.FloatTensor(targets_pos_tcr)
            targets_pos_tcr = Variable(targets_pos_tcr, requires_grad=False).float()
            targets_neg_tcr = torch.FloatTensor(targets_neg_tcr)
            targets_neg_tcr = Variable(targets_neg_tcr, requires_grad=False).float()

            ### creating the positive and negative sets for the DIM-pep part
            inputs_pep_pos = Variable(inputs_pep, requires_grad=False).float()
            p = np.random.permutation(np.arange(inputs_pep.shape[0]))
            inputs_pep_neg = Variable(inputs_pep[p], requires_grad=False).float()
            targets_pos_pep = np.zeros((inputs_pep_pos.shape[0],2))
            targets_neg_pep = np.zeros((inputs_pep_neg.shape[0],2))
            targets_pos_pep[:,1]+=1
            targets_neg_pep[:,0]+=1
            targets_pos_pep = torch.FloatTensor(targets_pos_pep)
            targets_pos_pep = Variable(targets_pos_pep, requires_grad=False).float()
            targets_neg_pep = torch.FloatTensor(targets_neg_pep)
            targets_neg_pep = Variable(targets_neg_pep, requires_grad=False).float()

            ### creating the immunogenicity target
            immunogen = Variable(immunogen, requires_grad=False).float()


            if not opt.cpu:
                inputs_tcr_pos = inputs_tcr_pos.cuda(opt.gpu_selection)
                inputs_tcr_neg = inputs_tcr_neg.cuda(opt.gpu_selection)
                targets_pos_tcr = targets_pos_tcr.cuda(opt.gpu_selection)
                targets_neg_tcr = targets_neg_tcr.cuda(opt.gpu_selection)

                inputs_pep_pos = inputs_pep_pos.cuda(opt.gpu_selection)
                inputs_pep_neg = inputs_pep_neg.cuda(opt.gpu_selection)
                targets_pos_pep = targets_pos_pep.cuda(opt.gpu_selection)
                targets_neg_pep = targets_neg_pep.cuda(opt.gpu_selection)

                immunogen = immunogen.cuda(opt.gpu_selection)


            # Forward pass: Compute predicted y by passing x to the model
            #import pdb; pdb.set_trace()
            inputs_tcr_pos = inputs_tcr_pos.squeeze().permute(0, 2, 1)
            inputs_tcr_neg = inputs_tcr_neg.squeeze().permute(0, 2, 1)
            inputs_pep_pos = inputs_pep_pos.squeeze().permute(0, 2, 1)
            inputs_pep_neg = inputs_pep_neg.squeeze().permute(0, 2, 1)
            y_pred11, y_pred12, y_pred21, y_pred22, immu_pred = my_model(inputs_tcr_pos,inputs_tcr_neg, inputs_pep_pos,inputs_pep_neg)#transform to float?
            #y_pred1 = y_pred1.float()
            #y_pred2 = y_pred2.float()

            #import pdb; pdb.set_trace()
            #y_pred1 = y_pred1.permute(1,0)
            #y_pred2 = y_pred2.permute(1,0)

            loss = criterion(y_pred11, targets_pos_tcr)
            losstemp = loss.cpu().data.reshape(1,).numpy()[0]
            loss_epoch.append(losstemp)

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            loss = criterion(y_pred12, targets_neg_tcr)
            losstemp = loss.cpu().data.reshape(1,).numpy()[0]
            loss_epoch.append(losstemp)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss = criterion(y_pred21, targets_pos_pep)
            losstemp = loss.cpu().data.reshape(1,).numpy()[0]
            loss_epoch.append(losstemp)

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            loss = criterion(y_pred22, targets_neg_pep)
            losstemp = loss.cpu().data.reshape(1,).numpy()[0]
            loss_epoch.append(losstemp)


            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss = criterion2(immu_pred, immunogen)
            losstemp = loss.cpu().data.reshape(1,).numpy()[0]
            loss_epoch.append(losstemp)


            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()



            if no_b % 5 == 0:
                print (f"Doing epoch {t},examples{no_b}/{len(dataset)}.Loss:{loss.data.cpu().numpy().reshape(1,)[0]}")


            kmerembs = my_model.fv_tcr
            np.save(f'{exp_dir}/tcr_embs/tcr_embs_batch_{no_b}',kmerembs.cpu().data.numpy())
            kmerembs = my_model.fv_pep
            np.save(f'{exp_dir}/pep_embs/pep_embs_batch_{no_b}',kmerembs.cpu().data.numpy())



        #print ("Saving the model...")
        loss_monitoring.append(np.mean(loss_epoch))
        monitoring.save_checkpoint(my_model, optimizer, t, opt, exp_dir)
        np.save(f'{exp_dir}/train_loss',np.array(loss_monitoring))



if __name__ == '__main__':
    main()
