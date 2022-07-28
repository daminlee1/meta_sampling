"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import argparse
import os, sys
import torch
import numpy as np

from utils.config import create_config
from utils.common_config import get_criterion, get_model, get_train_dataset,\
                                get_val_dataset, get_train_dataloader,\
                                get_val_dataloader, get_train_transformations,\
                                get_val_transformations, get_optimizer,\
                                adjust_learning_rate
from utils.evaluate_utils import contrastive_evaluate
from utils.memory import MemoryBank
from utils.train_utils import simclr_train
from utils.utils import fill_memory_bank
from termcolor import colored
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
colors_list = list(colors._colors_full_map.values())
import csv
from os import listdir
from os.path import isfile, join

# Parser
parser = argparse.ArgumentParser(description='SimCLR')
parser.add_argument('--config_env',
                    help='Config file for the environment')
parser.add_argument('--config_exp',
                    help='Config file for the experiment')
parser.add_argument("--save_features",
                    help="save the features of each image", type=bool, default=False)
parser.add_argument("--plot_nn_dist",
                    help="plot the distance of nearest neighbors", type=bool, default=False)
parser.add_argument("--gpus", type=int, nargs='+', default=[0], help="List of device ids.")
args = parser.parse_args()

def save_feature(memory_bank, imageset, csv_name):
    base_features = memory_bank.features.cpu().numpy()
    print("Base feature shapes : {},{}".format(base_features.shape[0],base_features.shape[1]))
    print("Saving base_features of trainDB")
    with open(imageset, 'r', encoding='UTF-8') as f:
        img_names = f.readlines()
        for n in img_names:
            n = n.replace("\n","")
    with open(csv_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        print("num of base_features : ", base_features.shape[0])
        for i, feat in enumerate(base_features):
            csv_data = [img_names[i]]+feat.tolist()
            writer.writerows([csv_data])
    print("finish saving feature")

def plot_nn_distance(indices, memory_bank, save_dir):
    base_features = memory_bank.features.cpu().numpy()
    feature_dim = base_features.shape[1]
    for i, index in enumerate(indices):
        if i % 100 != 0:
            continue
        plt.clf()
        l2_min_list = []
        for j,idx in enumerate(index):
            if j == 0:
                continue
            sa = 0
            for f in range(feature_dim):
                sa += pow(base_features[i,f] - base_features[idx,f],2)
            sa = pow(sa, 0.5)
            l2_min_list.append(sa)
        x_range = np.arange(1,len(index))
        plt.bar(x_range,np.array(l2_min_list), label=str(i), color=colors_list[0])
        plt.xlabel("nn index")
        plt.ylabel("L2 norm")
        plt.legend(loc="upper right")
        fig_name = save_dir+"img"+str(i)+"_L2norm.jpg"
        plt.title("img"+str(i)+"_L2norm")
        plt.savefig(fig_name)

def main():
    # Retrieve config file
    p = create_config(args.config_env, args.config_exp)
    print(colored(p, 'red'))
    
    # Model
    print(colored('Retrieve model', 'blue'))
    model = get_model(p)
    print('Model is {}'.format(model.__class__.__name__))
    print('Model parameters: {:.2f}M'.format(sum(p.numel() for p in model.parameters()) / 1e6))
    print(model)
    # multi-gpu
    print("GPUS : ", args.gpus)
    if len(args.gpus) == 1:
        torch.cuda.set_device(int(args.gpus[0]))
        model = model.cuda()
    elif len(args.gpus) > 1:
        gpus_use = [int(g) for g in args.gpus]
        print(gpus_use)
        model = torch.nn.DataParallel(model,device_ids=gpus_use)
        model = model.cuda()
        model.to(f'cuda:{model.device_ids[0]}')
    else:
        print("gpus is none")
        sys.exit(1)
   
    # CUDNN
    print(colored('Set CuDNN benchmark', 'blue')) 
    torch.backends.cudnn.benchmark = True
    
    # Dataset
    print(colored('Retrieve dataset', 'blue'))
    train_transforms = get_train_transformations(p)
    print('Train transforms:', train_transforms)
    val_transforms = get_val_transformations(p)
    print('Validation transforms:', val_transforms)
    train_dataset = get_train_dataset(p, train_transforms, to_augmented_dataset=True,
                                        split='train+unlabeled') # Split is for stl-10
    val_dataset = get_val_dataset(p, val_transforms) 
    train_dataloader = get_train_dataloader(p, train_dataset)
    val_dataloader = get_val_dataloader(p, val_dataset)
    print('Dataset contains {} train samples {} valid samples'.format(len(train_dataset), len(val_dataset)))
    
    # Memory Bank
    print(colored('Build MemoryBank', 'blue'))
    base_dataset = get_train_dataset(p, val_transforms, split='train') # Dataset w/o augs for knn eval
    #not shuffle
    base_dataloader = get_val_dataloader(p, base_dataset) 
    memory_bank_base = MemoryBank(len(base_dataset), 
                                p['model_kwargs']['features_dim'],
                                p['num_classes'], p['criterion_kwargs']['temperature'])
    memory_bank_base.cuda()

    memory_bank_val = MemoryBank(len(val_dataset),
                                p['model_kwargs']['features_dim'],
                                p['num_classes'], p['criterion_kwargs']['temperature'])
    memory_bank_val.cuda()

    # Criterion
    print(colored('Retrieve criterion', 'blue'))
    criterion = get_criterion(p)
    print('Criterion is {}'.format(criterion.__class__.__name__))
    criterion = criterion.cuda()

    # Optimizer and scheduler
    print(colored('Retrieve optimizer', 'blue'))
    optimizer = get_optimizer(p, model)
    print(optimizer)
 
    # Checkpoint
    if os.path.exists(p['pretext_checkpoint']):
        print(colored('Restart from checkpoint {}'.format(p['pretext_checkpoint']), 'blue'))
        checkpoint = torch.load(p['pretext_checkpoint'], map_location='cpu')
        optimizer.load_state_dict(checkpoint['optimizer'])
        model.load_state_dict(checkpoint['model'])
        model.cuda()
        start_epoch = checkpoint['epoch']
    else:
        print(colored('No checkpoint file at {}'.format(p['pretext_checkpoint']), 'blue'))
        start_epoch = 0
        model = model.cuda()
    
    # Training
    print(colored('Starting main loop', 'blue'))
    loss_minimum = 999
    for epoch in range(start_epoch, p['epochs']):
        print(colored('Epoch %d/%d' %(epoch, p['epochs']), 'yellow'))
        print(colored('-'*15, 'yellow'))

        # Adjust lr
        lr = adjust_learning_rate(p, optimizer, epoch)
        print('Adjusted learning rate to {:.5f}'.format(lr))
        
        # Train
        print('Train ...')
        loss = simclr_train(train_dataloader, model, criterion, optimizer, epoch)

        if loss_minimum > loss:
            print("Saving model - now loss : {} , current minimum loss : {}".format(loss, loss_minimum))
            loss_minimum = loss
            torch.save(model.state_dict(), os.path.join(p['pretext_dir'],"model_epoch"+str(epoch)+"_"+str(int(loss*100))+".tar"))
        # Fill memory bank
        #print('Fill memory bank for kNN...')
        #fill_memory_bank(base_dataloader, model, memory_bank_base)

        # Evaluate (To monitor progress - Not for validation)
        #print('Evaluate ...')
        #top1 = contrastive_evaluate(val_dataloader, model, memory_bank_base)
        #print('Result of kNN evaluation is %.2f' %(top1)) 
        
        # Checkpoint
        print('Checkpoint ...')
        torch.save({'optimizer': optimizer.state_dict(), 'model': model.state_dict(), 
                    'epoch': epoch + 1}, p['pretext_checkpoint'])

    # Save final model
    torch.save(model.state_dict(), p['pretext_model'])

    # Mine the topk nearest neighbors at the very end (Train) 
    # These will be served as input to the SCAN loss.
    print(colored('Fill memory bank for mining the nearest neighbors (train) ...', 'blue'))
    # fill the all features which is the output of each training data. [N, features_dim]
    fill_memory_bank(base_dataloader, model, memory_bank_base)
    topk = 20
    print('Mine the nearest neighbors (Top-%d)' %(topk)) 
    # select topk nearest-neighbor of each training data
    indices, acc = memory_bank_base.mine_nearest_neighbors(topk)
    print('Accuracy of top-%d nearest neighbors on train set is %.2f' %(topk, 100*acc))
    np.save(p['topk_neighbors_train_path'], indices)

    if args.save_features is True:
        save_feature(memory_bank_base, base_dataset.train_txt, "SCAN_GODTrain211111_2.0_od_tstld_kor_128dim_211110.csv")
    #if args.plot_nn_dist is True:
        #plot_nn_distance(indices, memory_bank_base, "./hist/")
   
    # Mine the topk nearest neighbors at the very end (Val)
    # These will be used for validation.
    print(colored('Fill memory bank for mining the nearest neighbors (val) ...', 'blue'))
    fill_memory_bank(val_dataloader, model, memory_bank_val)
    topk = 10
    print('Mine the nearest neighbors (Top-%d)' %(topk)) 
    indices, acc = memory_bank_val.mine_nearest_neighbors(topk)
    print('Accuracy of top-%d nearest neighbors on val set is %.2f' %(topk, 100*acc))
    np.save(p['topk_neighbors_val_path'], indices)   

 
if __name__ == '__main__':
    main()
