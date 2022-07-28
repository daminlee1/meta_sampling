import argparse
import os,sys
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
parser.add_argument('--gpus',
                    help="the index of GPU you use", type=int, default=0)
parser.add_argument("--plot_nn_dist",
                    help="plot the distance of nearest neighbors", type=bool, default=False)
parser.add_argument('--pretext_model', dest='pretext_model',
                    help="pretext_model", type=str, default=None)
parser.add_argument('--save_feature_name', dest='save_feature_name',
                    help="the name of feature_file.csv", type=str, default=None)
args = parser.parse_args()

def save_feature(memory_bank, dataloader, csv_name):
    base_features = memory_bank.features.cpu().numpy()
    print("Base feature shapes : {},{}".format(base_features.shape[0],base_features.shape[1]))
    print("Dataloader len : {}".format(len(dataloader)))
    print("Saving CSV file : ", csv_name)
    print("Saving SCAN representation model features of DB")
    file_list = []
    for i, batch in enumerate(dataloader):
        imgs_name = batch['meta']['im_name']
        for img in imgs_name:
            file_list.append(img)
        if i % 100 == 0:
            print('Save Feature[%d/%d]' %(i, len(dataloader)))

    with open(csv_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        print("num of base_features : ", base_features.shape[0])
        for i, feat in enumerate(base_features):
            csv_data = [file_list[i]]+feat.tolist()
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
    
    print("GPUs : ", args.gpus)
    torch.cuda.set_device(args.gpus)
    # Retrieve config file
    p = create_config(args.config_env, args.config_exp)
    print(colored(p, 'red'))

    model = get_model(p)
    print('Model is {}'.format(model.__class__.__name__))
    print('Model parameters: {:.2f}M'.format(sum(p.numel() for p in model.parameters()) / 1e6))
    print(model)
    #model = torch.nn.DataParallel(model)
    model = model.cuda()

    # CUDNN
    print(colored('Set CuDNN benchmark', 'blue')) 
    torch.backends.cudnn.benchmark = True
    
    # Dataset
    print(colored('Retrieve dataset', 'blue'))
    train_transforms = get_train_transformations(p)
    print('Train transforms:', train_transforms)
    
    # # Memory Bank
    print(colored('Build MemoryBank', 'blue'))
    base_dataset = get_train_dataset(p, train_transforms, split='train')
    print('Dataset contains {}'.format(len(base_dataset)))
    base_dataloader = get_val_dataloader(p, base_dataset)
    memory_bank_base = MemoryBank(len(base_dataset),
                                p['model_kwargs']['features_dim'],
                                p['num_classes'], p['criterion_kwargs']['temperature'])
    memory_bank_base.cuda(args.gpus)

    # Checkpoint
    if os.path.exists(args.pretext_model):
        print(colored('Load the best model {}'.format(args.pretext_model), 'blue'))
        best_model = torch.load(args.pretext_model, map_location='cpu')
        #model.load_state_dict(checkpoint['model'])
        for key, value in best_model.copy().items():
            new_key = key.replace('module.','')
            best_model[new_key] = best_model.pop(key)
        model.load_state_dict(best_model, strict=False)
        model.cuda()
    else:
        print("Fail to find pretext model checkpoint")
        sys.exit(1)

    # Mine the topk nearest neighbors at the very end (Train) 
    # These will be served as input to the SCAN loss.
    print(colored('Fill memory bank for mining the nearest neighbors (train) ...', 'blue'))
    # fill the all features which is the output of each training data. [N, features_dim]
    fill_memory_bank(base_dataloader, model, memory_bank_base)

    #save the features of each data
    save_feature(memory_bank_base, base_dataloader, args.save_feature_name+".csv")
    # if args.plot_nn_dist is True:
    #    plot_nn_distance(indices, memory_bank_base, "./hist/")
    memory_bank_base.reset()
    memory_bank_base.release()

    print("Finish")
 
if __name__ == '__main__':
    main()
