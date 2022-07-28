"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import argparse
import os, sys
import torch
import torchvision.transforms as transforms
import torchsummary as summary
import numpy as np
import pynvml
import csv
from PIL import Image
import matplotlib.pyplot as plt

from utils.config import create_config
from utils.common_config import get_model, get_train_dataset,\
                                get_val_dataset, get_val_dataloader, get_val_transformations
from utils.memory import MemoryBank
from utils.utils import fill_memory_bank
from termcolor import colored

# Parser
parser = argparse.ArgumentParser(description='MoCo')
parser.add_argument('--config_env',
                    help='Config file for the environment')
parser.add_argument('--config_exp',
                    help='Config file for the experiment')
parser.add_argument("--gpus", type=int, nargs='+', default=[0], help="List of device ids.")
args = parser.parse_args()

def get_memory_free_MiB(gpu_index):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(int(gpu_index))
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return mem_info.free // 1024 ** 2

def get_memory_total_MiB(gpu_index):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(int(gpu_index))
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return mem_info.total // 1024 ** 2

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
    using_gpus = [int(g) for g in args.gpus]
    
    for i in using_gpus:
        print("GPU total memory : {} free memory : {}".format(get_memory_total_MiB(i), get_memory_free_MiB(i)))
        if get_memory_free_MiB(i) / get_memory_total_MiB(i) < 0.5:
            print("Avaliable memory is {}%, GPU is already used now, Exit process".format(get_memory_free_MiB(i) / get_memory_total_MiB(i)))
            sys.exit(1)
    if len(using_gpus) == 1:
        torch.cuda.set_device(using_gpus[0])
        model = torch.nn.DataParallel(model, device_ids=using_gpus)
        model = model.cuda()
    elif len(using_gpus) == 0:
        if torch.cuda.is_available():
            model = model.cuda()
        else:
            print("Disable to use GPU. Exit process")
            sys.exit(1)
    elif len(using_gpus) > 1:
        model = torch.nn.DataParallel(model, device_ids=using_gpus)
        model = model.cuda()
        model.to(f'cuda:{model.device_ids[0]}')
    
    # CUDNN
    print(colored('Set CuDNN benchmark', 'blue')) 
    torch.backends.cudnn.benchmark = True
    
    
    # Dataset
    print(colored('Retrieve dataset', 'blue'))
    moco_transforms = get_val_transformations(p)
    train_dataset = get_train_dataset(p, moco_transforms) 
    val_dataset = get_val_dataset(p, moco_transforms)
    train_dataloader = get_val_dataloader(p, train_dataset)
    val_dataloader = get_val_dataloader(p, val_dataset)
    print('Dataset contains {}/{} train/val samples'.format(len(train_dataset), len(val_dataset)))
    
    # # # Show transformed image
    # for i, batch in enumerate(val_dataloader):
    #     images, targets, img_name = batch
    #     image_np = images.detach().cpu().numpy() * 255
    #     _img_data = np.array(np.transpose(image_np[0], (1,2,0)), dtype=np.uint8)
    #     img_data = Image.fromarray(_img_data)
    #     img_data.save("moco_test/"+str(i)+".jpg")
    # sys.exit(1)

    # Memory Bank
    print(colored('Build MemoryBank', 'blue'))
    print(p['model_kwargs']['features_dim'])
    memory_bank_train = MemoryBank(len(train_dataset),
                                   2048,
                                   p['num_classes'],
                                   p['temperature']) 
    memory_bank_train.cuda()
    memory_bank_val = MemoryBank(len(val_dataset),
                                 2048,
                                 p['num_classes'],
                                 p['temperature']) 
    memory_bank_val.cuda()

    
    # Load the official MoCoV2 checkpoint
    print(colored('Loading moco v2 checkpoint', 'blue'))
    #os.system('wget -L https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_800ep/moco_v2_800ep_pretrain.pth.tar')
    moco_state = torch.load('moco_v2_800ep_pretrain.pth.tar', map_location='cpu')

    
    # Transfer moco weights
    print(colored('Transfer MoCo weights to model', 'blue'))
    new_state_dict = {}
    state_dict = moco_state['state_dict']
    for k in list(state_dict.keys()):
        # Copy backbone weights
        if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
            new_k = 'module.backbone.' + k[len('module.encoder_q.'):]
            new_state_dict[new_k] = state_dict[k]
        
        # Copy mlp weights
        elif k.startswith('module.encoder_q.fc'):
            new_k = 'module.contrastive_head.' + k[len('module.encoder_q.fc.'):] 
            new_state_dict[new_k] = state_dict[k] 

        else:
            raise ValueError('Unexpected key {}'.format(k)) 

    model.load_state_dict(new_state_dict)
    os.system('rm -rf moco_v2_800ep_pretrain.pth.tar')
    print("val_dataset.file_txt : ", val_dataset.file_txt)
 
    # Save final model
    print(colored('Save pretext model', 'blue'))
    torch.save(model.module.state_dict(), p['pretext_model'])
    model.module.contrastive_head = torch.nn.Identity() # In this case, we mine the neighbors before the MLP. 

    # #test_img = torch.randn([3,416,416], dtype=torch.Float32)
    # summary.summary(model, input_size=(3, 416, 416), device='cuda')

    # Mine the topk nearest neighbors (Train)
    # These will be used for training with the SCAN-Loss.
    topk = 30
    print(colored('Mine the nearest neighbors (Train)(Top-%d)' %(topk), 'blue'))
    transforms = get_val_transformations(p)
    val_dataset = get_val_dataset(p, transforms) 
    fill_memory_bank(val_dataloader, model, memory_bank_val)
    indices, acc = memory_bank_val.mine_nearest_neighbors(topk)
    print('Accuracy of top-%d nearest neighbors on train set is %.2f' %(topk, 100*acc))
    np.save(p['topk_neighbors_train_path'], indices)
   
    save_feature(memory_bank_val, val_dataset.file_txt, "SCANmocoDataFeatures.csv") 
     
    # Mine the topk nearest neighbors (Validation)
    # These will be used for validation.
    # topk = 5
    # print(colored('Mine the nearest neighbors (Val)(Top-%d)' %(topk), 'blue'))
    # fill_memory_bank(val_dataloader, model, memory_bank_val)
    # print('Mine the neighbors')
    # indices, acc = memory_bank_val.mine_nearest_neighbors(topk)
    # print('Accuracy of top-%d nearest neighbors on val set is %.2f' %(topk, 100*acc))
    np.save(p['topk_neighbors_val_path'], indices)
    
    #save_feature(memory_bank_val, val_dataset.train_txt, "SCANmoco_SVKPI3000km_128dim_220217.csv")


if __name__ == '__main__':
    main()
