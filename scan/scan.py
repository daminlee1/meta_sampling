"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import argparse
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import torch
import pynvml
import csv
import cv2
from collections import OrderedDict

from termcolor import colored
from utils.config import create_config
from utils.common_config import get_train_transformations, get_val_transformations,\
                                get_train_dataset, get_train_dataloader,\
                                get_val_dataset, get_val_dataloader,\
                                get_optimizer, get_model, get_criterion,\
                                adjust_learning_rate
from utils.evaluate_utils import get_predictions, scan_evaluate, hungarian_evaluate, get_lowesthead_predictions
from utils.train_utils import scan_train
from tensorboardX import SummaryWriter

FLAGS = argparse.ArgumentParser(description='SCAN Loss')
FLAGS.add_argument('--config_env', help='Location of path config file')
FLAGS.add_argument('--config_exp', help='Location of experiments config file')
FLAGS.add_argument('--gpus', type=int, nargs='+', default=[0], help="List of GPU device id")

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

def main():
    args = FLAGS.parse_args()
    p = create_config(args.config_env, args.config_exp)
    print(colored(p, 'red'))

    # Data
    print(colored('Get dataset and dataloaders', 'blue'))
    train_transformations = get_train_transformations(p)
    val_transformations = get_val_transformations(p)
    train_dataset = get_train_dataset(p, train_transformations, 
                                        split='train', to_neighbors_dataset = True)
    val_dataset = get_val_dataset(p, val_transformations, to_neighbors_dataset = True)
    train_dataloader = get_train_dataloader(p, train_dataset, _collate_fn = 'cluster')
    val_dataloader = get_val_dataloader(p, val_dataset, _collate_fn = 'cluster')
    print('Train transforms:', train_transformations)
    #print('Validation transforms:', val_transformations)
    print('Train samples %d ' %(len(train_dataset)))
    
    # Model
    print(colored('Get model', 'blue'))
    model = get_model(p, p['pretext_model'])
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
        print("Error, args.gpus is None")
        sys.exit(1)
    elif len(using_gpus) > 1:
        model = torch.nn.DataParallel(model, device_ids=using_gpus)
        model = model.cuda()
        model.to(f'cuda:{model.device_ids[0]}')
    
    # CUDNN
    print(colored('Set CuDNN benchmark', 'blue')) 
    torch.backends.cudnn.benchmark = True

    # Optimizer
    print(colored('Get optimizer', 'blue'))
    optimizer = get_optimizer(p, model, p['update_cluster_head_only'])
    print(optimizer)
    
    # Warning
    if p['update_cluster_head_only']:
        print(colored('WARNING: SCAN will only update the cluster head', 'red'))

    # Loss function
    print(colored('Get loss', 'blue'))
    criterion = get_criterion(p) 
    criterion.cuda()
    print(criterion)
    iter_count = 0
    start_epoch = 0
    best_loss_head = None
    best_loss = 1e4
    print(colored('No checkpoint file at {}'.format(p['scan_checkpoint']), 'blue'))
    # # Checkpoint
    # if os.path.exists(p['scan_checkpoint']):
    #     print(colored('Restart from checkpoint {}'.format(p['scan_checkpoint']), 'blue'))
    #     checkpoint = torch.load("./output/checkpoint.pth.tar", map_location='cpu')
    #     model.load_state_dict(checkpoint['model'])
    #     optimizer.load_state_dict(checkpoint['optimizer'])        
    #     start_epoch = checkpoint['epoch']
    #     iter_count = checkpoint['iter']
    #     #best_loss = checkpoint['best_loss']
    #     #best_loss_head = checkpoint['best_loss_head']
        
    #Setting the torch log directory to use tensorboard
    torch_writer = SummaryWriter("./output")
 
    # Main loop
    # print(colored('Starting main loop', 'blue'))
    # for epoch in range(start_epoch, p['epochs']):
    #     print(colored('Epoch %d/%d' %(epoch+1, p['epochs']), 'yellow'))
    #     print(colored('-'*15, 'yellow'))

    #     # Adjust lr
    #     lr = adjust_learning_rate(p, optimizer, epoch)
    #     print('Adjusted learning rate to {:.5f}'.format(lr))

    #     # Train
    #     print('Train ...')
    #     iter_count, lowest_loss_head = scan_train(train_dataloader, model, criterion, optimizer, epoch, p['update_cluster_head_only'], torch_writer, iter_count)

    #     # # Evaluate 
    #     # print('Make prediction on validation set ...')
    #     # predictions = get_predictions(p, train_dataloader, model)

    #     # print('Evaluate based on SCAN loss ...')
    #     # scan_stats = scan_evaluate(predictions)
    #     # print(scan_stats)
    #     # lowest_loss_head = scan_stats['lowest_loss_head']
    #     # lowest_loss = scan_stats['lowest_loss']
       
    #     # if lowest_loss < best_loss:
    #     #     print('New lowest loss on validation set: %.4f -> %.4f' %(best_loss, lowest_loss))
    #     #     print('Lowest loss head is %d' %(lowest_loss_head))
    #     #     best_loss = lowest_loss
    #     #     best_loss_head = lowest_loss_head
    #     #     torch.save({'model': model.module.state_dict(), 'head': best_loss_head}, p['scan_model'])

    #     # else:
    #     #     print('No new lowest loss on validation set: %.4f -> %.4f' %(best_loss, lowest_loss))
    #     #     print('Lowest loss head is %d' %(best_loss_head))

    #     #print('Evaluate with hungarian matching algorithm ...')
    #     #clustering_stats = hungarian_evaluate(lowest_loss_head, predictions, compute_confusion_matrix=False)
    #     #print(clustering_stats)     
    #     torch.save({'model': model.module.state_dict(), 'head': lowest_loss_head}, p['scan_model'])

    #     # Checkpoint
    #     print('Checkpoint ...')
    #     torch.save({'optimizer': optimizer.state_dict(), 'model': model.state_dict(), 
    #                 'epoch': epoch + 1, 'best_loss': best_loss, 'best_loss_head': best_loss_head},
    #                  p['scan_checkpoint'])
    
    # Evaluate and save the final model
    print(colored('Evaluate best model based on SCAN metric at the end', 'blue'))
    print(colored('Load checkpoint file at {}'.format(p['scan_model']), 'blue'))
    model_checkpoint = torch.load(p['scan_model'], map_location='cpu')
    
    # if len(using_gpus) > 1:
    #     new_state_dict = OrderedDict()
    #     for n, v in model_checkpoint['model'].items():
    #         name = "module."+ n
    #         new_state_dict[name] = v
    # else:
    #     new_state_dict = model_checkpoint['model'].copy()
    new_state_dict = OrderedDict()
    for n, v in model_checkpoint['model'].items():
        #name = "module."+ n
        name = n
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    #model.load_state_dict(model_checkpoint['model']) #.module
    predictions = get_predictions(p, val_dataloader, model, return_names=True)
    scan_stats = scan_evaluate(predictions)
    print("lowest_loss_head : ", scan_stats['lowest_loss_head'], " loss : ", scan_stats['lowest_loss'])
    lowest_loss_head = scan_stats['lowest_loss_head']
    lowest_loss = scan_stats['lowest_loss']
    
    header = predictions[lowest_loss_head]
    preds = header['predictions']
    probs = header['probabilities']
    img_names = header['img_names']
    
    print(preds.shape, len(img_names))
    
    for i in range(27):
        mask = preds==i
        print("Save {} group clustering".format(i))
        each_preds = preds[mask].numpy().tolist()
        each_probs = probs[mask].numpy().tolist()
        each_names = []
        for mi, mv in enumerate(mask.tolist()):
            if mv:
                each_names.append(img_names[mi])
        group_name = "./clusters/cluster"+str(i)
        if not os.path.exists("./clusters"):
            os.mkdir("./clusters")
        if not os.path.exists(group_name):
            os.mkdir(group_name)
        with open(group_name+"/data_nothreshold.csv", 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            #print("preds : {} probs : {} names :{} ".format(len(each_preds), len(each_probs), len(each_names)))
            for j, (c, p, n) in enumerate(zip(each_preds, each_probs, each_names)):
                # if p[i] < 0.5:
                #     continue
                csv_data = [n,c,p[i]]
                writer.writerows([csv_data])
                # if j % 20 != 0:
                #     continue
                # img = cv2.imread("/damin/data/GODTrain211111/JPEGImages/"+n)
                # cv2.imwrite(group_name+"/"+n, img)
    
if __name__ == "__main__":
    main()
