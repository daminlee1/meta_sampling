"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import torch
import numpy as np
from utils.utils import AverageMeter, ProgressMeter
import time

def simclr_train(train_loader, model, criterion, optimizer, epoch):
    """ 
    Train according to the scheme from SimCLR
    https://arxiv.org/abs/2002.05709
    """
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(len(train_loader),
        [losses],
        prefix="Epoch: [{}]".format(epoch))

    model.train()

    for i, batch in enumerate(train_loader):
        images = batch['image']
        images_augmented = batch['image_augmented']
        b, c, h, w = images.size()
        input_ = torch.cat([images.unsqueeze(1), images_augmented.unsqueeze(1)], dim=1)
        input_ = input_.view(-1, c, h, w) 
        input_ = input_.cuda(non_blocking=True)
        targets = batch['target'].cuda(non_blocking=True)

        output = model(input_).view(b, 2, -1)
        loss = criterion(output)
        losses.update(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 25 == 0:
            progress.display(i)
    return losses.avg

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def scan_train(train_loader, model, criterion, optimizer, epoch, update_cluster_head_only=False, torch_writer=None, iter_count=None):
    """ 
    Train w/ SCAN-Loss
    """
    total_losses = AverageMeter('Total Loss', ':.4e')
    consistency_losses = AverageMeter('Consistency Loss', ':.4e')
    entropy_losses = AverageMeter('Entropy', ':.4e')
    progress = ProgressMeter(len(train_loader),
        [total_losses, consistency_losses, entropy_losses],
        prefix="Epoch: [{}]".format(epoch))

    if update_cluster_head_only:
        model.eval() # No need to update BN
    else:
        model.train() # Update BN

    init_time = time.time()
    for i, batch in enumerate(train_loader):
        # Forward pass
        anchors, neighbors, _, _, _ = batch
        #anchors = batch['anchor'].cuda()
        #neighbors = batch['neighbor'].cuda()

        if update_cluster_head_only: # Only calculate gradient for backprop of linear layer
            with torch.no_grad():
                anchors_features = model(anchors, forward_pass='backbone')
                neighbors_features = model(neighbors, forward_pass='backbone')
            anchors_output = model(anchors_features, forward_pass='head')
            neighbors_output = model(neighbors_features, forward_pass='head')

        else: # Calculate gradient for backprop of complete network
            anchors_output = model(anchors)
            neighbors_output = model(neighbors)     

        # Loss for every head
        total_loss, consistency_loss, entropy_loss = [], [], []
        for anchors_output_subhead, neighbors_output_subhead in zip(anchors_output, neighbors_output):
            total_loss_, consistency_loss_, entropy_loss_ = criterion(anchors_output_subhead,
                                                                         neighbors_output_subhead)
            total_loss.append(total_loss_)
            consistency_loss.append(consistency_loss_)
            entropy_loss.append(entropy_loss_)

        # Register the mean loss and backprop the total loss to cover all subheads
        const_loss = np.mean([v.item() for v in consistency_loss])
        entrp_loss = np.mean([v.item() for v in entropy_loss])
        t_loss = np.mean([v.item() for v in total_loss])
        total_losses.update(t_loss)
        consistency_losses.update(const_loss)
        entropy_losses.update(entrp_loss)

        total_loss_sum = torch.sum(torch.stack(total_loss, dim=0))

        optimizer.zero_grad()
        total_loss_sum.backward()
        optimizer.step()

        heads_cl = ['heads'+str(i)+'_cl' for i in range(len(consistency_loss))]
        heads_el = ['heads'+str(i)+'_el' for i in range(len(entropy_loss))]
        heads_tl = ['heads'+str(i)+'_tl' for i in range(len(total_loss))]
        
        torch_writer.add_scalar("lr", get_lr(optimizer), iter_count)
        torch_writer.add_scalar("latency", time.time()-init_time, iter_count)
        torch_writer.add_scalar("total_loss", t_loss, iter_count)
        torch_writer.add_scalar("consistency_loss", const_loss, iter_count)
        torch_writer.add_scalar("entropy_loss", entrp_loss, iter_count)
        for tn, cn, en, t, c, e in zip(heads_tl, heads_cl, heads_el, total_loss, consistency_loss, entropy_loss):
            torch_writer.add_scalar(tn, t.item(), iter_count)
            torch_writer.add_scalar(cn, c.item(), iter_count)
            torch_writer.add_scalar(en, e.item(), iter_count)
        #select lowest loss head
        lowest_loss_head_idx = np.argmin([tl.cpu().detach().numpy() for tl in total_loss])
        iter_count += 1
        init_time = time.time()
        if i % 25 == 0:
            progress.display(i)
            
        if i % 1000 == 0:
            torch.save({'optimizer': optimizer.state_dict(), 'model': model.state_dict(), 
                        'epoch': epoch, 'iter': iter_count}, "./output/checkpoint.pth.tar")
    return iter_count, lowest_loss_head_idx


def selflabel_train(train_loader, model, criterion, optimizer, epoch, ema=None):
    """ 
    Self-labeling based on confident samples
    """
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(len(train_loader), [losses],
                                prefix="Epoch: [{}]".format(epoch))
    model.train()

    for i, batch in enumerate(train_loader):
        images = batch['image'].cuda(non_blocking=True)
        images_augmented = batch['image_augmented'].cuda(non_blocking=True)

        with torch.no_grad(): 
            output = model(images)[0]
        output_augmented = model(images_augmented)[0]

        loss = criterion(output, output_augmented)
        losses.update(loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if ema is not None: # Apply EMA to update the weights of the network
            ema.update_params(model)
            ema.apply_shadow(model)
        
        if i % 25 == 0:
            progress.display(i)
