"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import torch
import numpy as np
import collections
int_classes = int
string_classes = str

def scan_collate(batch):
    batch = [data for data in batch if data is not None]
    #skip invalid frames
    if len(batch) == 0:
        return
    
    #batch['anchor'], batch['neighbor'], batch['possible_neighbors'], batch['target'], batch['img_name']
    anchor, neighbor, possible_neighbor, target, img_name = list(zip(*batch))
    anchors = torch.stack(anchor, 0)
    neighbors = torch.stack(neighbor, 0)
    possible_neighbors = torch.stack(possible_neighbor, 0)
    targets = torch.FloatTensor(target)
    
    return anchors, neighbors, possible_neighbors, targets, img_name

""" Custom collate function """
def collate_custom(batch):
    batch = [data for data in batch if data is not None]
    #skip invalid frames
    if len(batch) == 0:
        return
    #batch['anchor'], batch['neighbor'], batch['possible_neighbors'], batch['target'], batch['img_name']
    anchor, target, img_name = list(zip(*batch))
    #_collated_data = [[b['image'], b['target'], b['meta']['img_name']] for b in batch]
    #print(_collated_data)
    anchors = torch.stack(anchor, 0)
    targets = torch.FloatTensor(target)

    
    return anchors, targets, img_name
