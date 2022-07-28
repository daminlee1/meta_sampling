import os,sys
import csv
import faiss
import argparse
import numpy as np
from xml.etree.ElementTree import parse
import cv2 as cv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

class_name = ["bg", "pedestrian", "rider", "car", "truck", "bus", "tsc", "tst", "tsr", "tl_car", "tl_ped"]
class_string = ["pedestrian", "rider_bicycle", "rider_bike", "bicycle", "bike",    "3-wheels_rider", "3-wheels", "sedan", "van", "pickup_truck", "truck", "mixer_truck", "excavator", "forklift", "ladder_truck", "truck_etc", "vehicle_etc", "box_truck", "trailer", "bus", "ehicle_special", "sitting_person", "ignored", "false_positive", "animal", "bird", "animal_ignored", "ts_circle", "ts_circle_speed", "ts_triangle", "ts_inverted_triangle", "ts_rectangle", "ts_rectangle_speed", "ts_diamonds", "ts_supplementary", "tl_car", "tl_ped", " tl_special", "tl_light_only", "ts_ignored", "tl_ignored", "tstl_ignore", "ts_sup_ignored", "ts_sup_letter", "ts_sup_drawing", "ts_sup_arrow", "ts_sup_zone", "ts_main_zone", "ts_rectangle_arrow", "tl_rear", "ts_rear", "obstacle_bollard_barricade", "obstacle_bollard_cylinder", "obstacle_bollard_marker", "obstacle_cone", "obstacle_drum", "obstacle_cylinder", "obstacle_bollard_special", "obstacle_bollard_stone", "obstacle_bollard_U_shaped", "parking_cylinder", "parking_sign", "parking_stopper_separated", "parking_stopper_bar", "parking_stopper_marble", "parking_special", "parking_lock", "blocking_bar", "blocking_special", "blocking_ignored", "parking_ignored", "obstacle_ignored", "stopline_normal", "stopline_special", "stopline_ignored", "arrow_normal", "arrow_special", "arrow_ignored", "crosswalk_normal", "crosswalk_special", "crosswalk_ignored", "speedbump_normal", "speedbump_special", "speedbump_ignored", "number_speed", "number_parkingzone", "number_special", "number_ignored", "text_normal", "text_parkingzone", "text_special", "text_ignored", "roadmark_triangle", "roadmark_diamond", "roadmark_bicycle", "roadmark_handicapped", "roadmark_pregnant", "roadmark_special", "roadmark_ignored", "rider_bicycle_2", "rider_bike_2", "rider_bicycle_human_body", "wheel_chair", "vehicle_special", "ts_chevron"]
class_index = [         1,             2,          2, -100000, -100000,              2,  -100000,     3,   3,            3,     4,           4,         4,        4,            4,   -100000,     -100000,         4,       4,   5,        -100000,              1, -100000,              0,      0,    0,              0,         6,               6,           7,                    7,            8,                  8,           8,          -100000,      9,     10,     -100000,       -100000,    -100000,    -100000,     -100000,        -100000,       -100000,        -100000,      -100000,     -100000,      -100000,            -100000, -100000, -100000,                    -100000,                   -100000,                 -100000,        -100000,      -100000,          -100000,                   -100000,                -100000,                   -100000,          -100000,      -100000,                   -100000,             -100000,                -100000,         -100000,      -100000,      -100000,          -100000,          -100000,         -100000,          -100000,         -100000,         -100000,           -100000,      -100000,       -100000,       -100000,          -100000,           -100000,           -100000,          -100000,           -100000,           -100000,      -100000,            -100000,        -100000,        -100000,     -100000,          -100000,      -100000,      -100000,           -100000,          -100000,          -100000,              -100000,           -100000,          -100000,          -100000,         -100000,      -100000,                  -100000,    -100000,          -100000,    -100000]


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_csv', dest='target_csv', help='target csv file to read',
                        default=None, type=str)
    parser.add_argument('--near_csv', dest='near_csv', help='near csv file to read',
                        default=None, type=str)
    parser.add_argument('--save_dir', dest='save_dir', help='save directory',
                        default=None, type=str)
    parser.add_argument('--target_db', dest='target_db', help='target_db',
                        default=None, type=str)
    parser.add_argument('--near_db', dest='near_db', help='near_db',
                        default=None, type=str)
    parser.add_argument('--nn_factor', dest='nn_factor', help='nearest neighbors top-k factors', nargs='+',
                        default=[20], type=int)
    parser.add_argument('--nn_npy', dest='nn_npy', help='nearest neighbor npy file',
                        default=None, type=str)
    parser.add_argument('--dist_thr', dest='dist_thr', help='threshold of cosine distance to use filtering the sequential frames',
                        default=0.98, type=float)
    parser.add_argument('--feature_dim', dest='feature_dim', help='feature_dim',
                        default=2048, type=int)
    args = parser.parse_args()
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return args

def is_float(element) -> bool:
    try:
        float(element)
        return True
    except ValueError:
        return False

def read_csv_v2(csv_file, dim, csvtype = "name"):
    data = []
    if csvtype == "name":
        with open(csv_file, 'r') as f:
            reader = csv.reader(f, delimiter="\t")
            for line in tqdm(reader, desc="Read name from CSV file", mininterval = 0.1):
                line = line[0].split(",")
                idx = 0
                for l in line:
                    if is_float(l):
                       break
                    idx += 1
                if idx != 0:
                    for i in range(idx):
                        name += line[i]+','
                    name = name[:-1]
                else:
                    name = line[0]
                data.append(str(name).replace("\n",""))
    elif csvtype == "code":
        with open(csv_file, 'r') as f:
            reader = csv.reader(f, delimiter="\t")
            for line in tqdm(reader, desc="Read feature from CSV file", mininterval = 0.1):
                line = line[0].split(",")
                idx = 0
                for l in line:
                    if is_float(l):
                       break
                    idx += 1
                code = line[idx + 1:]
                # print('code:',code)
                line_float = [float(i) for i in code]
                data.append(line_float)
    else :
        print("unknown csvtype")
        sys.exit(1)    
    return data

def read_csv(csv_file, dim, csvtype = "name"):
    data = []
    if csvtype == "name":
        with open(csv_file, 'r') as f:
            reader = csv.reader(f, delimiter="\t")
            for line in tqdm(reader, desc="Read name from CSV file", mininterval = 0.1):
                line = line[0].split(",")
                data.append(str(line[0]).replace("\n",""))
    elif csvtype == "code":
        with open(csv_file, 'r') as f:
            reader = csv.reader(f, delimiter="\t")
            for line in tqdm(reader, desc="Read feature from CSV file", mininterval = 0.1):
                line = line[0].split(',')
                line_float = [float(i) for i in line[1:dim+1]]
                data.append(line_float)
    else :
        print("unknown csvtype")
        sys.exit(1)    
    return data

class FeatureBank(object):
    def __init__(self, n, dim):
        self.n = n
        self.dim = dim
        self.ptr = 0
        self.features = np.zeros((self.n, self.dim), dtype=np.float32)
        self.device = 'cpu'

    def update(self, features):
        b = len(features)        
        assert(b + self.ptr <= self.n)

        self.features[self.ptr:self.ptr+b] = np.array(features)

        self.ptr += b

    def reset(self):
        self.ptr = 0

    def cuda(self, gpu=0):
        self.to('cuda:' + str(gpu))

    def cpu(self):
        self.to('cpu')

    def to(self, device):
        self.features = self.features.to(device)
        self.device = device

def nearest_neighbors_of_each_feature(topk, near_feature_bank, target_feature_bank):
    feature_target = target_feature_bank.features
    feature_near = near_feature_bank.features

    n, dim = feature_near.shape[0]+1, feature_near.shape[1]
    index = faiss.IndexFlatIP(dim)
    index = faiss.index_cpu_to_all_gpus(index)
    index.add(feature_near)

    target_n = feature_target.shape[0]
    indices = []
    for i in range(target_n):
        each_target = np.array([feature_target[i]])
        dist, idx = index.search(each_target, topk+1)
        new_idx = [i]
        for j in idx[0]:
            new_idx.append(j)
        indices.append(new_idx)
    indices = np.array(indices)
    return indices

def nearest_neighbors(topk, target_feature_bank):
    feature_target = target_feature_bank.features
    n, dim = feature_target.shape[0]+1, feature_target.shape[1]
    index = faiss.IndexFlatIP(dim)
    index = faiss.index_cpu_to_all_gpus(index)
    index.add(feature_target)
    distances, indices = index.search(feature_target, topk+1)
    return distances, indices

def analyze_cls(feature_cls, idx):
    ped_counts = 0
    rider_counts = 0
    for c in feature_cls[idx]:
        if "pedestrian" == c:
            ped_counts += 1
        elif "rider" == c:
            rider_counts += 1
    return ped_counts, rider_counts

def get_name_from_xml(anno):
    root = anno.getroot()
    obj = root.findall("object")
    name = [x.findtext("name") for x in obj]
    name_cls = [class_name[class_index[class_string.index(k)]] for k in name if k in class_string if class_index[class_string.index(k)] > 0 ]
    return name_cls

def get_name_xml(anno_dir, name_list):
    names = []
    for name in name_list:
        name = name.replace(".png", ".xml")
        name = name.replace(".jpg", ".xml")
        anno = parse(anno_dir + name)
        name = get_name_from_xml(anno)
        names.append(name)
    return names

def Average(lst):
    return sum(lst) / len(lst)

def plot_distance(distances):
    distance_mean = [d.mean() for d in distances]
    distance_std = [d.std() for d in distances]
    
    # for dist, idx in zip(distances, indices):
    #     print("dist : " , dist)
    #     target_feature = np.array(target_data["code"][idx[0]])
    #     near_feature = np.array([target_data["code"][id] for id in idx])
    #     #euclidean distance
    #     # my_dist = []
    #     # for n in near_feature:
    #     #     sum = 0
    #     #     for j in range(128):
    #     #         sum += pow(target_feature[j] - n[j],2)
    #     #     sum = pow(sum,0.5) / 128
    #     #     my_dist.append(sum)
    #     # print("my_dist : " , my_dist)
    #     #cosine distance
    #     # cos_dist = []
    #     # for near in near_feature:
    #     #     l2_f = 0
    #     #     l2_n = 0
    #     #     inner_prod = 0
    #     #     for f, n in zip(target_feature, near):
    #     #         l2_f += pow(f,2)
    #     #         l2_n += pow(n,2)
    #     #         inner_prod += f*n
    #     #     l2_f = pow(l2_f,0.5)
    #     #     l2_n = pow(l2_n,0.5)
    #     #     cos_value = inner_prod/(l2_f*l2_n)
    #     #     cos_dist.append(cos_value)
    #     # print("cos dist : ", cos_dist)

    if not os.path.exists(args.save_dir+"/sequential"):
        os.mkdir(args.save_dir+"/sequential")
    for i, (m, s) in enumerate(zip(distance_mean, distance_std)):
        if m > 0.98:
            print("{}th image : mean {} std {}".format(i, m, s))
            sns.displot(x=distances[i])
            plt.title("m_"+str(m)+"_s_"+str(s))
            plt.savefig(args.save_dir+"/sequential/"+"nn_"+str(i)+".jpg")
            plt.clf()

    sns.histplot(x=distance_mean)
    plt.savefig(args.save_dir+"/histo_distance_mean.jpg")
    plt.clf()
    sns.histplot(x=distance_std)
    plt.savefig(args.save_dir+"/histo_distance_std.jpg")

# find the sequential frames in DB and remove it.
def filtering_sequential_data(distances, indices, thr = 0.98):
    cand_list = np.arange(len(indices))
    for i, c in enumerate(cand_list):
        if c == -1:
            continue
        #delete target index in distances
        near_dist = np.delete(distances[c], 0)
        seq_dist = np.reshape(np.argwhere(near_dist>=thr),(-1))
        seq_indices = [indices[c,idx] for idx in seq_dist]
        #exclude the indices of sequential frame in cand_list 
        for s in seq_indices:
            cand_list[s] = -1

    seq_indices_cand = np.reshape(np.argwhere(cand_list==-1),(-1))
    print("seq_indices_cand : ", seq_indices_cand.shape)
    unique_cand_list = np.delete(cand_list, seq_indices_cand)
    print("unique_cand_list : ", unique_cand_list.shape)
    return unique_cand_list

import random
from numpy import dot
from numpy.linalg import norm

def calculate_cosine_distance(target : list, near : list):
    num_target = len(target)
    num_near = len(near)
    candidates = list()
    sub_near = num_near // 10
    #scores = np.zeros((num_target, sub_near), dtype=np.int8)
    t_idx = 0
    for t in tqdm(target, desc = "calculate_cosine_distance", mininterval = 0.1):
        t = np.array(t)
        rand_indices = [random.randint(0, num_near - 1) for _ in range(sub_near)]
        for j, n_idx in enumerate(rand_indices):
            n = np.array(near[n_idx])
            cos_sim = dot(t, n)/(norm(t)*norm(n))
            #scores[t_idx,j] = int(cos_sim*100)
            if cos_sim < 0.3 and not n_idx in candidates:
                candidates += [n_idx]
        t_idx += 1

    # for t_idx in range(num_target):
    #     for n_idx in range(num_near):
    #         if scores[t_idx,n_idx] < 25 and not n_idx in candidates:
    #             candidates += [n_idx]
    #     if t_idx % 1000 == 0:
    #         print("process {} / {}".format(t_idx, num_target))
    return candidates


# find nearest neighbor images of Target DB in Near(Training) DB
def find_nearest_neighbor(unique_indices = None):
    print("---------find_nearest_neighbor---------")

    target_data = dict({"name" : [], "code" : []})
    near_data = dict({"name" : [], "code" : []})
    
    if "220425" in args.near_csv:
        near_data["code"] = read_csv_v2(args.near_csv, dim=args.feature_dim, csvtype='code')
        near_data["name"] = read_csv_v2(args.near_csv, dim=args.feature_dim, csvtype='name')
    else:
        near_data["name"] = read_csv(args.near_csv, dim=args.feature_dim, csvtype='name')
        near_data["code"] = read_csv(args.near_csv, dim=args.feature_dim, csvtype='code')

    target_data["name"] = read_csv(args.target_csv, dim=args.feature_dim, csvtype='name')
    target_data["code"] = read_csv(args.target_csv, dim=args.feature_dim, csvtype='code')
    
    if unique_indices is not None:
        tmp_target_name = []
        tmp_target_code = []
        for idx in unique_indices:
            tmp_target_name.append(target_data["name"][idx])
            tmp_target_code.append(target_data["code"][idx])
        target_data["name"] = tmp_target_name
        target_data["code"] = tmp_target_code
    
    print("--CSV INFO--")
    print("Target name len : {} , code len : {} / Near name len : {}, code len : {}".format(len(target_data['name']),len(target_data['code']),len(near_data['name']),len(near_data['code'])))
    print("Target data dim : {}, Near data dim : {}".format(len(target_data['code'][0]), len(near_data['code'][0])))
    features_target = FeatureBank(len(target_data['code']), len(target_data['code'][0]))
    features_near = FeatureBank(len(near_data['code']), len(near_data['code'][0]))
    

    features_target.update(target_data["code"])
    features_near.update(near_data['code'])

    print("features_target shape : ", features_target.features.shape)
    print("features_near shape : ", features_near.features.shape)
    
    # nn_factors = np.sort(np.array(args.nn_factor))[::-1]

    nn_indices_list = []
    for nn_f in tqdm(args.nn_factor, desc="Compare similarity between target and near and get nn indices", mininterval = 0.1):
        indices = nearest_neighbors_of_each_feature(topk=nn_f, near_feature_bank=features_near, target_feature_bank=features_target)
        np.save(args.save_dir+"/topk"+str(nn_f)+"_neighbors_val_from_train.npy",indices)
        nn_indices_list.append(indices)
    # sampling(target_data["name"], near_data['name'], nn_indices_list[0])
    
    return nn_indices_list, near_data["name"]

# find nearest neighbor images of each Near(Training) DB
def find_nearest_neighbor_self_data():
    print("---------find_nearest_neighbor_self_data---------")
    target_data = dict({"name" : [], "code" : []})
    target_data["name"] = read_csv(args.target_csv, dim=args.feature_dim, csvtype='name')
    target_data["code"] = read_csv(args.target_csv, dim=args.feature_dim, csvtype='code')
    
    print("--CSV INFO--")
    print("Target name len : {} , code len : {} ".format(len(target_data['name']),len(target_data['code'])))
    print("Target data dim : {}".format(len(target_data['code'][0])))
    features_target = FeatureBank(len(target_data['code']), len(target_data['code'][0]))
    features_target.update(target_data["code"])
    print("features_target shape : ", features_target.features.shape)
    
    distances, indices = nearest_neighbors(topk=args.nn_factor[0], target_feature_bank=features_target)
    np.save(args.save_dir+"/topk"+str(args.nn_factor[0])+"_neighbors_target.npy",indices)
    print(distances.shape)
    
    unique_data_indices = filtering_sequential_data(distances=distances, indices=indices, thr=args.dist_thr)
    print("unique data length : ", len(unique_data_indices))
    
    return unique_data_indices
    #sampling(target_data["name"], indices)


def pedrider_maker():
    data = np.load(args.nn_npy)
    print(data.shape)

    target_names = read_csv(args.target_csv, dim=args.feature_dim, csvtype='name')
    near_names = read_csv(args.near_csv, dim=args.feature_dim, csvtype='name')

    target_anno_dir = args.target_db + "/Annotations/"
    near_anno_dir = args.near_db + "/Annotations/"
    
    target_cls = get_name_xml(target_anno_dir, target_names)
    near_cls = get_name_xml(near_anno_dir, near_names)

    n = data.shape[0]    
    pedrider_img = dict({"ped" : [], "rider" : []})
    other_img = dict({"ped" : [], "rider" : []})

    candidates = np.array([], dtype=np.int64)
    
    for i in range(n):
        if i % 1000 == 0:
            print(i, "th process")
        if "pedestrian" in target_cls[i] or "rider" in target_cls[i]:
            ped_cnt = 0
            rider_cnt = 0
            for d in data[i][1:]:
                ped_cnt += near_cls[d].count("pedestrian")
                rider_cnt += near_cls[d].count("rider")
            ped_cnt /= len(data[i])
            rider_cnt /= len(data[i])
            pedrider_img['ped'].append(ped_cnt)
            pedrider_img['rider'].append(rider_cnt)
            candidates = np.concatenate((candidates, data[i][1:args.nn_factor[0]*2+1]))
            print(target_names[data[i][0]], near_names[data[i][1]], near_names[data[i][2]], near_names[data[i][3]], near_names[data[i][4]])
            sys.exit(1)
        else:
            ped_cnt = 0
            rider_cnt = 0
            for d in data[i][1:]:
                ped_cnt += near_cls[d].count("pedestrian")
                rider_cnt += near_cls[d].count("rider")
            ped_cnt /= len(data[i])
            rider_cnt /= len(data[i])
            other_img['ped'].append(ped_cnt)
            other_img['rider'].append(rider_cnt)
            candidates = np.concatenate((candidates, data[i][1:args.nn_factor[0]+1]))
    print("Pedrider img mean counts : ped{}, rider{}".format(Average(pedrider_img['ped']), Average(pedrider_img['rider']) ))
    print("Other img mean counts : ped{}, rider{}".format(Average(other_img['ped']), Average(other_img['rider']) ))

    print("Total sampling data (before unique) : ", candidates.shape)
    unique_cand = np.unique(candidates)
    print("Total sampling data (after unique) : ",unique_cand.shape)
    
    dir_imgset = args.save_dir + "/ImageSets/"
    if not os.path.exists(dir_imgset):
        os.mkdir(dir_imgset)
    if os.path.exists(dir_imgset+"scan.txt"):
        print("Already exist scan.txt, so remove it")
        os.remove(dir_imgset+"scan.txt")
    f = open(dir_imgset+"scan.txt", "w")
    db_counts = 0
    for i, cand in enumerate(unique_cand):
        if i % 1000 == 0:
            print("Make nn db - process {}/{} , the number of data made now : {}".format(i, len(unique_cand),db_counts))
        img_name = near_names[cand]
        pure_name = img_name.replace(".png", "")
        pure_name = pure_name.replace(".jpg", "")
        db_counts += 1
        f.writelines(pure_name+"\n")
    
    print("Total nearest data : " ,db_counts)

def make_imgset(_near_indices : list, near_data_name : list):
    print("---------Make_imgset----------")
    if _near_indices is None:
        print("Loading nn_npy")
        near_indices = [np.load(args.nn_npy)]
    else:
        near_indices = _near_indices
    
    near_imgsets = []
    idx = 0
    for data in tqdm(near_indices, desc="Make ImageSets", mininterval = 0.1):

        # target_names = read_csv(args.target_csv, dim=args.feature_dim, csvtype='name')
        # near_names = read_csv(args.near_csv, dim=args.feature_dim, csvtype='name')

        n = data.shape[0]    
        candidates = np.array([], dtype=np.int64)
        for i in range(n):
            # if i % 1000 == 0:
            #     print(i, "th process")
            candidates = np.concatenate((candidates, data[i][1:args.nn_factor[idx]+1]), axis = 0)

        unique_cand = np.unique(candidates)
        print("nn {} Total sampling data : (before) {} (after unique) {}".format(args.nn_factor[idx], candidates.shape, unique_cand.shape))

        dir_imgset = args.save_dir + "/ImageSets/"
        if not os.path.exists(dir_imgset):
            os.mkdir(dir_imgset)
        name_imgset = dir_imgset+"scan_nn"+str(args.nn_factor[idx])+".txt"
        if os.path.exists(name_imgset):
            print("Already exist scan.txt, so remove it")
            os.remove(name_imgset)
        f = open(name_imgset, "w")
        db_counts = 0
        for i, cand in enumerate(unique_cand):
            if i % 10000 == 0:
                print("nn {} make db - process {}/{} , the number of data made now : {}".format(args.nn_factor[idx], i, len(unique_cand),db_counts))
            img_name = near_data_name[cand]
            pure_name = img_name.replace(".png", "")
            pure_name = pure_name.replace(".jpg", "")
            db_counts += 1
            f.writelines(pure_name+"\n")
        print("nn {} Total nearest data : {}".format(args.nn_factor[idx], db_counts))
        idx += 1
        near_imgsets += [name_imgset]
    
    return near_imgsets

def imgset_check(near_imgsets : list):
    
    for imgset in near_imgsets:
        img_list = []
        with open(imgset, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                img_list.append(line)

        anno_path = args.near_db + "Annotations/"
        for anno in tqdm(img_list, desc="Check ImageSet possible error", mininterval = 0.1):
            if not os.path.exists(anno_path+anno+".xml"):
                print(anno)

def sampling(target_names, near_names, indices):
    total_imgs = len(indices)
    JPEG_path_target = args.target_db + "/JPEGImages/"
    JPEG_path_near = args.near_db + "/JPEGImages/"
    for i, idx in enumerate(indices):
        # if os.path.exists(args.save_dir+"/nn_"+str(i)+target_names[idx[0]]+".jpg"):
        #     continue
        if i % 100 == 0:
            print("sampling nn - process {}/{}".format(i, total_imgs))
        img_src = cv.imread(JPEG_path_target + target_names[idx[0]]+".jpg")
        img_src = cv.resize(img_src,(236,236))
        img_num = args.nn_factor[0]+1
        for j in range(1,img_num):
            img_near = cv.imread(JPEG_path_near + near_names[idx[j]]+".jpg")
            img_near = cv.resize(img_near,(236,236))
            img_src = cv.hconcat([img_src,img_near])
        cv.imwrite(args.save_dir+"/nn_"+str(i)+target_names[idx[0]]+".jpg", img_src)

def do_sampling():
    target_data = dict({"name" : [], "code" : []})
    near_data = dict({"name" : [], "code" : []})
    print("Read CSV")
    target_data["name"] = read_csv(args.target_csv, dim=args.feature_dim, csvtype='name')
    near_data["name"] = read_csv(args.near_csv, dim=args.feature_dim, csvtype='name')

    print("Loading nn_npy")
    data = np.load(args.nn_npy)
    sampling(target_data["name"], target_data['name'], data)

if __name__ == "__main__":
    args = arg_parse()
    print(args)
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    # do_sampling()
    
    #Remove the sequential frames in target DB using feature's cosine similarity
    unique_indices = find_nearest_neighbor_self_data()
    
    #Make Nearest-neighbor of each target data using target_csv and near_csv (features)
    near_indices, near_data_name = find_nearest_neighbor(unique_indices) 

    #Make ImageSets by sampling top-k nearest neighbors
    near_imgsets = make_imgset(near_indices, near_data_name)
    
    #pedrider_maker()

    imgset_check(near_imgsets)