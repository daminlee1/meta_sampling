import argparse
import cv2
import csv, sys, os
import numpy as np
import faiss
from eval_meta import *
 
def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_nn', dest='target_nn', help='nearset neighbor npy file',
                        default=None, type=str)
    parser.add_argument('--target_csv', dest='target_csv', help='target csv file',
                        default=None, type=str)
    parser.add_argument('--near_csv', dest='near_csv', help='near csv file to read',
                        default=None, type=str)
    parser.add_argument('--output_path', dest='output_path', help='output directory',
                        default=None, type=str)
    parser.add_argument('--target_db', dest='target_db', help='target_db',
                        default=None, type=str)
    parser.add_argument('--near_db', dest='near_db', help='near_db',
                        default=None, type=str)
    parser.add_argument('--nn_factor', dest='nn_factor', help='nn_factor',
                        default=None, type=int)
    parser.add_argument('--dist_thr', dest='dist_thr', help='dist_thr',
                        default=0.98, type=float)
    args = parser.parse_args()
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return args

roads = ["city", "highway", "rural", "etc"]
timezones = ["day", "night", "dawn_evening"]
weathers = ["clear", "rainy","heavy-rainy","snow","fog", 'etc']

def read_csv(csv_file, dim = None, csvtype = "name"):
    data = []
    if csvtype == "name":
        with open(csv_file, 'r') as f:
            reader = csv.reader(f, delimiter="\t")
            for i, line in enumerate(reader):
                line = line[0].split(",")
                data.append(str(line[0])[:-1])
    elif csvtype == "code":
        with open(csv_file, 'r') as f:
            reader = csv.reader(f, delimiter="\t")
            for i, line in enumerate(reader):
                line = line[0].split(",")
                line_float = [float(i) for i in line[1:dim+1]]
                data.append(line_float)
    else :
        print("unknown csvtype")
        sys.exit(1)    
    return data

def sampling(target_names, near_names, indices, metadata = None):
    total_imgs = len(indices)
    JPEG_path_target = args.target_db + "/JPEGImages/"
    JPEG_path_near = args.near_db + "/JPEGImages/"
    for i, idx in enumerate(indices):
        if os.path.exists(args.output_path+"/nn_"+str(i)+target_names[idx[0]]):
            continue
        if i % 100 == 0:
            print("sampling nn - process {}/{}".format(i, total_imgs))
            print(JPEG_path_target + target_names[idx[0]])
            img_src = cv2.imread(JPEG_path_target + target_names[idx[0]])
            img_src = cv2.resize(img_src,(236,236))
            if metadata is not None:
                cv2.putText(img_src, str(roads.index(metadata[idx[0]]['road'])) + str(timezones.index(metadata[idx[0]]['time_zone'])) + str(weathers.index(metadata[idx[0]]['weather_item'])), (110,200),
                                         cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0),2)
            img_num = args.nn_factor+1
            for j in range(1,img_num):
                img_near = cv2.imread(JPEG_path_near + near_names[idx[j]])
                img_near = cv2.resize(img_near,(236,236))
                if metadata is not None:
                    cv2.putText(img_near, str(roads.index(metadata[idx[0]]['road'])) + str(timezones.index(metadata[idx[0]]['time_zone'])) + str(weathers.index(metadata[idx[0]]['weather_item'])), (110,200),
                                             cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0),2)
                img_src = cv2.hconcat([img_src,img_near])
            cv2.imwrite(args.output_path+"/nn_"+str(i)+target_names[idx[0]], img_src)

def nearest_neighbors(topk, target_feature_bank):
    feature_target = target_feature_bank.features
    n, dim = feature_target.shape[0]+1, feature_target.shape[1]
    index = faiss.IndexFlatIP(dim)
    index = faiss.index_cpu_to_all_gpus(index)
    index.add(feature_target)
    distances, indices = index.search(feature_target, topk+1)
    return distances, indices

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

# find the sequential frames in DB and remove it.
def filtering_sequential_data(distances, indices, thr = 0.98, names = None):
    cand_list = np.arange(len(indices))
    for i, c in enumerate(cand_list):
        if c == -1:
            continue
        #delete target index in distances
        near_dist = np.delete(distances[c], 0)
        seq_dist = np.reshape(np.argwhere(near_dist>=thr),(-1))
        seq_indices = [indices[c,idx] for idx in seq_dist]
        # #sampling the near images
        # original_img = cv.imread(args.target_db + "JPEGImages/"+names[i])
        # original_img = cv.resize(original_img,(236,236))
        # if len(seq_indices) == 0:
        #     continue
        # for seq_idx in seq_indices:
        #     img = cv.imread(args.target_db + "JPEGImages/"+names[seq_idx])
        #     img = cv.resize(img, (236,236))
        #     original_img = cv.hconcat([original_img, img])
        # cv.imwrite("./aptiv_train_near/"+str(i)+ "_" + str(int(thr*100))+".jpg", original_img)
        #exclude the indices of sequential frame in cand_list 
        for s in seq_indices:
            cand_list[s] = -1

    seq_indices_cand = np.reshape(np.argwhere(cand_list==-1),(-1))
    print("seq_indices_cand : ", seq_indices_cand.shape)
    unique_cand_list = np.delete(cand_list, seq_indices_cand)
    print("unique_cand_list : ", unique_cand_list.shape)
    return unique_cand_list

def filter_similar_data(target_data):
    print("---------find_nearest_neighbor_self_data---------")
    
    print("--CSV INFO--")
    print("Target name len : {} , code len : {} ".format(len(target_data['name']),len(target_data['code'])))
    print("Target data dim : {}".format(len(target_data['code'][0])))
    features_target = FeatureBank(len(target_data['code']), len(target_data['code'][0]))
    features_target.update(target_data["code"])
    print("features_target shape : ", features_target.features.shape)
    
    distances, indices = nearest_neighbors(topk=args.nn_factor, target_feature_bank=features_target)
    #np.save(args.save_dir+"/topk"+str(args.nn_factor)+"_neighbors_target.npy",indices)
    print(distances.shape)
    
    unique_data_indices = filtering_sequential_data(distances=distances, indices=indices, thr=args.dist_thr, names=target_data["name"])
    print("unique data length : ", len(unique_data_indices))
    
    return unique_data_indices
    #sampling(target_data["name"], indices)

def mining_nearest_neighbor(target_data):
    print("---------find_nearest_neighbor_self_data---------")
    
    print("--CSV INFO--")
    print("Target name len : {} , code len : {} ".format(len(target_data['name']),len(target_data['code'])))
    print("Target data dim : {}".format(len(target_data['code'][0])))
    features_target = FeatureBank(len(target_data['code']), len(target_data['code'][0]))
    features_target.update(target_data["code"])
    print("features_target shape : ", features_target.features.shape)
    
    distances, indices = nearest_neighbors(topk=args.nn_factor, target_feature_bank=features_target)
    
    return distances, indices, features_target
    #sampling(target_data["name"], indices)

def save_feature(memory_bank, imageset, csv_name):
    base_features = memory_bank.features
    print("Base feature shapes : {},{}".format(base_features.shape[0],base_features.shape[1]))
    print("Saving base_features of trainDB")
    # with open(imageset, 'r', encoding='UTF-8') as f:
    #     img_names = f.readlines()
    #     for n in img_names:
    #         n = n.replace("\n","")
    with open(csv_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        print("num of base_features : ", base_features.shape[0])
        for i, feat in enumerate(base_features):
            csv_data = [imageset[i]]+feat.tolist()
            writer.writerows([csv_data])
    print("finish saving feature")

if __name__ == "__main__":
    args = arg_parse()
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)

    target_data = dict({"name" : [], "code" : []})
    target_data["name"] = read_csv(args.target_csv, csvtype='name')
    target_data["code"] = read_csv(args.target_csv, dim=2048, csvtype='code')
    
    nn_indices = np.load(args.target_nn)
    
    print(len(target_data['name']), nn_indices.shape)

    unique_data_list = filter_similar_data(target_data)
    
    unique_target_data = dict({'name': [], 'code' : []})
    for i, uidx in enumerate(unique_data_list):
        unique_target_data['name'].append(target_data['name'][uidx])
        unique_target_data['code'].append(target_data['code'][uidx])
    
    unique_nn_data_distance, unique_nn_data_indices, feature_bank = mining_nearest_neighbor(unique_target_data)
    
    print("unique_nn_data : ", unique_nn_data_indices.shape)
    np.save(args.output_path +"/topk_unique_nn.npy", unique_nn_data_indices )
    save_feature(feature_bank, unique_target_data['name'] , "SCANmoco_unique_svkpi3000km_dim2048_220217.csv")

    #Get meta-data of Target DB
    anno_list = []
    for i, tidx in enumerate(unique_nn_data_indices[:,0]):
        tname = unique_target_data['name'][tidx]
        anno_list.append(args.target_db + "/Annotations/" + tname + ".xml")
    metadata = read_meta_info(anno_list)
    analyze_meta_data(metadata)
    
    #add ".jpg" in name of targets
    for i, tname in enumerate(unique_target_data['name']):
        unique_target_data['name'][i] = tname + ".png"
    
    sampling(unique_target_data['name'], unique_target_data["name"], unique_nn_data_indices, metadata)
    
    #evaluate meta_info
    #calculate mean, var of nn_distance
    mean_nn_dist = [np.mean(nn_value[1:]) for nn_value in unique_nn_data_distance]
    var_nn_dist = [np.var(nn_value[1:]) for nn_value in unique_nn_data_distance] #np.var(unique_nn_data_distance[:,1:])

    #calculate the correspondence of Time, Road and weathers
    meta_nn = np.zeros((len(metadata),6))
    for i, nn_indices in enumerate(unique_nn_data_indices):
        target_meta = metadata[nn_indices[0]]
        r_count = 0 
        t_count = 0
        w_count = 0
        for nd in nn_indices[1:]:
            near_meta = metadata[nd]
            if target_meta['road'] == near_meta['road']:
                r_count += 1
            if target_meta['time_zone'] == near_meta['time_zone']:
                t_count += 1
            if target_meta['weather_item'] == near_meta['weather_item']:
                w_count += 1
            meta_nn[i, :] = [r_count, t_count, w_count, roads.index(target_meta['road']), timezones.index(target_meta['time_zone']), weathers.index(target_meta['weather_item'])]

    #Write the result to csv format 
    eval_data = {'index' : ["dist_mean", 'dist_var', 't_score', 'r_score', 'w_score', 'time', 'road', 'weather']}
    for i, (dm, dv, me) in enumerate(zip(mean_nn_dist, var_nn_dist, meta_nn)):
        ts, rs, ws, t, r, w = me[:]
        eval_data[i] = [dm, dv, ts, rs, ws, t, r, w]
        print(i)
        
        
    df = pd.DataFrame(eval_data)
    print(df)
    df = df.T
    df.to_csv('./svmoco_3000km_2048dim_analysis.csv')
    
    
    
    
    