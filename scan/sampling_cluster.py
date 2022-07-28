import csv
import os,sys
import numpy as np
import argparse
import xml.etree.ElementTree as elemTree

#T224
# class_string = ["pedestrian", "rider_bicycle", "rider_bicycle_2", "rider_bike", "rider_bike_2", "rider_bicycle_human_body", \
#     "rider_bike_human_body", "bicycle", "bike", "3-wheels_rider", "3-wheels", "sedan", "van", "pickup_truck", "truck", "mixer_truck", \
#     "excavator", "forklift", "ladder_truck", "truck_etc", "vehicle_etc", "vehicle_special", "box_truck", "trailer", "bus", \
#     "ehicle_special", "sitting_person", "wheel_chair", "ignored", "false_positive", "animal", "bird", "animal_ignored", \
#     "ts_circle", "ts_circle_speed", "ts_triangle", "ts_inverted_triangle", "ts_rectangle", "ts_rectangle_speed", "ts_diamonds", \
#     "ts_supplementary", "tl_car", "tl_ped", "tl_special", "tl_light_only", "ts_ignored", "tl_ignored", "tstl_ignore", \
#     "ts_sup_ignored", "ts_sup_letter", "ts_sup_drawing", "ts_sup_arrow", "ts_sup_zone", "ts_main_zone", "ts_rectangle_arrow",\
#     "ts_chevron", "tl_rear", "ts_rear", "tl_counter", "obstacle_bollard_barricade", "obstacle_bollard_cylinder", \
#     "obstacle_bollard_marker", "obstacle_cone", "obstacle_drum", "obstacle_cylinder", "obstacle_bollard_special", \
#     "obstacle_bollard_stone", "obstacle_bollard_u_shaped", "parking_cylinder", "parking_sign", "parking_stopper_separated", \
#     "parking_stopper_bar", "parking_stopper_marble", "parking_special", "parking_lock", "blocking_bar", "blocking_special", \
#     "blocking_ignored", "parking_ignored", "obstacle_ignored", "stopline_normal", "stopline_special", "stopline_ignored", \
#     "arrow_normal", "arrow_special", "arrow_ignored", "crosswalk_normal", "crosswalk_special", "crosswalk_ignored", \
#     "speedbump_normal", "speedbump_special", "speedbump_ignored", "number_speed", "number_parkingzone", "number_special", \
#     "number_ignored", "text_normal", "text_parkingzone", "text_special", "text_ignored", "roadmark_triangle", "roadmark_diamond", \
#     "roadmark_bicycle", "roadmark_handicapped", "roadmark_pregnant", "roadmark_special", "roadmark_ignored"]
# class_index = [1, 2, 2, 3, 3, -100000, -100000, -100000, -100000, 3, -100000, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 5, 1, 1, -100000,\
#     0, 0, 0, 0, 7, 7, 7, 7, 7, 7, 7, -100000, 8, -100000, -100000, -100000, -100000, -100000, -100000, -100000, 7, 7, 7, 7, -100000, -100000,\
#     -100000, -100000, -100000, -100000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\
#     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

#M576
##class_names: "[pedestrian, bicycle, motorbike, car, truck, bus, ts, tl, obstacle_cone, obstacle_cylinder, obstacle_general, arrow, diamond, stopline, crosswalk]"
class_string = ["pedestrian", "rider_bicycle", "rider_bicycle_2", "rider_bike", "rider_bike_2", "rider_bicycle_human_body", "rider_bike_human_body", "bicycle", "bike", "3-wheels_rider", "3-wheels", "sedan", "van", "pickup_truck", "truck", "mixer_truck", "excavator", "forklift", "ladder_truck", "truck_etc", "vehicle_etc", "vehicle_special", "box_truck", "trailer", "bus", "ehicle_special", "sitting_person", "wheel_chair", "ignored", "false_positive", "animal", "bird", "animal_ignored", "ts_circle", "ts_circle_speed", "ts_triangle", "ts_inverted_triangle", "ts_rectangle", "ts_rectangle_speed", "ts_diamonds", "ts_supplementary", "tl_car", "tl_ped", "tl_special", "tl_light_only", "ts_ignored", "tl_ignored", "tstl_ignore", "ts_sup_ignored", "ts_sup_letter", "ts_sup_drawing", "ts_sup_arrow", "ts_sup_zone", "ts_main_zone", "ts_rectangle_arrow", "ts_chevron", "tl_rear", "ts_rear", "tl_counter", "obstacle_bollard_barricade", "obstacle_bollard_cylinder", "obstacle_bollard_marker", "obstacle_cone", "obstacle_drum", "obstacle_cylinder", "obstacle_bollard_special", "obstacle_bollard_stone", "obstacle_bollard_u_shaped", "parking_cylinder", "parking_sign", "parking_stopper_separated", "parking_stopper_bar", "parking_stopper_marble", "parking_special", "parking_lock", "blocking_bar", "blocking_special", "blocking_ignored", "parking_ignored", "obstacle_ignored", "sod_ignored", "stopline_normal", "stopline_special", "stopline_ignored", "arrow_normal", "arrow_special", "arrow_ignored", "crosswalk_normal", "crosswalk_special", "crosswalk_ignored", "speedbump_normal", "speedbump_special", "speedbump_ignored", "number_speed", "number_parkingzone", "number_special", "number_ignored", "text_normal", "text_parkingzone", "text_special", "text_ignored", "roadmark_triangle", "roadmark_arrow", "roadmark_diamond", "roadmark_stopline", "roadmark_crosswalk", "roadmark_notice_cross_intersection", "roadmark_notice_t_intersection", "roadmark_speed", "roadmark_bicycle", "roadmark_handicapped", "roadmark_pregnant", "roadmark_special", "roadmark_ignored", "sit_person"]
class_index = [         1,             2,               2,          3,            3,                  -100000,               -100000, -100000, -100000,              3,  -100000,     4,   4,            4,     5,           5,         5,        5,            5,         5,           5,               5,         5,       5,   6,              5,              1,           1, -100000,              0,      0,    0,              0,         7,               7,           7,                    7,            7,                  7,           7,          -100000,      8, -100000,     -100000,             8,    -100000,    -100000,     -100000,        -100000,             7,              7,            7,           7,      -100000,            -100000,    -100000, -100000, -100000,    -100000,                         11,                        10,                      10,             9,            11,                10,                       11,                     11,                        11,               10,            0,                         0,                   0,                      0,               0,            0,            0,                0,                0,               0,          -100000,     -100000,              14,          -100000,          -100000,           12,       -100000,       -100000,               15,           -100000,           -100000,          -100000,           -100000,           -100000,      -100000,            -100000,        -100000,        -100000,     -100000,          -100000,      -100000,      -100000,           -100000,             12,               13,                14,                 15,                            -100000,                        -100000,        -100000,          -100000,              -100000,           -100000,          -100000,          -100000, -100000]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_imgset', dest='target_imgset', help='target imageset',
                        default=None, type=str)
    parser.add_argument('--target_data', dest='target_data', help='target csv file to read',
                        default="/damin/data/GODTrain220425/", type=str)
    parser.add_argument('--output_dir', dest='output_dir', help='save directory',
                        default=None, type=str)
    parser.add_argument('--num_cluster', dest='num_cluster', help='the number of clusters',
                        default=None, type=int)
    parser.add_argument('--total_sampling', dest='total_sampling', help='the total sampling counts',
                        default=None, type=int)
    parser.add_argument('--num_class', dest='num_class', help='the number of class',
                        default=15, type=int)
    parser.add_argument('--analysis', dest='analysis', help='analyze the cluster',
                        default=False, type=bool)
    parser.add_argument('--show', dest='show', help='save the cluster images to output_dir',
                        default=False, type=bool)
    args = parser.parse_args()
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return args

def parseCSVData(csvfile):
    with open(csvfile, newline='') as f:
        lines = csv.reader(f, delimiter=' ')
        score_list = []
        name_list = []
        group_idx = -1
        for line in lines:
            
            #name, group_idx, score = ', '.join(line).split(",")[:]
            data_list = ', '.join(line).split(",")[:]
            score = data_list[-1]
            group_idx = data_list[-2]
            name = ",".join(data_list[:-2])
            # if len(data_list) > 3:
            #     print(name,score,group_idx, data_list)
            name = name.replace(".jpg","")
            name = name.replace(".png","")
            score_list += [float(score)]
            name_list += [name]
        score_list = np.array(score_list)
        sort_index = np.argsort(-score_list) #sorted(range(len(score_list)), key=lambda k: score_list[k])
        
        name_list = [name_list[i] for i in sort_index]
        score_list = [score_list[i] for i in sort_index]
        
        return name_list, score_list, group_idx

'''
input : annotation file xml_path type : str
output : the counts of classes. type : np.array
'''
def parseXMLData(xmlfile):
    xml_doc_2d = elemTree.parse(xmlfile)
    xml_root_2d = xml_doc_2d.getroot()
    objects = xml_root_2d.findall('object')
    class_list = np.zeros(args.num_class, dtype=np.uint8)
    for idx, obj in enumerate(objects):
        name = obj.find('name').text.lower()
        name = name.replace('night_', '')
        fixed_class = class_index[class_string.index(name)]
        if fixed_class == -100000 or fixed_class == 0:
            continue
        class_list[fixed_class-1] += 1
    return class_list

cls_weight_couting = [0.033363865, 0.076716555, 0.064226838, 0.01780908, 0.05422874, 0.075879928, 0.040558113, 0.059448397, 0.091303267,\
                      0.050973443, 0.068891334, 0.075934542, 0.118413933, 0.088029836, 0.08422213]

def parseXMLDataCounts(xmlfile, weight=False):
    xml_doc_2d = elemTree.parse(xmlfile)
    xml_root_2d = xml_doc_2d.getroot()
    objects = xml_root_2d.findall('object')
    cnt = 0
    for idx, obj in enumerate(objects):
        name = obj.find('name').text.lower()
        name = name.replace('night_', '')
        fixed_class = class_index[class_string.index(name)]
        if fixed_class == -100000 or fixed_class == 0:
            continue
        if not weight:
            cnt += 1
        elif weight:
            if fixed_class < 9:
                cnt += cls_weight_couting[fixed_class-1]
            elif fixed_class > 8:
                cnt += 0.5 * cls_weight_couting[fixed_class-1]
    return cnt


# def filterClusterData(data, cluster_data, use_class_info = False):
#     filter_cluster_data = []
#     filter_cluster_class_info = []
#     for i in range(args.num_cluster):
#         csv_name = "./cluster"+str(i)+"/data_nothreshold.csv"
#         name_list, score_list, group_idx = cluster_data[i]['name'], cluster_data[i]['score'], cluster_data[i]['group_idx']
        
#         ss_index = min(range(len(score_list)), key=lambda k: abs(score_list[k]-0.5))

#         score_list = score_list[:ss_index+1]
#         name_list = name_list[:ss_index+1]
        
#         total_num_of_class = np.zeros(args.num_class,dtype=np.uint64)
#         cdata = {}
#         if len(score_list) == 0:
#             filter_cluster_class_info.append(total_num_of_class)
#             filter_cluster_data.append(cdata)
#         else:
#             data = set(data)
#             for j, (name, score) in enumerate(zip(name_list, score_list)):
#                 if name in data:
#                     if use_class_info:
#                         xml_path = args.target_data + "Annotations/" + name + ".xml"
#                         class_info = parseXMLData(xml_path)
#                         total_num_of_class += class_info
#                     if not cdata:
#                         cdata['name'] = [name]
#                         cdata['index'] = [j]
#                         cdata['score'] = [score]
#                     else:
#                         cdata['name'] += [name]
#                         cdata['index'] += [j]
#                         cdata['score'] += [score]

#             filter_cluster_class_info.append(total_num_of_class)
#             filter_cluster_data.append(cdata)
#             print("filter cluster{} : {}".format(i, len(filter_cluster_data[i]['index'])))
#             if use_class_info:
#                 print("cluster{}'s class : {}".format(i, total_num_of_class))
            
#     return filter_cluster_data, filter_cluster_class_info

def filterClusterData(data, cluster_data, use_class_info = False):
    filter_cluster_data = []
    filter_cluster_class_info = []
    for i in range(args.num_cluster):
        #csv_name = "./cluster"+str(i)+"/data_nothreshold.csv"
        name_list, score_list, group_idx = cluster_data[i]['name'], cluster_data[i]['score'], cluster_data[i]['group_idx']
        
        ss_index = min(range(len(score_list)), key=lambda k: abs(score_list[k]-0.5))

        score_list = score_list[:ss_index+1]
        name_list = name_list[:ss_index+1]
        
        total_num_of_class = np.zeros(args.num_class,dtype=np.uint64)
        cdata = {}

        data = set(data)
        for j, (name, score) in enumerate(zip(name_list, score_list)):
            if name in data:
                if use_class_info:
                    xml_path = args.target_data + "Annotations/" + name + ".xml"
                    class_info = parseXMLData(xml_path)
                    total_num_of_class += class_info
                if not cdata:
                    cdata['name'] = [name]
                    cdata['index'] = [j]
                    cdata['score'] = [score]
                else:
                    cdata['name'] += [name]
                    cdata['index'] += [j]
                    cdata['score'] += [score]
        
        #ignore SOD, Roadmark when calculating weight sampling
        total_num_of_class = total_num_of_class[:8]
        filter_cluster_class_info.append(total_num_of_class)
        filter_cluster_data.append(cdata)
        
        print("filter cluster{} : {}".format(i, len(filter_cluster_data[i]['index'])))
        if use_class_info:
            print("cluster{}'s class : {}".format(i, total_num_of_class))
            
    return filter_cluster_data, filter_cluster_class_info


def getSamplingWeight(class_distribution, num_cluster, eps = 1e-7):
    class_distribution = np.array(class_distribution)
    normalized_class_distribution = np.zeros_like(class_distribution, dtype=np.float64)
    weighted_distribution = np.zeros(num_cluster, dtype=np.float64)

    #get the number of data per class
    total_counts_per_class = np.sum(class_distribution,axis=0)
    
    #get class's weight in total dataset
    weight_class =  1 - total_counts_per_class / np.sum(total_counts_per_class,axis=0)
    
    print("class weight_class : ", weight_class)
    
    for i in range(normalized_class_distribution.shape[0]):
        normalized_class_distribution[i,:] = class_distribution[i,:] / (total_counts_per_class[:] + eps)
        weighted_distribution[i] = np.sum(normalized_class_distribution[i,:] * weight_class[:], axis=0)

    weighted_distribution /= np.sum(weighted_distribution, axis=0)
    
    print("weighted_distribution : " , weighted_distribution)

    return weighted_distribution


def sampleHighScoreOrder(cluster_data, filtered_cluster_data, target_counts_per_cluster : list):
    sampling_name = []
    for i in range(args.num_cluster):
        name_list, indicies, score_list = cluster_data[i]['name'], filtered_cluster_data[i]['index'], filtered_cluster_data[i]['score']
        
        target_counts = target_counts_per_cluster[i]
        sampling_counts = 0
        last_score = 0
        for j, (index, score) in enumerate(zip(indicies, score_list)):
            sampling_counts += 1
            sampling_name += [name_list[index]]
            last_score = float(score)
            
            if sampling_counts % 5000 == 0:
                print("{}/{}, skip {} images".format(sampling_counts, target_counts, j + 1 - sampling_counts))

            if sampling_counts == target_counts:
                print("cluster"+str(i) + " sampling {} done, min score: {}, skip images: {}".format(target_counts, last_score,  j + 1 - sampling_counts))
                break
        if sampling_counts < target_counts:
            print("cluster"+str(i) + " sampling {} done, min score :{}".format(sampling_counts, last_score))
    return sampling_name

def sampleSelectiveRandomOrder(cluster_data, filtered_cluster_data, target_counts_per_cluster : list):
    sampling_name = []
    #Select random indices
    #score[1,0.9] : 3000, score[0.9,0.8] : 3000, ... score[0.6,0.5] : 3000
    _sampling_count = 0
    score_step = [0.9, 0.8, 0.7, 0.6, 0.5]
    for i in range(args.num_cluster):
        #print("CLUSTER {}".format(i))
        target_counts = target_counts_per_cluster[i]
        target_counts_per_sector = round(target_counts/5)
        np.random.seed(0)
        origin_names = cluster_data[i]['name']
        name, index, score =filtered_cluster_data[i]['name'], filtered_cluster_data[i]['index'], filtered_cluster_data[i]['score']
        
        #valid_index = []
        #get number of data in each score step
        min_index = 0
        for j in range(5):
            #find the index of nearset score with 0.5
            ss_index = min(range(len(score)), key=lambda k: abs(score[k]-score_step[j]))
            
            #num of data in score section is under target_counts_per_sector.
            if ss_index - min_index < target_counts_per_sector:
                #valid_index += index[min_index:ss_index+1]
                sampling_name += name[min_index:ss_index+1]
            else:
                rand_index = np.random.randint(min_index, ss_index, size=target_counts_per_sector)
                #valid_index += rand_index.tolist()
                sampling_name += [name[ri] for ri in rand_index]
                
            #print("score_step {} index {} min_index {} => valid_index {}".format(score_step[j], ss_index, min_index, len(valid_index)))
            min_index = ss_index + 1
        
        print("cluster"+str(i) + " sampling {} done", len(sampling_name) - _sampling_count)
        _sampling_count = len(sampling_name)
        #sampling_name += [name[v] for v in valid_index]

    return sampling_name
        

def sampleGTPriorityOrder(cluster_data, filtered_cluster_data, target_counts_per_cluster : list):
    sampling_name = []
    for i in range(args.num_cluster):
        name_list, indicies, score_list = cluster_data[i]['name'], filtered_cluster_data[i]['index'], filtered_cluster_data[i]['score']
        target_counts = target_counts_per_cluster[i]
        #read annotations
        gt_cnts_per_imgs = np.zeros(len(indicies), dtype=np.float32)
        name_list_per_gtcnts = np.zeros(len(indicies), dtype=np.int64)
        
        for j, (index, score) in enumerate(zip(indicies, score_list)):
            anno_path = args.target_data + "Annotations/" + name_list[index] + ".xml"
            cnts = parseXMLDataCounts(anno_path, weight = True)
            gt_cnts_per_imgs[j] = cnts
            name_list_per_gtcnts[j] = index
            if j % 1000 == 0 and j != 0:
                print("cluter{} {}/{}".format(i,j, len(indicies)))

        a = np.random.randint(0, 100, size=1)
        np.random.seed(a)

        bigger_gt_cnts = np.argsort(-gt_cnts_per_imgs)
        for j in bigger_gt_cnts[:target_counts]:
            sampling_name += [name_list[name_list_per_gtcnts[j]]]
    return sampling_name

if __name__ == "__main__":
    args = parse_args()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    od_tstld_data_list = []
    
    if args.target_imgset is not None:
        with open(args.target_imgset, 'r') as f:
            lines = f.readlines()
            for l in lines:
                l = l.replace("\n","")
                od_tstld_data_list += [l]
        save_imgset = args.target_imgset[args.target_imgset.rfind('/') + 1:].replace(".txt", "")+"_scan.txt"
    else:
        print("target_imgset is None value")
        sys.exit(1)
            
    print("Dataset size : {}".format(len(od_tstld_data_list)))

    #parsing CSV cluster data
    cluster_data = []
    for i in range(args.num_cluster):
        cdata = {}
        print("cluster"+str(i))
        csv_name = "./clusters/cluster"+str(i)+"/data_nothreshold.csv"
        name_list, score_list, group_idx = parseCSVData(csv_name)
        
        if len(score_list) == 0:
            continue

        cdata['name'] = name_list
        cdata['score'] = score_list
        cdata['group_idx'] = group_idx
        cluster_data.append(cdata)
    
    print("Cluster data : length {}".format(len(cluster_data)))
    
    args.num_cluster = len(cluster_data)
    
    #save the volume of all cluster data
    if args.analysis:
        with open('./cluster_analysis.txt', 'w') as f:
            for i in range(args.num_cluster):
                f.write('cluster,{},counts,{},mean,{}\n'.format(i,len(cluster_data[i]['name']), np.array(cluster_data[i]['score']).mean()))
    
    #filter_cluster_data has "index" and "score"
    #index : original index list of cluster_data
    #score : score list
    filtered_cluster_data, filter_cluster_class_info = filterClusterData(od_tstld_data_list, cluster_data, use_class_info=True)
    
    if args.show:
        print("show clustering result")
        import shutil
        for i, fdata in enumerate(filtered_cluster_data):
            print("{} cluster -save images".format(i))
            if not os.path.exists(args.output_dir + "/cluster_show"):
                os.mkdir(args.output_dir + "/cluster_show")
            for name, score in zip(fdata['name'], fdata['score']):
                shutil.copy(args.target_data+"JPEGImages/"+name+".png",args.output_dir + "/cluster_show/" + str(i)+"_"+str(int(score*1000))+".png")
    
    #save class info distribution per each cluster
    if args.analysis:
        with open('./class_distribution_per_cluster.csv', 'w', newline="") as f:
            csvwriter = csv.writer(f)
            for ci in filter_cluster_class_info:
                csvwriter.writerow(ci)

    #GT awared sampling
    weighted_distribution = getSamplingWeight(filter_cluster_class_info, num_cluster=args.num_cluster)

    if weighted_distribution is None:
        weight_sampling = [args.total_sampling // args.num_cluster for _ in range(args.num_cluster)]
    else:
        weight_sampling = [int(args.total_sampling * weighted_distribution[wd]) for wd in range(weighted_distribution.shape[0])]
    print("weight_sampling : ", weight_sampling)

    #sampling_name = sampleHighScoreOrder(cluster_data, filtered_cluster_data, weight_sampling)
    
    #sampling_name = sampleSelectiveRandomOrder(cluster_data, filtered_cluster_data, weight_sampling)
    
    sampling_name = sampleGTPriorityOrder(cluster_data, filtered_cluster_data, weight_sampling)
    
    #analyze the data
    cls_cnts = np.zeros(args.num_class, dtype=np.uint64)
    for name in sampling_name:
        anno_path = args.target_data + "Annotations/" + name + ".xml"
        cls_cnt = parseXMLData(anno_path)
        cls_cnts += cls_cnt
    print("{} class counts : {}".format(args.num_class, cls_cnts))

    #check data
    od_tstld_data_list = set(od_tstld_data_list)
    for sn in sampling_name:
        if sn not in od_tstld_data_list:
            print(sn)
            print("Error, Exist data not in the target_imgset")
            

    print("Total sampling counts : {}".format(len(sampling_name)))
    
    with open(args.output_dir+"/"+save_imgset, 'w') as f:
        for name in sampling_name:
            f.write(name +"\n")
    f.close()
        
        
        
        
        

    