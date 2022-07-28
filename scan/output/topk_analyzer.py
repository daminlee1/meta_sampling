import numpy as np
import os,sys,csv
from os import listdir, name
from os.path import isfile, join
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
from xml.etree.ElementTree import parse

def parse_args():
    parser = argparse.ArgumentParser(description="argument of topk_analyzer")
    parser.add_argument('--nn_file', dest='nn_file', help="neighbors_data",
                        default=None, type=str)
    parser.add_argument('--target_dir', dest='target_dir', help="Target image directory",
                        default=None, type=str)
    parser.add_argument('--nn_dir', dest='nn_dir', help="the directory of images which is used to extract Nearest neighbor",
                        default= None, type=str)
    parser.add_argument('--save_dir', dest='save_dir', help="the directory to use save",
                        default= None, type=str)
    parser.add_argument('--nn_factor', dest='nn_factor', help="nn_factor is the nth neighbors to make db",
                        default= 3, type=int)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args

def count_class(class_set, indices):
    counts = np.zeros(len(class_set))
    for i, cset in enumerate(class_set):
        for idx in indices:
            if idx in cset:
                counts[i] += 1
    return counts

def get_name_from_xml(anno):
    root = anno.getroot()
    obj = root.findall("object")
    name = [x.findtext("name") for x in obj]
    return name
                

def main():
    data = np.load(args.nn_file)
    print(data.shape)
    
    num_target = data.shape[0]
    num_nn = min(args.nn_factor, data.shape[1])
    print("num_target : {}, num_nn : {}".format(num_target, num_nn))
    
    file_list = [f for f in listdir(args.nn_dir) if isfile(join(args.nn_dir, f))]
    file_list = [f for f in file_list if ".png" in str(f) or ".jpg" in str(f)]
    if args.target_dir is not None:
        target_list = [f for f in listdir(args.target_dir) if isfile(join(args.target_dir, f))]
        target_list = [f for f in target_list if ".png" in str(f) or ".jpg" in str(f)]
    
    #Change index array to name array
    nn_name_list = []
    for i in range(num_target):
        names = np.array([target_list[data[i,0]]], dtype=str)
        for j in range(1,num_nn+1):
            names = np.append(names, file_list[data[i,j]])
        nn_name_list.append(names)
    nn_name_list = np.array(nn_name_list)
    print("nn_name_list : ", nn_name_list.shape)
    print(nn_name_list)
    
    nn_anno_dir = args.nn_dir.replace("JPEGImages/", "Annotations/")
    target_anno_dir = args.target_dir.replace("JPEGImages/", "Annotations/")
    print("nn_anno_dir : {}, target_anno_dir : {}".format(nn_anno_dir, target_anno_dir))
    
    class_name = ["bg", "pedestrian", "rider", "car", "truck", "bus", "tsc", "tst", "tsr", "tl_car", "tl_ped"]
    class_string = ["pedestrian", "rider_bicycle", "rider_bike", "bicycle", "bike",    "3-wheels_rider", "3-wheels", "sedan", "van", "pickup_truck", "truck", "mixer_truck", "excavator", "forklift", "ladder_truck", "truck_etc", "vehicle_etc", "box_truck", "trailer", "bus", "ehicle_special", "sitting_person", "ignored", "false_positive", "animal", "bird", "animal_ignored", "ts_circle", "ts_circle_speed", "ts_triangle", "ts_inverted_triangle", "ts_rectangle", "ts_rectangle_speed", "ts_diamonds", "ts_supplementary", "tl_car", "tl_ped", " tl_special", "tl_light_only", "ts_ignored", "tl_ignored", "tstl_ignore", "ts_sup_ignored", "ts_sup_letter", "ts_sup_drawing", "ts_sup_arrow", "ts_sup_zone", "ts_main_zone", "ts_rectangle_arrow", "tl_rear", "ts_rear", "obstacle_bollard_barricade", "obstacle_bollard_cylinder", "obstacle_bollard_marker", "obstacle_cone", "obstacle_drum", "obstacle_cylinder", "obstacle_bollard_special", "obstacle_bollard_stone", "obstacle_bollard_U_shaped", "parking_cylinder", "parking_sign", "parking_stopper_separated", "parking_stopper_bar", "parking_stopper_marble", "parking_special", "parking_lock", "blocking_bar", "blocking_special", "blocking_ignored", "parking_ignored", "obstacle_ignored", "stopline_normal", "stopline_special", "stopline_ignored", "arrow_normal", "arrow_special", "arrow_ignored", "crosswalk_normal", "crosswalk_special", "crosswalk_ignored", "speedbump_normal", "speedbump_special", "speedbump_ignored", "number_speed", "number_parkingzone", "number_special", "number_ignored", "text_normal", "text_parkingzone", "text_special", "text_ignored", "roadmark_triangle", "roadmark_diamond", "roadmark_bicycle", "roadmark_handicapped", "roadmark_pregnant", "roadmark_special", "roadmark_ignored", "rider_bicycle_2", "rider_bike_2", "rider_bicycle_human_body", "wheel_chair", "vehicle_special", "ts_chevron"]
    class_index = [         1,             2,          2, -100000, -100000,              2,  -100000,     3,   3,            3,     4,           4,         4,        4,            4,   -100000,     -100000,         4,       4,   5,        -100000,              1, -100000,              0,      0,    0,              0,         6,               6,           7,                    7,            8,                  8,           8,          -100000,      9,     10,     -100000,       -100000,    -100000,    -100000,     -100000,        -100000,       -100000,        -100000,      -100000,     -100000,      -100000,            -100000, -100000, -100000,                    -100000,                   -100000,                 -100000,        -100000,      -100000,          -100000,                   -100000,                -100000,                   -100000,          -100000,      -100000,                   -100000,             -100000,                -100000,         -100000,      -100000,      -100000,          -100000,          -100000,         -100000,          -100000,         -100000,         -100000,           -100000,      -100000,       -100000,       -100000,          -100000,           -100000,           -100000,          -100000,           -100000,           -100000,      -100000,            -100000,        -100000,        -100000,     -100000,          -100000,      -100000,      -100000,           -100000,          -100000,          -100000,              -100000,           -100000,          -100000,          -100000,         -100000,      -100000,                  -100000,    -100000,          -100000,    -100000]

    set5_name = ["Ped", "Rider" ,"Car", "TS", "TL"]
    class_set5 = [[1],[2],[3,4,5],[6,7,8],[9,10]]
    error_set5 = [list(), list(), list(),list(),list()]
    error_abs_set5 = [list(),list(), list(),list(),list()]
    total_counts_set5 = [list(),list(), list(),list(),list()]
    for i in range(num_target):
        if i % 1000 == 0:
            print("process {}/{}".format(i,num_target))
        anno_t = parse(target_anno_dir + nn_name_list[i,0].replace(".png",".xml"))
        name_t = get_name_from_xml(anno_t)
        #print(nn_name_list[i,0], len(name_t), name_t)
        indices_t = [class_index[class_string.index(k)] for k in name_t if k in class_string if class_index[class_string.index(k)] > 0 ]
        #print(indices_t)
        counts_t = count_class(class_set5, indices_t)
        #print("counts_t : " ,counts_t)
        for j in range(1,num_nn+1):
            anno_n_name = nn_anno_dir + nn_name_list[i,j].replace(".jpg",".xml")
            if os.path.exists(anno_n_name) is False:
                continue 
            anno_n = parse(nn_anno_dir + nn_name_list[i,j].replace(".jpg",".xml"))
            name_n = get_name_from_xml(anno_n)
            indices_n = [class_index[class_string.index(k)] for k in name_n if k in class_string if class_index[class_string.index(k)] > 0 ]
            counts_n = count_class(class_set5, indices_n)

            error_counts = counts_t - counts_n
            for k, e in enumerate(error_counts):
                error_set5[k].append(e)
                error_abs_set5[k].append(abs(e))
                
            for k, c in enumerate(counts_n):
                total_counts_set5[k].append(c)
    
    print("error_set5 : ", np.array(error_set5).shape)
    print("error_abs_set5 : ", np.array(error_abs_set5).shape)
    
    for i, e in enumerate(error_set5):
        plt.clf()
        plt.hist(e, bins=30, label=set5_name[i])
        plt.savefig("./hist_error_"+set5_name[i]+"_nn"+str(num_nn)+".png")
    for i, e in enumerate(total_counts_set5):
        plt.clf()
        plt.hist(e, bins=30, label=set5_name[i])
        plt.savefig("./hist_counts_"+set5_name[i]+"_nn"+str(num_nn)+".png")
        
    for i, e in enumerate(error_abs_set5):
        print(set5_name[i], "'s mean error : ", np.array(e).mean())
            
    
    
    
    
if __name__ == "__main__":
    args = parse_args()
    main()