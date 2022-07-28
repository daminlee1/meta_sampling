import numpy as np
import csv
import PIL
from PIL import Image
import os,sys
from os import listdir, name
from os.path import isfile, join
import cv2 as cv
import argparse, shutil

def parse_args():
    parser = argparse.ArgumentParser(description="argument of topk_reader")
    parser.add_argument('--nn_file', dest='nn_file', help="neighbors_data",
                        default=None, type=str)
    parser.add_argument('--target_dir', dest='target_dir', help="Target image directory",
                        default=None, type=str)
    parser.add_argument('--target_imgset', dest='target_imgset', help="Imageset of Target data",
                        default=None, type=str)
    parser.add_argument('--nn_dir', dest='nn_dir', help="the directory of images which is used to extract Nearest neighbor",
                        default= None, type=str)
    parser.add_argument('--nn_imgset', dest='nn_imgset', help="Imageset of nn data",
                        default= None, type=str)
    parser.add_argument('--save_dir', dest='save_dir', help="the directory to use save",
                        default= None, type=str)
    parser.add_argument('--make_db', dest='make_db', help="the directory to use save",
                        default= False, type=bool)
    parser.add_argument('--nn_factor', dest='nn_factor', help="nn_factor is the nth neighbors to make db",
                        default= 2, type=int)
    parser.add_argument('--od_tstld_file', dest='od_tstld_file', help="od_tstld_file",
                        default= None, type=str)
    parser.add_argument('--save_nn', dest="save_nn", help="save the nearest neighbors of each image",
                        default=False, type=bool)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    print("args:",args)
    return args

def main():
    data = np.load(args.nn_file)
    print(data.shape)

    candidates = np.array([], dtype=np.int64)
    for i,d in enumerate(data):
        if i % 1000 == 0:
            print(i, "th process")
        candidates = np.concatenate((candidates, d[1:args.nn_factor+1]))
    print(candidates.shape)
    unique_cand = np.unique(candidates)
    print(unique_cand.shape)

    if args.od_tstld_file is not None:
        od_tstld = []
        with open(args.od_tstld_file) as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                od_tstld.append(line)
        print("od_tstld_file length : ", len(od_tstld))

    
    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)

    nn_img_data = []
    with open(args.nn_imgset, 'r', encoding='UTF-8') as f:
        img_names = f.readlines()
    for n in img_names:
        n = n.replace("\n","")
        if os.path.exists(args.nn_dir + n + ".jpg"):
            nn_img_data.append(args.nn_dir + n + ".jpg")
        elif os.path.exists(args.nn_dir + n + ".png"):
            nn_img_data.append(args.nn_dir + n + ".png")
    print(len(nn_img_data))

    if args.target_dir is not None:
        target_img_data = []
        with open(args.target_imgset, 'r', encoding='UTF-8') as f:
            img_names = f.readlines()
        for n in img_names:
            n = n.replace("\n","")
            if os.path.exists(args.target_dir + n + ".jpg"):
                target_img_data.append(args.target_dir + n + ".jpg")
            elif os.path.exists(args.target_dir + n + ".png"):
                target_img_data.append(args.target_dir + n + ".png")
        print(len(target_img_data))
    
    if args.make_db is True:
        dir_jpeg = args.save_dir+"/JPEGImages/"
        dir_anno = args.save_dir+"/Annotations/"
        dir_imgset = args.save_dir+"/ImageSets/"
        dirs = [dir_jpeg, dir_anno, dir_imgset]
        for d in dirs:
            if not os.path.isdir(d):
                os.mkdir(d)
            elif os.path.isdir(d):
                print("Already exist {}, so remove it".format(d))
                shutil.rmtree(d, ignore_errors=True)
                os.mkdir(d)

        if os.path.exists(dir_imgset+"scan.txt"):
            print("Already exist scan.txt, so remove it")
            os.remove(dir_imgset+"scan.txt")
        f = open(dir_imgset+"scan.txt", "w")

        db_counts = 0
        for i, cand in enumerate(unique_cand):
            if i % 1000 == 0:
                print("Make nn db - process {}/{} , the number of data made now : {}".format(i, len(unique_cand),db_counts))
            img_name = nn_img_data[cand]
            pure_name = img_name[:img_name.find(".",len(img_name)-5)].replace(args.nn_dir,"")
            anno_file = (img_name.replace("JPEGImages","Annotations"))[:(1+img_name.find(".",len(img_name)-5))]+".xml"
            if args.od_tstld_file is not None:
                if not pure_name in od_tstld:
                    continue
            if os.path.exists(img_name) and os.path.exists(anno_file):
                db_counts += 1
                # shutil.copy(args.nn_dir + img_name, dir_jpeg + img_name)
                # shutil.copy(anno_file, dir_anno + pure_name + ".xml")
                f.writelines(pure_name+"\n")
            else:
                continue
                #print("Not exist file : ", args.nn_dir + img_name)
        f.close()
        print("Total generated data : ", db_counts)
        print("Finish making nn db")
    elif args.save_nn:
        for i in range(data.shape[0]):
            if os.path.exists(args.save_dir+"nn_"+str(i)+".jpg"):
                continue
            if i % 100 == 0:
                print("merge nn - process {}/{}".format(i, data.shape[0]))
            img_src = cv.imread(target_img_data[data[i,0]])
            img_src = cv.resize(img_src,(236,236))
            img_num = args.nn_factor+1
            for j in range(1,img_num):
                img_near = cv.imread(nn_img_data[data[i,j]])
                img_near = cv.resize(img_near,(236,236))
                img_src = cv.hconcat([img_src,img_near])
            cv.imwrite(args.save_dir+"nn_"+str(i)+".jpg", img_src)

    nn_name_list = []
    for i in range(data.shape[0]):
        if i % 100 == 0:
            print("convering name_nn - process {}/{}".format(i, data.shape[0]))
        names = np.array([], dtype=str)
        for j in data[i]:
            n = nn_img_data[j].replace(args.nn_dir,"")
            names = np.append(names,n)
        nn_name_list.append(names)
    
    nn_name_list = np.array(nn_name_list)
    print("nn_name_list shape : ", nn_name_list.shape)
    np.save(args.save_dir+"/topk-val-names.npy", nn_name_list)

if __name__ == "__main__":
    args = parse_args()
    main()
