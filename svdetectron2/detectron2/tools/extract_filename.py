import os


if __name__ == "__main__":
    dataset_path = '/data1/yjkim/alt_eval_output_img'

    f = open('./imgfilename.txt', 'w')

    for i in os.listdir(dataset_path):
        if os.path.isdir(dataset_path + '/' + i):
            print(dataset_path + '/' + i)
            for fname in os.listdir(dataset_path + '/' + i):
                if os.path.isfile(dataset_path + '/' + i + '/' + fname):
                    # print(fname)
                    f.write(fname + '\n')

    f.close()
