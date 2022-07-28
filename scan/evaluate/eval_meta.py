import sys,os
import argparse
from glob import glob
from lxml import etree
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser('make_imageSets')
    parser.add_argument('--jpeg_dir', dest="jpeg_dir", help="directory of JPEG",
                        default=None, type=str)
    parser.add_argument('--anno_dir', dest='anno_dir', help="directory of Annotation",
                        default=None, type=str)
    parser.add_argument('--save_dir', dest="save_dir", help="save directory",
                        default=None, type=str)
    
    if len(sys.argv) == 1:
        print("no arguments error")
        sys.exit(1)
    
    args = parser.parse_args()
    return args

roads = ["city", "highway", "rural", "etc"]
timezones = ["day", "night", "dawn_evening"]
weathers = ["clear", "rainy","heavy-rainy","snow","fog", 'etc']

def timezone_type(timezone_num):
    if 0<= timezone_num <=59:
        return "night"
    elif 60 <= timezone_num <=99:
        return "dawn_evening"
    elif 100 <= timezone_num <= 255:
        return "day"
    else:
        #print("%d(img_idx)udb meta info(time:%d) is wrong -> timezone value must be in 0~255/ or value is none" %(img_idx,timezone_num))
        return False

def weather_type(weather_item):
    weather_all = {'clean_road':"clear",
                   'wet_light_road':"clear",
                   'wet_medium_road':"rainy",
                   'light_reflection_light_road':"rainy", 
                   'light_reflection_medium_road':"rainy",
                   'wiper_light':"rainy",
                   'wet_severe_road':"heavy-rainy",
                   'light_reflection_severe_road':"heavy-rainy",
                   'snow_light_road':"snow", 
                   'snow_medium_road':"snow", 
                   'snow_severe_road':"snow", 
                   'snow_light_sidewalk':"snow",
                   'snow_medium_sidewalk':"snow",
                   'snow_severe_sidewalk':"snow",
                   'fog_light':"fog", 
                   'fog_medium':"fog", 
                   'fog_severe':"fog", 
                   'road_etc':"etc", 
                   'wiper_severe':"etc",
                   'none':'etc'
                   }
    if weather_item in weather_all.keys():
        return weather_all[weather_item]
    else:
        #print("%d(img_idx)udb meta info(weather number: %d) is wrong or value is none" %(img_idx,weather_num))
        return False

def read_meta_info(xml_files):
    keys = ['file_name', 'time_zone', 'weather_item', 'road']
    output = []
    for xml_file in xml_files:
        each_file = dict.fromkeys(keys)
        each_file['file_name'] = xml_file

        with open(xml_file) as fobj:
            xml = fobj.read()
        root = etree.fromstring(xml)
        for appt in root.getchildren():
            if appt.tag == "meta_info":
                for elem in appt.getchildren():
                    text = elem.text
                    
                    if elem.tag == "timezone_item":
                        each_file['time_zone'] = timezone_type(int(text))
                    elif elem.tag == "weather_item":
                        each_file['weather_item'] = weather_type(text)
                        # if each_file['weather_item'] is False:
                        #     print(text)
                    elif elem.tag == "road":
                        each_file['road'] = text
        output.append(each_file)
    return output

def analyze_meta_data(meta_data):
    print("total meta data num : ", len(meta_data))
    
    num_roads = {'city' : 0,
                 'highway' : 0,
                 'rural' : 0,
                 'etc' : 0}
    num_timezones = {'day' : 0,
                     'night' : 0,
                     'dawn_evening' : 0}
    num_weathers = {'clear' : 0,
                    'rainy' : 0,
                    'heavy-rainy' : 0,
                    'snow' : 0,
                    'fog' : 0,
                    'etc' : 0}
    
    for xd in meta_data:
        num_roads[xd['road']] += 1
        num_timezones[xd['time_zone']] += 1
        num_weathers[xd['weather_item']] += 1
    data = {}
    data['roads'] = [num_roads['city'], num_roads['highway'], num_roads['rural'], num_roads['etc'], 0, 0]
    data['timezones'] = [num_timezones['day'], num_timezones['night'], num_timezones['dawn_evening'], 0, 0, 0]
    data['weathers'] = [num_weathers['clear'], num_weathers['rainy'], num_weathers['heavy-rainy'], num_weathers['snow'], num_weathers['fog'], num_weathers['etc']]
    
    df = pd.DataFrame(data)
    print(df)
    
    df.to_csv('./metadata_analysis.csv')

if __name__ == "__main__":
    
    args = parse_args()
    
    if args.jpeg_dir is not None:
        img_list = glob(args.jpeg_dir + "*.jpg")
        img_list_png = glob(args.jpeg_dir + "*.png")
        if len(img_list_png) != 0:
            for i in img_list_png:
                img_list.append(i)
        img_names = [os.path.basename(x) for x in img_list]
        img_names = [ i.replace(".jpg","") for i in img_names]
        img_names = [ i.replace(".png","") for i in img_names]
    elif args.anno_dir is not None:
        anno_list = glob(args.anno_dir + "*.xml")
        img_names = [os.path.basename(x) for x in anno_list]
        img_names = [ i.replace(".xml","") for i in anno_list]
    
    print("read anno_files : ", len(anno_list))
    
    #get meta-info of anno_list
    xml_metadata = read_meta_info(anno_list)
    
    print("total meta data num : ", len(xml_metadata))
    
    num_roads = {'city' : 0,
                 'highway' : 0,
                 'rural' : 0,
                 'etc' : 0}
    num_timezones = {'day' : 0,
                     'night' : 0,
                     'dawn_evening' : 0}
    num_weathers = {'clear' : 0,
                    'rainy' : 0,
                    'heavy-rainy' : 0,
                    'snow' : 0,
                    'fog' : 0,
                    'etc' : 0}
    
    for xd in xml_metadata:
        num_roads[xd['road']] += 1
        num_timezones[xd['time_zone']] += 1
        num_weathers[xd['weather_item']] += 1
    data = {}
    data['roads'] = [num_roads['city'], num_roads['highway'], num_roads['rural'], num_roads['etc'], 0, 0]
    data['timezones'] = [num_timezones['day'], num_timezones['night'], num_timezones['dawn_evening'], 0, 0, 0]
    data['weathers'] = [num_weathers['clear'], num_weathers['rainy'], num_weathers['heavy-rainy'], num_weathers['snow'], num_weathers['fog'], num_weathers['etc']]
    
    df = pd.DataFrame(data)
    print(df)
    
    df.to_csv('./metadata_analysis.csv')
