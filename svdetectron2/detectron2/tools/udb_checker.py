import os
import xml.etree.ElementTree as ET
import cv2



if __name__ == "__main__":
    dataset_root_path = '/data1/yjkim/udb_god_sod/godsod_all'

    # dataset_dir = [file for file in os.listdir(dataset_root_path) if os.path.isdir(os.path.join(dataset_root_path, file))]
    dataset_dir = ['Annotations']
    
    classset = []

    classset_sod = [            
        'obstacle_cone',        
        'obstacle_bollard_stone',
        'obstacle_bollard_cylinder',
        'obstacle_bollard_barricade',
        'obstacle_bollard_marker',
        'obstacle_bollard_U_shaped',
        'obstacle_bollard_special',
        'obstacle_cylinder',
        'obstacle_drum',   
        'obstacle_ignored',
        'parking_sign',
        'parking_stopper_marble',
        'parking_stopper_separated',
        'parking_stopper_bar',
        'parking_lock',
        'parking_cylinder',
        'parking_special',
        'parking_ignored',
        'blocking_bar',
        'blocking_special',
        'blocking_ignored'
    ]

    classset_sod_target = [            
        'obstacle_cone',        
        'obstacle_bollard_stone',
        'obstacle_bollard_cylinder',
        'obstacle_bollard_barricade',
        'obstacle_bollard_marker',
        'obstacle_bollard_U_shaped',
        'obstacle_bollard_special',
        'obstacle_cylinder',
        'obstacle_drum',   
        'obstacle_ignored',
        'parking_sign',
        'parking_stopper_marble',
        'parking_stopper_separated',
        'parking_stopper_bar',
        'parking_lock',
        'parking_cylinder',
        'parking_special',
        'parking_ignored',
        'blocking_bar',
        'blocking_special',
        'blocking_ignored'
    ]

    special_cls = [
        'obstacle_bollard_special',
        'parking_special',
        'blocking_special'
    ]


    all_classes = [
        'sedan', 
        'van', 
        'pedestrian', 
        'box_truck', 
        'ignored', 
        'ts_circle', 
        'tl_car', 
        'tl_ignored', 
        'ts_ignored', 
        'ts_rear', 
        'ts_rectangle', 
        'rider_bike', 
        'truck', 
        'pickup_truck', 
        'ts_rectangle_speed', 
        'bus', 
        'ts_diamonds', 
        'ts_sup_ignored', 
        'obstacle_ignored', 
        'tl_rear', 
        'vehicle_etc', 
        'rider_bicycle', 
        'bicycle', 
        'tl_ped', 
        'obstacle_cone', 
        'ehicle_special', 
        'ts_triangle', 
        'ts_sup_letter', 
        'truck_etc', 
        'sitting_person', 
        'ts_sup_arrow', 
        'bird', 
        'animal_ignored', 
        'obstacle_bollard_special', 
        'obstacle_cylinder', 
        'bike', 
        'animal', 
        'trailer', 
        'ts_Inverted_triangle', 
        'tl_light_only', 
        'obstacle_bollard_cylinder', 
        'ts_rectangle_arrow', 
        'parking_ignored', 
        'parking_stopper_marble', 
        'blocking_ignored', 
        'obstacle_bollard_marker',
        'tl_special',
        'excavator',
        'obstacle_bollard_stone',
        'forklift',
        'parking_sign',
        'mixer_truck',
        'parking_cylinder',
        '3-wheels',
        'obstacle_bollard_U_shaped',
        '3-wheels_rider',
        'parking_stopper_bar',
        'ladder_truck', 
        'parking_stopper_separated', 
        'obstacle_drum', 
        'obstacle_bollard_barricade', 
        'ts_sup_drawing', 
        'blocking_bar', 
        'parking_special', 
        'ts_main_zone', 
        'parking_lock', 
        'ts_circle_speed', 
        'blocking_special', 
        'false_positive', 
        'ts_sup_zone', 
        'vehicle_special'
    ]

    for dirname in dataset_dir:
        dataset_path = os.path.join(dataset_root_path, dirname)
        # dataset_path = os.path.join(dataset_path, 'Annotations')
        # dataset_path = os.path.join(dataset_path, 'labels/labels_od_tstld_sod')

        xml_files = os.listdir(dataset_path)
        
        for xml_file in xml_files:
            xml_file_path = os.path.join(dataset_path, xml_file)
            xml_tree = ET.parse(xml_file_path).getroot()
            objects = xml_tree.findall('object')
            
            target_objs = []

            for obj in objects:
                cls_name = obj.findtext('name')
                if cls_name not in classset:
                    classset.append(cls_name)
                
            #     if cls_name in special_cls:                    
            #         target_objs.append(obj)
                    
            # if len(target_objs) != 0:
            #     # open image
            #     jpg_file_path = xml_file_path.replace("Annotations", "JPEGImages")
            #     jpg_file_path = jpg_file_path.replace("xml", "jpg")

            #     im = cv2.imread(jpg_file_path)

            #     for tobj in target_objs:
            #         pt1 = (
            #             int(float(tobj[1].findtext('xmin'))),
            #             int(float(tobj[1].findtext('ymin')))
            #         )
            #         pt2 = (                        
            #             int(float(tobj[1].findtext('xmax'))),
            #             int(float(tobj[1].findtext('ymax')))
            #         )
            #         # draw bbox
            #         text_org = (pt1[0], pt1[1] - 20)
            #         cv2.rectangle(im, pt1, pt2, (255, 0, 0), 2)
            #         cv2.putText(im, tobj.findtext('name'), text_org, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))

            #     target_save_dir = os.path.join(dataset_root_path, dirname)
            #     target_save_dir = os.path.join(target_save_dir, 'JPEGImages_target')
            #     if os.path.isdir(target_save_dir) != True:
            #         os.makedirs(target_save_dir)
                
            #     # save image
            #     save_impath = os.path.join(target_save_dir, xml_file.replace('xml', 'jpg'))
            #     cv2.imwrite(save_impath, im)

        

    print('== final ==')
    print(classset)



