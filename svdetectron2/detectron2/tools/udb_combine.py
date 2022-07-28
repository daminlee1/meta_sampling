import os
import cv2
import glob
import argparse
import xml.etree.ElementTree as ET
 



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-xml1', required=True)
    parser.add_argument('-xml2', required=True)
    parser.add_argument('-xml3', required=True)
    parser.add_argument('-output', required=True)
    args = parser.parse_args()
    
    xml1_path = args.xml1       # od
    xml2_path = args.xml2       # tstld
    xml3_path = args.xml3       # sod
    output_path = args.output
 

    if not os.path.isdir( output_path ):
        os.makedirs(output_path)
 

    xmls_od     = glob.glob( os.path.join(xml1_path, "*.xml") )
    # xmls_tstld  = glob.glob( os.path.join(xml2_path, "*.xml") )
    # xmls_sod    = glob.glob( os.path.join(xml3_path, "*.xml") )

    count = 0

    for xml_od in xmls_od:
        xml_tstld   = os.path.join( xml2_path, os.path.basename(xml_od) )
        xml_sod     = os.path.join( xml3_path, os.path.basename(xml_od) )
        
        # merge
        xml_one = ET.parse(xml_od)
        root_one = xml_one.getroot()

        if os.path.isfile(xml_tstld):
            xml_two = ET.parse(xml_tstld)
            root_two = xml_two.getroot()
            objs_two = root_two.findall('object')
            root_one.extend(objs_two)

        if os.path.isfile(xml_sod):
            xml_three = ET.parse(xml_sod)
            root_three = xml_three.getroot()
            objs_three = root_three.findall('object')
            root_one.extend(objs_three)
  
        xml_one.write( os.path.join(output_path, os.path.basename(xml_od)) )

        count = count + 1
        print(count)
    print('finished!')
