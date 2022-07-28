# Copyright (c) StradVision, Inc. and its affiliates. All Rights Reserved

def write_xml(img_shape, predictions, class_name, out_xml_filename):    
    height, width = img_shape[:2]
    output_path = out_xml_filename
    xml_strs = ['<annotation>\n<size><width>{}</width>\n<height>{}</height></size>'.format(width, height)]
    anns = predictions['instances']
    for i in range(len(anns)):
        bbox = anns[i].pred_boxes.tensor.cpu().numpy()
        x1 = bbox[0, 0]
        y1 = bbox[0, 1]
        x2 = bbox[0, 2]
        y2 = bbox[0, 3]        
        pred_label = class_name[anns[i].pred_classes.cpu().numpy()[0]]
        str_obj = '<object><name>{}</name><bndbox><xmin>{:.2f}</xmin><ymin>{:.2f}</ymin><xmax>{:.2f}</xmax><ymax>{:.2f}</ymax></bndbox></object>'.format(pred_label, x1, y1, x2, y2)
        xml_strs.append(str_obj)
    xml_strs.append('</annotation>\n')
    with open(output_path, 'w') as f:
        f.write('\n'.join(xml_strs))


def write_xml2(img_shape, det_result, out_xml_filename):    
    height, width = img_shape[:2]
    output_path = out_xml_filename
    xml_strs = ['<annotation>\n<size><width>{}</width>\n<height>{}</height></size>'.format(width, height)]
    for obj in det_result:
        bbox = obj["bbox"]
        x1 = bbox[0]
        y1 = bbox[1]
        x2 = bbox[2]
        y2 = bbox[3]        
        pred_label = obj["class_name"]
        str_obj = '<object><name>{}</name><bndbox><xmin>{:.2f}</xmin><ymin>{:.2f}</ymin><xmax>{:.2f}</xmax><ymax>{:.2f}</ymax></bndbox></object>'.format(pred_label, x1, y1, x2, y2)
        xml_strs.append(str_obj)
    xml_strs.append('</annotation>\n')
    with open(output_path, 'w') as f:
        f.write('\n'.join(xml_strs))