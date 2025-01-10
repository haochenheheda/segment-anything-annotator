import json
import os
import cv2
import numpy as np
import io
from PIL import Image
import base64


def stringToRGB(base64_string):
    "문자열을 이미지 RGB로 나타낼 때 "
    imgdata = base64.b64decode(base64_string)
    dataBytesIO = io.BytesIO(imgdata)
    image = Image.open(dataBytesIO)
    return cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB) # 


def points_to_mask(points, img_h, img_w):
    """포인트 데이터를 처리하여 마스크 반환"""
    points_array = np.array(points).reshape((int(len(points)), 2))
    contours = points_array.astype(np.int32)
    
    # 마스크 생성
    mask = np.zeros((img_h, img_w))
    mask = cv2.fillPoly(mask, [contours], 255)

    return mask 

def points_to_bbox(points):
    """
    input : points
    return : x1, y1, x2, y2
    """
    points_array = np.array(points).reshape((int(len(points)), 2))
    poly_xs = points_array[..., 0]
    poly_ys = points_array[..., 1]
    x1, x2 = np.min(poly_xs), np.max(poly_xs)
    y1, y2 = np.min(poly_ys), np.max(poly_ys)
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
    return x1, y1, x2, y2


def get_normalized_cxwyh_bbox(bbox, img_w, img_h):
    """바운딩 박스를 정규화된 cxywh 형식으로 변환"""
    x1, y1, x2, y2 = bbox
    cx = (x2 + x1) / 2
    cy = (y2 + y1) / 2
    w = (x2 - x1)
    h = (y2 - y1)
    # float 형식으로 변환
    cx, cy, w, h = map(float, [cx/img_w, cy/img_h, w/img_w, h/img_h])
    return cx, cy, w, h 


def parse_json(json_path, class_dict):
    """
    json 파일을 읽어서 image, labels, bboxes, masks를 반환하는 함수
    input : json_path
    return : image, labels, bboxes, masks
    image : cv2 image
    labels : list of label (int)
    bboxes : list of bbox (int)
    masks : list of mask (cv2 image)
    """
    
    # load json 
    load_data = json.load(open(json_path, 'r'))

    # get image
    image = stringToRGB(load_data['imageData'])
    img_h, img_w, c = image.shape

    # set return data
    labels, bboxes, masks = [], [], []
    
    # set group data
    group_data = {}

    for shape in load_data['shapes'] :

        # get current_group_id
        current_group_id = shape['group_id']
        
        # group_id가 없으면 건너뛰기
        if current_group_id is None:
            continue
        
        if shape['label'] in ['Object']:
            continue

        # group_id가 처음 등장하면 초기화
        if current_group_id not in group_data:
            group_data[current_group_id] = {
                'points': [],
                'mask': None,
                'label': class_dict[shape['label']]  # 그룹의 label 저장
            }

        # 현재 group_id에 points 추가
        group_data[current_group_id]['points'] += shape['points']

        # 현재 group_id에 mask 추가
        new_mask = points_to_mask(shape['points'], img_h, img_w)
        if group_data[current_group_id]['mask'] is None:
            group_data[current_group_id]['mask'] = new_mask
        else:
            group_data[current_group_id]['mask'] = cv2.bitwise_or(
                group_data[current_group_id]['mask'], 
                new_mask
            )

    # 각 그룹별 처리
    for _, data in group_data.items():

        label = data['label']         
        bbox = points_to_bbox(data['points'])
        mask = data['mask']       
            
        labels.append(label)
        bboxes.append(bbox)
        masks.append(mask)

    return image, labels, bboxes, masks

if __name__ == "__main__":
    json_path = 'output/IMG_9766..json'
    class_dict = {'girl' : 0,
                  'boy' : 1,
                  'mother' : 2,
                  'father' : 3,}
    class_dict_reverse = {v: k for k, v in class_dict.items()}
    file_name = os.path.basename(json_path).split('.')[0]

    yolo_cvt_dir = 'output_/yolo_cvt'
    os.makedirs(yolo_cvt_dir, exist_ok=True)
    masks_dir = os.path.join(yolo_cvt_dir, 'masks')
    images_dir = os.path.join(yolo_cvt_dir, 'images')
    bboxes_dir = os.path.join(yolo_cvt_dir, 'bboxes')

    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(bboxes_dir, exist_ok=True)

    # json 하나당 처리
    image, labels, bboxes, masks = parse_json(json_path, class_dict)


    # # save bbox
    # with open(os.path.join(target_dir, json_name.replace('.json', '.txt')), "w") as f:
    #     f.write(bbox)
    # # save mask
    # cv2.imwrite(
    #     filename=os.path.join(target_dir, json_name.replace('.json', '.jpg')),
    #     img=mask
    # )

    # visualize
    for i, (label, bbox, mask) in enumerate(zip(labels, bboxes, masks)):
        # cx, cy, w, h = bbox.split(' ')[1:]
        image_vis = image.copy()
        image_h, image_w, _ = image.shape
        lab = label 
        x1, y1, x2, y2 = bbox

        image_vis = cv2.rectangle(image_vis,
                                    pt1=(x1, y1),
                                    pt2=(x2, y2),
                                    color=(0, 255, 0),
                                    thickness=2)
        
        # text_size
        text_size = cv2.getTextSize(text=class_dict_reverse[lab],
                                    fontFace=cv2.FONT_HERSHEY_PLAIN,
                                    fontScale=1,
                                    thickness=1)[0]

        # text_rec
        cv2.rectangle(image_vis,
                        pt1=(x1, y1),
                        pt2=(x1 + text_size[0] + 3, y1 + text_size[1] + 4),
                        color=[0, 255, 0],
                        thickness=-1)

        # put text
        cv2.putText(image_vis,
                    text=class_dict_reverse[lab],
                    org=(x1 + 10, y1 + 10),  # must be int
                    fontFace=0,
                    fontScale=0.4,
                    color=(0, 0, 0))

        cv2.imshow("mask", mask)
        cv2.imshow('image', image_vis)
        cv2.waitKey(0)

        #  # save image
        # cv2.imwrite(
        #     filename=os.path.join(yolo_cvt_dir, 'images', file_name + '.jpg'),
        #     img=image
        # )
        # # save bbox
        # with open(os.path.join(yolo_cvt_dir, 'bboxes', file_name + '.txt'), "w") as f:
        #     f.write(bbox)
        
        # # save mask
        # cv2.imwrite(
        #     filename=os.path.join(yolo_cvt_dir, 'masks', file_name + '.jpg'),
        #     img=mask
        # )

