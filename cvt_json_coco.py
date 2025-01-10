import datetime
from PIL import Image
from io import BytesIO
import numpy as np
import cv2

def make_image_meta(image_name, image_array, output_dir):
    height, width = image_array.shape[:2]
    file_name = f"{i:010d}.jpg"
    cv2.imwrite(f"{output_dir}/{file_name}", image_array)
    image_meta = {
        "id": i,
        "width": width,
        "height": height,
        "file_name": file_name,
        "orignal_file_name": image_name
    }
    return image_meta


def make_coco_anns(labels, boxes, segmentations, areas):
    annotations = []
    for label, box, segmentation, area in zip(labels, boxes, segmentations, areas):
        annotation = {}
        category_id = label_2_category[label]
        annotation['category_id'] = category_id
        # COCO 데이터셋의 bbox 형식 (x_min, y_min, width, height)으로 변환
        x_min, y_min, x_max, y_max = box
        width = x_max - x_min
        height = y_max - y_min
        box = [x_min, y_min, width, height]
        annotation['box'] = box

        annotation['segmentation'] = segmentation
        annotation['area'] = area
        annotations.append(annotation)
    return annotations


def create_coco_format(images_meta, annotations, categories, start_annotation_id):
    coco_format = {
        "info": {
            "description": "Example Dataset",
            "url": "http://example.com",
            "version": "1.0",
            "year": 2024,
            "contributor": "Your Name",
            "date_created": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        "licenses": [
            {
                "id": 1,
                "name": "Example License",
                "url": "http://example.com/license"
            }
        ],
        "images": images_meta,
        "annotations": [],
        "categories": categories
    }
    
    annotation_id = start_annotation_id

    for image_meta in images_meta:
        image_id = image_meta["id"]
        print(image_id)
        for ann in annotations[image_id]:
            segmentation = ann["segmentation"]
            bbox = ann["box"]
            area = ann["area"]
            coco_format["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": ann["category_id"],
                "segmentation": [list(np.array(segmentation).flatten())],
                "area": area,
                "bbox": bbox,
                "iscrowd": 0
            })
            annotation_id += 1
    
    return coco_format