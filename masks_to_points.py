import os
import cv2
import numpy as np
import supervision as sv

def mask_to_polygons(mask_img):
    # Assuming `mask_to_polygons` returns polygons in the format that is compatible with YOLO format
    contours, _ = cv2.findContours(mask_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    polygons = [contour.flatten().tolist() for contour in contours if contour.size >= 6]  # Ensure at least 3 points to form a polygon
    return polygons

def normalize_points(points, width, height):
    normalized = []
    for x, y in zip(points[0::2], points[1::2]):
        normalized.append(x / width)
        normalized.append(y / height)
    return normalized

def mask2poly(mask_path):
    # Load the binary mask in grayscale
    mask_img = cv2.imread(mask_path, flags=cv2.IMREAD_GRAYSCALE)

    # Convert each mask to a polygon
    polygons = mask_to_polygons(mask_img)

    height, width = mask_img.shape[:2]
    normalized_polygons = [normalize_points(polygon, width, height) for polygon in polygons]

    return normalized_polygons

def recorrer_dataset(dataset_path="dataset"):
    labels_dir = os.path.join(dataset_path, 'labels')
    os.makedirs(labels_dir, exist_ok=True)

    for class_index in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_index)
        if not os.path.isdir(class_path) or class_index == 'labels':
            continue

        for frame_mask in os.listdir(class_path):
            frame, extension = os.path.splitext(frame_mask)
            if extension.lower() != '.png':
                continue
            
            mask_path = os.path.join(class_path, frame_mask)
            polygons = mask2poly(mask_path)

            label_path = os.path.join(labels_dir, f"{frame}.txt")
            with open(label_path, 'a') as label_file:
                for polygon in polygons:
                    if polygon:  # Ensure non-empty polygon
                        label_file.write(f"{class_index} {' '.join(map(str, polygon))}\n")


if __name__ == "__main__":
    recorrer_dataset("dataset/SHORT_INCREDIBLE_salmon_run_Underwater_footage_100")