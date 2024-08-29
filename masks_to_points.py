import os
import cv2
import numpy as np

def mask_to_polygons(mask_img, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE, epsilon=0.0025):
    # Find contours in the binary mask
    contours, _ = cv2.findContours(mask_img, mode, method)
    
    polygons = []
    for contour in contours:
        if contour.size >= 6:  # Ensure at least 3 points to form a polygon
            # Apply contour approximation to reduce the number of points
            approx_contour = cv2.approxPolyDP(contour, epsilon * cv2.arcLength(contour, True), True)
            polygons.append(approx_contour.flatten().tolist())
    
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


def visualize_masks(image_path, mask_labels):
    # Load the image using OpenCV
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image at {image_path}")
        return

    # Read the mask labels file
    with open(mask_labels, 'r') as file:
        lines = file.readlines()

    # Iterate over each line in the mask labels file
    for line in lines:
        parts = line.strip().split()
        class_index = int(parts[0])  # The class index (could be used for different colors)

        # Extract the polygon points from the mask labels
        points = np.array([float(coord) for coord in parts[1:]]).reshape(-1, 2)
        points[:, 0] *= image.shape[1]  # Scale x coordinates to image width
        points[:, 1] *= image.shape[0]  # Scale y coordinates to image height
        points = points.astype(np.int32)

        # Draw the contour on the image
        cv2.drawContours(image, [points], contourIdx=-1, color=(0, 255, 0), thickness=2)

    # Display the image with contours
    cv2.imshow("Image with Contours", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    recorrer_dataset("dataset/SHORT_INCREDIBLE_salmon_run_Underwater_footage_100")
    mask_labels = "dataset/SHORT_INCREDIBLE_salmon_run_Underwater_footage_100/labels/00050.txt" 
    image = "videos/SHORT_INCREDIBLE_salmon_run_Underwater_footage_100/00050.jpg"
    
    visualize_masks(image, mask_labels)