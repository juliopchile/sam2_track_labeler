import os
import shutil
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

    # Recorrer cada carpeta con nombre 0, 1, 2, 3, ..., nth class
    for class_index in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_index)
        if not os.path.isdir(class_path) or class_index == 'labels':
            continue

        # Recorrer cada máscara de cada frame asociado a esa clase
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


def combine_batches(input_directory, output_directory):
    # Lista de batches ordenada
    batches = sorted([d for d in os.listdir(input_directory) if os.path.isdir(os.path.join(input_directory, d))])
    
    for batch in batches:
        batch_path = os.path.join(input_directory, batch)
        
        # Lista de clases dentro de cada batch, ordenada
        classes = sorted([d for d in os.listdir(batch_path) if os.path.isdir(os.path.join(batch_path, d))])
        
        for class_name in classes:
            class_path = os.path.join(batch_path, class_name)
            output_class_path = os.path.join(output_directory, class_name)
            
            # Crear la carpeta de clase en el directorio de salida si no existe
            if not os.path.exists(output_class_path):
                os.makedirs(output_class_path)
            
            # Lista de imágenes dentro de la clase
            images = sorted(os.listdir(class_path))
            
            for image in images:
                source_image_path = os.path.join(class_path, image)
                destination_image_path = os.path.join(output_class_path, image)
                
                # Copy files only if they don't exist in the destination directory
                if not os.path.exists(destination_image_path):
                    shutil.copy2(source_image_path, destination_image_path)

if __name__ == "__main__":
    masks_batches_path = "saves/SHORT_azul_mask"
    masks_save_path = "dataset/SHORT_azul"
    
    combine_batches(masks_batches_path, masks_save_path)
    recorrer_dataset(masks_save_path)
    