import os
import shutil
import cv2
import numpy as np

from numpy.typing import NDArray
from typing import Any

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

# Función para obtener las dimensiones de una imagen usando cv2
def get_image_dimensions(image_path):
    img = cv2.imread(image_path)  # Cargar la imagen
    height, width = img.shape[:2]  # Obtener las dimensiones (alto, ancho)
    return width, height


def filtrar_dataset(original_labels_path, output_labels_path, original_image_directory, max_size=3200):
     # Asegurarse de ordenar los archivos por nombre
    label_files = sorted([os.path.join(original_labels_path, file) for file in os.listdir(original_labels_path)])
    image_files = sorted([os.path.join(original_image_directory, file) for file in os.listdir(original_image_directory)])

    # Recorrer cada archivo con etiqutas y cada imagen, la imagen puede ser mapa de profundidad o imagen original
    # solo se utiliza para calcular las dimensiones de la imagen
    for label_file, image_file in zip(label_files, image_files):
        frame = os.path.splitext(os.path.basename(label_file))[0]
        dimx, dimy = get_image_dimensions(image_file)

        new_id_list, new_labels = filter_polygons(label_file, dimx=dimx, dimy=dimy, max_size=max_size)
        save_polygons_to_file(new_id_list, new_labels, output_labels_path, frame=frame)


# Función para guardar una lista de polígonos como labels
def save_polygons_to_file(idx_list, polygons_list, save_path, frame=0):
    save_path = os.path.normpath(save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Asegurar que el frame tenga 5 dígitos con ceros a la izquierda
    frame_str = f"{int(frame):05d}"

    label_path = os.path.join(save_path, f"{frame_str}.txt")

    with open(label_path, 'w') as label_file:
        for class_index, polygon in zip(idx_list, polygons_list):
            if len(polygon) > 0:  # Asegurar que el polígono no esté vacío
                label_file.write(f"{class_index} {' '.join(map(str, polygon.flatten()))}\n")


def filter_polygons(mask_labels:str, dimx, dimy, max_size):
    nuevos_labels = []
    nueva_id_list = []
    idx_list, polygons = get_polygons_from_file(mask_labels)
    max_threshold = max_size*(dimx*dimy)/409600
    
    for idx, polygon in zip(idx_list, polygons):
        mask = mask_from_polygon(polygon, dimx, dimy)
        size = count_pixels(mask)
        
        # Si el poligono es mayor al threshold, guardarlo
        if size > max_threshold:
            nuevos_labels.append(polygon)
            nueva_id_list.append(idx)
    
    return nueva_id_list, nuevos_labels


# Función para obtener los poligonos de un archivo de etiqutas
def get_polygons_from_file(mask_labels: str) -> tuple[list[int] ,list[NDArray[Any]]]:
    polygons = []
    obj_ids = []
    
    # Read the mask labels file
    with open(mask_labels, 'r') as file:
        lines = file.readlines()

    for line in lines:
        parts = line.strip().split()
        obj_id = parts[0]
        points = np.array([float(coord) for coord in parts[1:]]).reshape(-1, 2)
        polygons.append(points)
        obj_ids.append(obj_id)
        

    return (obj_ids, polygons)


# Función para obtener una máscara binaria dado un poligono
def mask_from_polygon(polygon: NDArray[Any], dimx, dimy):
    # Crear una imagen binaria vacía de tamaño (dimy, dimx)
    mask = np.zeros((dimy, dimx), dtype=np.uint8)
    
    # Convertir las coordenadas normalizadas a coordenadas de imagen
    polygon_scaled = np.array([[int(x * dimx), int(y * dimy)] for x, y in polygon], dtype=np.int32)
    
    # Dibujar el polígono en la máscara con valor 1 (blanco)
    cv2.fillPoly(mask, [polygon_scaled], 1)
    
    return mask

# Calcular el tamaño de la máscara en pixeles
def count_pixels(mask: NDArray[Any]):
    return np.sum(mask)

if __name__ == "__main__":
    masks_batches_path = "saves/SHORT_azul_mask"
    masks_save_path = "dataset/SHORT_azul"
    
    combine_batches(masks_batches_path, masks_save_path)
    recorrer_dataset(masks_save_path)
    