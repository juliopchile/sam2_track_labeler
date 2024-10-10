import numpy as np
import cv2


# If video False then masks are handled correctly
# Masks are given by the image predictor (for example)
# masks, scores, logits = predictor.predict(
#     point_coords=None,
#     point_labels=None,
#     box=input_boxes,
#     multimask_output=False
# )
# masks.shape # (number of masks, batches (multimask_output), H, W)
# Sometimes the shape is simply (number of masks, H, W) but this is not important because the code handles the other situation

# If video is True then masks need to be preprocesed before.
# Mask are given as logits by the video predictor when adding prompts
# For example:
# frame_idx, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
#     inference_state=self.inference_state,
#     frame_idx=self.current_frame,
#     obj_id=self.current_idx,
#     points=np.array([[x,y]], dtype=np.float32),
#     labels=np.array([label], dtype=np.int32),
# )
# out_mask_logits.shape                         # torch.Size([number of masks, 1, H, W])
# (out_mask_logits > 0.0).cpu().numpy().shape   # (number of masks, 1, H, W)
# 
# This logits are the one the function receives and are an array of False and True values but need to be converted
# to ones and zeros to be managed
def draw_masks_on_image(image, masks, obj_idxs, random_color=False, borders=True, video=True, text=True):
    """
    Draws masks onto the image.
    Parameters:
        image (numpy.ndarray): The original image in RGB format. Shape: (H, W, 3).
        masks (numpy.ndarray): The masks to draw. Shape: (num_masks, batch=1, H, W).
        random_color (bool): If True, use random colors for each mask. Otherwise, use a fixed color.
        borders (bool): If True, draw borders around the masks.
    Returns:
        numpy.ndarray: The image with masks drawn, in RGB format.
    """
    # Make a copy of the image to avoid modifying the original
    output_image = image.copy()
    height, width, _ = output_image.shape
    # Ensure the image is in float format for blending
    output_image = output_image.astype(np.float32)
    if video:
        masks = (masks > 0.0).cpu().numpy().astype(np.uint8).astype(np.float32)
   
    for mask, idx in zip(masks, obj_idxs):
        # Remove the batch dimension
        mask = mask.squeeze(0)  # Now shape: (H, W)
        # Define the color for the mask
        if random_color:
            # Generate a random color
            color = np.random.randint(0, 256, size=(3,), dtype=np.uint8)
        else:
            # Use a unique color per ID
            color = np.array([idx * 37 % 256, idx * 67 % 256, idx * 97 % 256], dtype=np.uint8)
        # Define the transparency factor
        alpha = 0.6
        # Create a colored version of the mask
        colored_mask = np.zeros((height, width, 3), dtype=np.uint8)
        colored_mask[mask == 1] = color
        # Blend the colored mask with the original image
        # Identify the regions where the mask is applied
        mask_indices = mask == 1
        mask_indices_3d = np.repeat(mask_indices[:, :, np.newaxis], 3, axis=2)
        # Apply blending
        output_image[mask_indices_3d] = (alpha * colored_mask[mask_indices_3d] + (1 - alpha) * output_image[mask_indices_3d])
        if borders:
            # Find contours in the mask
            mask_uint8 = (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # Smooth the contours
            smoothed_contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            # Draw the contours on the image
            cv2.drawContours(output_image, smoothed_contours, -1, (255, 255, 255), thickness=2)

        # Calculate the centroid of the mask using image moments
        if text:
            centroid = calculate_centroid(mask_uint8)
            if centroid:
                cX, cY = centroid
                # Add text (ID) at the centroid of the mask
                cv2.putText(output_image, str(idx), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)

    # Convert the image back to uint8 format
    output_image = np.clip(output_image, 0, 255).astype(np.uint8)
    return output_image


def calculate_centroid(mask):
    """Calcula el centroide de una máscara binaria.
    
    Parameters:
        mask (numpy.ndarray): Máscara binaria 2D de la que se calculará el centroide.
    
    Returns:
        (int, int): Coordenadas del centroide (cX, cY).
    """
    M = cv2.moments(mask)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        return cX, cY
    else:
        return None  # Si el área es cero, no hay centroide.
    


def draw_points(image, coords, labels, marker_size=15):
    output_image = image.copy()
    for coord, label in zip(coords, labels):
        color = (0, 255, 0) if label == 1 else (0, 0, 255)
        cv2.drawMarker(output_image, tuple(coord), color, markerType=cv2.MARKER_STAR, markerSize=marker_size, thickness=2, line_type=cv2.LINE_AA)
    return output_image


def draw_boxes(image, boxes, line_color=(0, 255, 0)):
    """
    Draws one or more boxes on the image.

    Parameters:
        image (numpy.ndarray): The original image in RGB format. Shape: (H, W, 3).
        boxes (list or tuple): A list of boxes or a single box. 
                               Each box is a tuple or array of (x0, y0, x1, y1).

    Returns:
        numpy.ndarray: The image with the boxes drawn.
    """
    output_image = image.copy()

    # If boxes is None or empty, return the original image
    if not boxes:
        return output_image

    # Handle single box case
    if isinstance(boxes, (tuple, list)) and len(np.shape(boxes)) == 2:
        boxes = [boxes]

    # Flatten the list if the boxes were provided as an array within a list
    if isinstance(boxes, list) and len(boxes) == 1 and isinstance(boxes[0], np.ndarray):
        boxes = boxes[0]

    # Draw each box on the image
    for box in boxes:
        # Extract coordinates
        if isinstance(box, np.ndarray):
            x0, y0, x1, y1 = box
        else:
            x0, y0, x1, y1 = box[0], box[1], box[2], box[3]
        
        # Draw the rectangle with the given coordinates
        output_image = cv2.rectangle(output_image, (x0, y0), (x1, y1), line_color, thickness=2)

    return output_image
