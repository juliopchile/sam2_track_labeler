import cv2
import numpy as np
from utility_functions import draw_masks_on_image, draw_points, calculate_centroid
from segmenter import Segmenter

class VideoFrameViewer:
    def __init__(self, frame_paths, segmenter: Segmenter, first_frame: int = 0):
        self.frame_paths = frame_paths
        self.total_frames = len(frame_paths)
        self.current_frame = 0              # Debe ir desde 0 a total_frames-1
        self.first_frame = first_frame
        self.last_frame = self.total_frames + self.first_frame - 1
        self.current_idx = 0
        self.items_id = [0]
        self.segmenter = segmenter
        
        cv2.namedWindow('Frame')
        cv2.createTrackbar('Frame', 'Frame', 0, self.total_frames - 1, self.on_trackbar)
        cv2.setMouseCallback('Frame', self.on_mouse_click)

    def display_frame(self):
        """Muestra el frame actual en la ventana de OpenCV."""
        frame_to_display = cv2.imread(self.frame_paths[self.current_frame], cv2.IMREAD_COLOR)

        # Dibujar máscaras y puntos si están disponibles
        masks = self.segmenter.get_masks(self.current_frame)
        prompts = self.segmenter.get_prompts(self.current_frame)

        if prompts:
            frame_to_display = draw_points(
                image=frame_to_display,
                coords=prompts[0],
                labels=prompts[1])
        if masks:
            frame_to_display = draw_masks_on_image(
                image=frame_to_display,
                masks=masks[0],
                obj_idxs=masks[1],
                random_color=False,
                video=True)

        height, width = frame_to_display.shape[:2]
        resized_frame = cv2.resize(frame_to_display, (width // 2, height // 2))
        cv2.imshow('Frame', resized_frame)

    def on_trackbar(self, val):
        """Callback para manejar cambios en el trackbar."""
        self.current_frame = val
        self.display_frame()

    def on_mouse_click(self, event, x, y, flags, param):
        """Callback para eventos del mouse."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.segmenter.add_prompt(self.current_frame, 2*x, 2*y, 1, self.current_idx)
            self.display_frame()
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.segmenter.add_prompt(self.current_frame, 2*x, 2*y, 0, self.current_idx)
            self.display_frame()

    def last_frame_prompts(self):
        items_list = self.items_id
        current_id = self.current_idx
        prompts = self.get_centroids_from_last_frame()
        return current_id, items_list, prompts

    def get_centroids_from_last_frame(self):
        masks_tuple = self.segmenter.get_masks(self.total_frames - 1)
        masks = masks_tuple[0]
        obj_idxs = masks_tuple[1]
        prompts = {}

        if masks_tuple:
            masks = (masks > 0.0).cpu().numpy().astype(np.uint8).astype(np.float32)
            for mask, idx in zip(masks, obj_idxs):
                mask = mask.squeeze(0)  # Now shape: (H, W)
                mask_uint8 = (mask * 255).astype(np.uint8)
                centroid = calculate_centroid(mask_uint8)
                prompts[idx] = centroid
        
        return prompts

    def use_last_batch_prompts(self, current_id, items_id, last_frame_prompts: dict):
        """Utiliza la información del batch anterior guardado por manager para inicializar este batch"""
        self.current_idx = current_id
        self.items_id = items_id
        for idx, centroid in last_frame_prompts.items():
            if centroid is not None:
                self.segmenter.add_prompt(self.current_frame, centroid[0], centroid[1], 1, idx)
                self.display_frame()
            
    def run(self, on_next_batch: bool):
        """Ejecuta el visualizador de frames."""
        self.display_frame()
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                return_sate = (False, 0, [], {})
                break
            elif key == ord('n'):  # Próximo batch
                current_id, items_list, prompts = self.last_frame_prompts()
                return_sate = (on_next_batch(), current_id, items_list, prompts)
                break
            elif key == ord('r'):  # 'r' for reseting all frames states
                self.segmenter.reset_state()
                print("Reset State")
            elif key == ord('p'):  # Propagación de prompts
                self.segmenter.propagate_prompts()
                self.display_frame()
            elif key == ord('s'):  # Guardar máscaras
                self.segmenter.save_masks(offset=self.first_frame)
                print("Masks Saved")
            elif key == ord('a'):  # 'a' for adding a new object
                self.add_item()
                print(f"Objects ID: {self.items_id}")
            elif key == ord('d'):  # 'd' for deleting the last object
                self.pop_item()
                print(f"Objects ID: {self.items_id}")
            elif key == ord('j'):  # 'j' for moving to previous object id
                self.prev_item()
                print(f"Current ID: {self.current_idx}")
            elif key == ord('k'):  # 'k' for moving to next object id
                self.next_item()
                print(f"Current ID: {self.current_idx}")
        cv2.destroyAllWindows()
        return return_sate

    def prev_item(self):
        n = len(self.items_id)
        if n == 0:
            self.current_idx = 0
        else:
            self.current_idx = (self.current_idx - 1) % n
    
    def next_item(self):
        n = len(self.items_id)
        if n == 0:
            self.current_idx = 0
        else:
            self.current_idx = (self.current_idx + 1) % n
        
    def add_item(self):
        self.items_id.append(len(self.items_id))

    def pop_item(self):
        if len(self.items_id) != 0:
            self.items_id.pop()
            print(f"Objects ID: {self.items_id}")
