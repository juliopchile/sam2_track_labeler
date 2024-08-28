import os
import cv2
import numpy as np
from utility_functions import draw_masks_on_image, draw_points

# Specify the path to the main directory, this is the segment-anything-2 path
main_directory = "/home/asdasd/segment-anything-2"

# Change the current working directory to the main directory
os.chdir(main_directory)
from sam2.build_sam import build_sam2_video_predictor

class VideoFrameViewer:
    model_config_dict = {
        'sam2_hiera_base_plus': 'sam2_hiera_b+.yaml',
        'sam2_hiera_large': 'sam2_hiera_l.yaml',
        'sam2_hiera_small': 'sam2_hiera_s.yaml',
        'sam2_hiera_tiny': 'sam2_hiera_t.yaml'
    }

    def __init__(self, video_dir, save_dir, device='cpu', sam2_checkpoint="sam2_hiera_large"):
        self.video_dir = video_dir
        self.device = device
        self.sam2_checkpoint = os.path.join("checkpoints", sam2_checkpoint + ".pt")
        self.model_cfg = VideoFrameViewer.model_config_dict[sam2_checkpoint]
        self.predictor = None
        self.inference_state = None
        
        # Scan all the JPEG frame names in the directory
        frame_names = [
            p for p in os.listdir(video_dir)
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
        ]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
        self.frame_paths = [os.path.join(video_dir, frame_name) for frame_name in frame_names]
        
        # Directory to save masks
        self.save_dir = os.path.join(save_dir, os.path.basename(video_dir))

        # Load all frames into a dictionary with indices as keys
        self.frames = {i: cv2.imread(frame_path, cv2.IMREAD_COLOR) for i, frame_path in enumerate(self.frame_paths)}
        self.frames_state = {i: False for i in range(len(self.frame_paths))}    # True if there are prompts associated to the frame
        self.prompted_frame = {i: cv2.imread(frame_path, cv2.IMREAD_COLOR) for i, frame_path in enumerate(self.frame_paths)}
        self.video_segments = {}  # video_segments contains the per-frame segmentation results
        self.prompts = {}
        self.current_idx = 0
        self.items_id = [0]
        self.current_frame = 0

        # Initialize OpenCV window
        cv2.namedWindow('Frame')

        # Create a trackbar (slider) to navigate through the frames
        cv2.createTrackbar('Frame', 'Frame', 0, len(self.frame_paths) - 1, self.on_trackbar)
        
        # Set the mouse callback function
        cv2.setMouseCallback('Frame', self.on_mouse_click)

    def build_sam2_model(self):
        self.predictor = build_sam2_video_predictor(self.model_cfg, self.sam2_checkpoint, device=self.device)
        self.inference_state = self.predictor.init_state(video_path=self.video_dir)
        self.predictor.reset_state(self.inference_state)

    def display_frame(self):
        """Display the frame at half the size."""
        frame_to_display = self.prompted_frame[self.current_frame] if self.frames_state[self.current_frame] else self.frames[self.current_frame]
        height, width = frame_to_display.shape[:2]
        resized_frame = cv2.resize(frame_to_display, (width // 2, height // 2))
        cv2.imshow('Frame', resized_frame)

    def on_trackbar(self, val):
        """Callback function for the trackbar."""
        self.current_frame = val
        self.display_frame()
    
    def on_mouse_click(self, event, x, y, flags, param):
        """Callback function for mouse events."""
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"Mouse clicked at: ({x}, {y})")
            self.prompt_frame(2*x, 2*y, 1)
        elif event == cv2.EVENT_RBUTTONDOWN:
            print(f"Mouse clicked at: ({x}, {y})")
            self.prompt_frame(2*x, 2*y, 0)

    def run(self):
        """Run the video frame viewer."""
        # Display the first frame
        self.display_frame()

        while True:
            # Wait for a key press for 1 millisecond
            key = cv2.waitKey(1) & 0xFF

            # Check if the key is 'q' for quit
            if key == ord('q'):
                break
            elif key == ord('r'):  # 'r' for reseting all frames states
                self.reset_video_state()
                self.reset_predictor_state()
                print("Reset State")
            elif key == ord('p'):  # 'p' for propagating
                self.propagate_prompts()
            elif key == ord('s'):  # 's' for saving masks
                self.save_masks()
            elif key == ord('a'):  # 'a' for adding a new object
                self.add_item()
            elif key == ord('d'):  # 'd' for deleting the last object
                self.pop_item()
            elif key == ord('j'):  # 'j' for moving to previous object id
                self.prev_item()
            elif key == ord('k'):  # 'k' for moving to next object id
                self.next_item()

        cv2.destroyAllWindows()
    
    def prompt_frame(self, x, y, label):
        # Add a specific prompt to the predictor object
        frame_idx, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
            inference_state=self.inference_state,
            frame_idx=self.current_frame,
            obj_id=self.current_idx,
            points=np.array([[x,y]], dtype=np.float32),
            labels=np.array([label], dtype=np.int32),
        )

        # Print the given prompt and mask result into the frame image
        self.prompted_frame[self.current_frame] = draw_masks_on_image(
            image=self.frames[self.current_frame],
            masks=out_mask_logits,
            obj_idxs=out_obj_ids,
            random_color=False,
            video=True)
        self.prompted_frame[self.current_frame] = draw_points(
            image=self.prompted_frame[self.current_frame],
            coords=[[x,y]], labels=[label])
        
        # Update frame to prompted for future visualization
        self.frames_state[self.current_frame] = True

        # Update shown image
        self.display_frame()
    
    def propagate_prompts(self):
        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(self.inference_state):
            # Save the masks of each frame in a dictionary
            self.video_segments[out_frame_idx] = {out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy() for i, out_obj_id in enumerate(out_obj_ids)}
            
            # Print the given prompt and mask result into the frame image
            self.prompted_frame[out_frame_idx] = draw_masks_on_image(
                image=self.frames[out_frame_idx],
                masks=out_mask_logits,
                obj_idxs=out_obj_ids,
                random_color=False,
                video=True)

            # Update frame to prompted for future visualization
            self.frames_state[out_frame_idx] = True

    def prev_item(self):
        n = len(self.items_id)
        if n == 0:
            self.current_idx = 0
        else:
            self.current_idx = (self.current_idx - 1) % n
        print(f"current_idx: {self.current_idx}")
    
    def next_item(self):
        n = len(self.items_id)
        if n == 0:
            self.current_idx = 0
        else:
            self.current_idx = (self.current_idx + 1) % n
        print(f"current_idx: {self.current_idx}")

    def add_item(self):
        self.items_id.append(len(self.items_id))
        print(f"Objects ID: {self.items_id}")

    def pop_item(self):
        if len(self.items_id) != 0:
            self.items_id.pop()
    
    def reset_video_state(self):
        self.frames_state = {i: False for i in range(len(self.frame_paths))}
        self.prompted_frame = {i: cv2.imread(frame_path, cv2.IMREAD_COLOR) for i, frame_path in enumerate(self.frame_paths)}

    def reset_predictor_state(self):
        self.predictor.reset_state(self.inference_state)

    def save_masks(self):
        # Iterate through each frame and object ID to save the corresponding mask
        for frame in self.video_segments.keys():
            for object_id in self.video_segments[frame].keys():
                # Extract the image, ensuring it's an 8-bit unsigned integer
                image_to_save = self.video_segments[frame][object_id].astype(np.uint8).squeeze()*255
                # Create the path with leading zeros in the frame number
                path_to_save = os.path.join(self.save_dir, f"{object_id}", f"{frame:05d}.png")
                # Save the image using the save_image method
                self.save_image(image_to_save, path_to_save)
                
    @classmethod
    def save_image(cls, image, path):
        # Ensure the directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Save the image using OpenCV
        cv2.imwrite(path, image)