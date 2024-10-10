import os
import cv2
import numpy as np
import torch

from config import SAM2_PATH

# Specify the path to the main directory, this is the segment-anything-2 path
main_directory = SAM2_PATH

# Change the current working directory to the main directory
os.chdir(main_directory)
from sam2.build_sam import build_sam2_video_predictor

class Segmenter:
    model_config_dict = {
        'sam2_hiera_base_plus': 'sam2_hiera_b+.yaml',
        'sam2_hiera_large': 'sam2_hiera_l.yaml',
        'sam2_hiera_small': 'sam2_hiera_s.yaml',
        'sam2_hiera_tiny': 'sam2_hiera_t.yaml'
    }

    def __init__(self, video_dir, save_dir, device='cpu', sam2_checkpoint="sam2_hiera_large"):
        self.video_dir = video_dir
        self.save_dir = save_dir
        self.device = device
        self.sam2_checkpoint = os.path.join("checkpoints", sam2_checkpoint + ".pt")
        self.model_cfg = Segmenter.model_config_dict[sam2_checkpoint]

        # Scan all the JPEG frame names in the directory
        frame_names = [
            p for p in os.listdir(video_dir)
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
        ]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
        self.frame_paths = [os.path.join(video_dir, frame_name) for frame_name in frame_names]

        self.predictor = build_sam2_video_predictor(self.model_cfg, self.sam2_checkpoint, device=self.device)
        self.inference_state = self.predictor.init_state(video_path=self.video_dir)
        self.reset_state()

        self.frame_state_prompt = {i: False for i in range(len(self.frame_paths))}    # True if there are prompts associated to the frame
        self.frame_state_masked = {i: False for i in range(len(self.frame_paths))}    # True if there are masks associated with the frame
        self.prompts = {i: ([],[]) for i in range(len(self.frame_paths))}  # Store prompts per frame
        self.video_segments = {}  # video_segments contains the per-frame segmentation results

    def reset_state(self):
        """Resets the predictor state and stored data."""
        self.predictor.reset_state(self.inference_state)
        self.frame_state_prompt = {i: False for i in range(len(self.frame_paths))}
        self.frame_state_masked = {i: False for i in range(len(self.frame_paths))}
        self.prompts = {i: ([],[]) for i in range(len(self.frame_paths))}
        self.video_segments = {}

    def add_prompt(self, frame_idx, x, y, label, obj_id=0):
        """Add prompt data for a given frame."""
        frame_idx, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
            inference_state=self.inference_state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            points=np.array([[x, y]], dtype=np.float32),
            labels=np.array([label], dtype=np.int32),
        )
        self.prompts[frame_idx][0].append([x,y])
        self.prompts[frame_idx][1].append(label)

        self.video_segments[frame_idx] = out_mask_logits, out_obj_ids
        self.frame_state_prompt[frame_idx] = True
        self.frame_state_masked[frame_idx] = True

    def propagate_prompts(self):
        """Propagate prompts for video frames."""
        for frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(self.inference_state):
            self.video_segments[frame_idx] = out_mask_logits, out_obj_ids
            self.frame_state_masked[frame_idx] = True

    def save_masks(self, offset=0):
        """Save masks for each frame."""
        os.makedirs(self.save_dir, exist_ok=True)
        for frame_idx, frame_segments in self.video_segments.items():
            out_mask_logits, out_obj_ids = frame_segments
            count_object_id = 0
            for object_id in out_obj_ids:
                image_to_save = out_mask_logits[count_object_id] > 0.0
                count_object_id += 1
                image_to_save = image_to_save.cpu().numpy().astype(np.uint8).squeeze()*255
                frame_to_save = frame_idx + offset
                path_to_save = os.path.join(self.save_dir, f"{object_id}", f"{frame_to_save:05d}.png")
                Segmenter.save_image(image_to_save, path_to_save)

    @staticmethod
    def save_image(image, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cv2.imwrite(path, image)

    def get_masks(self, frame_idx):
        """Get masks for a specific frame."""
        return self.video_segments.get(frame_idx, None)

    def get_prompts(self, frame_idx):
        """Get prompts for a specific frame."""
        return self.prompts.get(frame_idx, None)
    
    def __del__(self):
        """Destructor for releasing resources used by the predictor."""
        if hasattr(self.predictor, 'device'):
            # Check if the predictor is using a CUDA device, then release the memory
            if self.predictor.device.type == 'cuda':
                # Deleting the predictor to free up GPU memory
                del self.predictor
                # Optionally, clear the PyTorch cache
                torch.cuda.empty_cache()
                print("Freed up GPU memory.")
        else:
            # This ensures that the predictor is safely deleted in non-GPU scenarios as well
            del self.predictor
            print("Freed up predictor resources.")