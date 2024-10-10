import os
import shutil

from segmenter import Segmenter
from video_frame_viewer import VideoFrameViewer

class Manager:
    def __init__(self, video_dir, save_dir, batch_size, device):
        self.video_dir = video_dir
        self.save_dir = save_dir
        self.batch_size = batch_size
        self.device = device
        self.video_name = os.path.basename(os.path.normpath(video_dir))
        
        self.frame_names = sorted(
            [p for p in os.listdir(video_dir) if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]],
            key=lambda p: int(os.path.splitext(p)[0])
        )
        
        self.batches_directories = self.create_batches()
        self.current_batch_index = 0

    def create_batches(self):
        """Crea lotes de imágenes."""
        batch_dirs = []
        n = len(self.frame_names)

        video_save_dir = os.path.join(self.save_dir, self.video_name)
        os.makedirs(video_save_dir, exist_ok=True)

        for i in range(0, n, self.batch_size):
            start_idx = max(0, i - 1)
            end_idx = min(i + self.batch_size, n)
            batch = self.frame_names[start_idx:end_idx]

            batch_dir = os.path.join(video_save_dir, f"batch_{i // self.batch_size:05d}")
            os.makedirs(batch_dir, exist_ok=True)
            batch_dirs.append(batch_dir)

            for index, frame_name in enumerate(batch):
                source = os.path.join(self.video_dir, frame_name)
                destination = os.path.join(batch_dir, f"{index:05d}.jpg")
                shutil.copyfile(source, destination)

        return batch_dirs

    def start(self):
        """Manage batch processing and display."""
        should_continue  = False
        current_idx = 0
        idx_list = []
        last_frame_prompts = {}
        current_batch = 0
        
        while self.current_batch_index < len(self.batches_directories):
            batch_dir = self.batches_directories[self.current_batch_index]
            frame_paths = [os.path.join(batch_dir, frame) for frame in sorted(os.listdir(batch_dir))]
            current_batch_name = os.path.basename(self.batches_directories[self.current_batch_index])
            save_dir = os.path.join(self.save_dir, f"{self.video_name}_mask", current_batch_name)

            # Crear Segmenter y viewer
            segmenter = Segmenter(batch_dir, save_dir, self.device)
            
            # Incluir información del batch anterior (excepto para batch 0)
            if should_continue: # Batch 1 en adelante
                viewer = VideoFrameViewer(frame_paths, segmenter, self.batch_size*current_batch-1)
                viewer.use_last_batch_prompts(current_idx, idx_list, last_frame_prompts)
            else:
                viewer = VideoFrameViewer(frame_paths, segmenter, 0)

            # Correr el viewer
            should_continue, current_idx, idx_list, last_frame_prompts = viewer.run(self.next_batch)
            current_batch += 1
            segmenter.reset_state()
            del viewer
            del segmenter

            # Decidir si continuar o no
            if not should_continue:
                break

    def next_batch(self):
        """Advance to the next batch."""
        self.current_batch_index += 1
        return self.current_batch_index < len(self.batches_directories)
