import os

#? SAM2
# Specify the path to the main directory, this is the segment-anything-2 path
SAM2_PATH = "/home/asdasd/segment-anything-2"

#? Video tracking with SAM2
# Path where the videos are saved as jpeg images
VIDEOS_PATH = "/home/asdasd/sam2_track_labeler/videos"
# Name of the video to track
VIDEO_NAME = "SHORT_INCREDIBLE_salmon_run_Underwater_footage"
# Where to save the images for the Viewer Class to save the batches (temp files)
SAVE_PATH_FOR_VIEWER = "/home/asdasd/sam2_track_labeler/saves"
# Number of frames to process per batch
BATCH_SIZE = 50
# Path where the batches mask are saved, used to combine batches
MASK_BATCHES_PATH = os.path.join(SAVE_PATH_FOR_VIEWER, VIDEO_NAME + "_mask")

# Path where the labeles are saved
DATASET_PATH = "/home/asdasd/sam2_track_labeler/dataset"
