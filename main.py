from video_frame_viewer import VideoFrameViewer
import torch


if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )


if __name__ == "__main__":
    # Directory containing the JPEG frames
    video_dir = "/home/asdasd/sam2_track_labeler/videos/SHORT_INCREDIBLE_salmon_run_Underwater_footage_100"
    save_dir = "/home/asdasd/sam2_track_labeler/dataset"

    viewer = VideoFrameViewer(video_dir, save_dir, device)
    viewer.build_sam2_model()
    viewer.run()
    