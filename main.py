import torch
from masks_to_points import combine_batches, recorrer_dataset, filtrar_dataset
from manager import Manager
from config import *


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
    #? Utilizar manager para visualizar el dataset y segmentarlo
    video_path = os.path.join(VIDEOS_PATH, VIDEO_NAME)
    save_path = SAVE_PATH_FOR_VIEWER
    batch_size = BATCH_SIZE
    #manager = Manager(video_path, save_path, batch_size, device)
    #manager.start()
    #del manager
    
    #? Combinar los batches de mascaras y convertirlas a poligonos en un dataset formato YOLO-seg
    masks_batches_path = MASK_BATCHES_PATH
    masks_save_path = os.path.join(DATASET_PATH, VIDEO_NAME)
    #combine_batches(masks_batches_path, masks_save_path)
    #recorrer_dataset(masks_save_path)
    
    # ? Filtrar el dataset para eliminar máscaras pequeñas (artefactos)
    original_labels = os.path.join(masks_save_path, "labels")
    filtered_labels = os.path.join(masks_save_path, "labeles_filtered")
    filtrar_dataset(original_labels, filtered_labels, video_path, 1024)