from PIL import Image
from pathlib import Path

import os

# force non-notebook backend
os.environ["MPLBACKEND"] = "Agg"
import time
import torch
import subprocess

import matplotlib.pyplot as plt
from torchvision import transforms
from mmseg.structures import SegDataSample

from segearthov3_segmentor import SegEarthOV3Segmentation
from config_local import *

# create output folder
os.makedirs("output", exist_ok=True)

# image extensions
IMAGE_EXTENSIONS = [
    ".tif",
    ".tiff",
    ".png",
    ".jpg",
    ".jpeg"
]

# find images
image_paths = []

for ext in IMAGE_EXTENSIONS:

    image_paths.extend(
        Path(INPUT_FOLDER).glob(
            f"*{ext}"
        )
    )

# only RGB images
image_paths = [
    p for p in image_paths
    if "_RGB" in p.name
]

image_paths = sorted(
    image_paths
)

print(
    f"Found "
    f"{len(image_paths)} "
    f"RGB images"
)

#prompt
name_list = [
    'Impervious surfaces',
    'Building',
    'Low vegetation',
    'Tree',
    'Car',
    'Clutter/background'
]

with open('./configs/my_name.txt', 'w') as writers:
    for i in range(len(name_list)):
        if i == len(name_list)-1:
            writers.write(name_list[i])
        else:
            writers.write(name_list[i] + '\n')

writers.close()

# create model once
model = SegEarthOV3Segmentation(
    type='SegEarthOV3Segmentation',
    model_type='SAM3',
    classname_path='./configs/my_name.txt',
    prob_thd=0.1,
    confidence_threshold=0.1,
    slide_stride=512,
    slide_crop=512,
)

# loop images
for idx, img_path in enumerate(image_paths, 1):

    start_time = time.time()
   
    print("=" * 50)
    print(
        f"[{idx}/{len(image_paths)}] "
        f"{img_path.name}"
    )

    img = Image.open(img_path)

    img_tensor = transforms.Compose([
        transforms.ToTensor(),
    ])(img).unsqueeze(0).to('cuda')

    data_sample = SegDataSample()

    img_meta = {
        'img_path': str(img_path),
        'ori_shape': img.size[::-1]
    }

    data_sample.set_metainfo(img_meta)

    print("Running prediction...")

    seg_pred = model.predict(
        img_tensor,
        data_samples=[data_sample]
    )

    seg_pred = (
        seg_pred[0]
        .pred_sem_seg
        .data
        .cpu()
        .numpy()
        .squeeze(0)
    )

    fig, ax = plt.subplots(
        1, 2,
        figsize=(12, 6)
    )

    ax[0].imshow(img)
    ax[0].axis('off')

    ax[1].imshow(
        seg_pred,
        cmap='viridis'
    )
    ax[1].axis('off')

    plt.tight_layout()

    output_path = (
        f"output/"
        f"{img_path.stem}_segmented.png"
    )

    plt.savefig(
        output_path,
        bbox_inches='tight'
    )

    plt.close()

    elapsed = (
    time.time()
    - start_time
)

    # GPU usage
    gpu_info = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu="
            "utilization.gpu,"
            "memory.used,"
            "memory.total",
            "--format=csv,noheader,nounits"
        ]
    ).decode().strip()

    print(
        f"Saved: "
        f"{output_path}"
    )

    print(
        f"Time: "
        f"{elapsed:.1f}s"
    )

    print(
        f"GPU stats: "
        f"{gpu_info}"
    )

print("DONE")