from PIL import Image
from pathlib import Path

import os
import time
import torch
import subprocess
import numpy as np

# force safe matplotlib backend
os.environ["MPLBACKEND"] = "Agg"

import matplotlib.pyplot as plt
from torchvision import transforms
from mmseg.structures import SegDataSample

from segearthov3_segmentor import (
    SegEarthOV3Segmentation
)

from config_local import *


# --------------------------------------------------
# create output folder
# --------------------------------------------------

os.makedirs(
    "output",
    exist_ok=True
)

# --------------------------------------------------
# image extensions
# --------------------------------------------------

IMAGE_EXTENSIONS = [
    ".tif",
    ".tiff",
    ".png",
    ".jpg",
    ".jpeg"
]

# --------------------------------------------------
# find images
# --------------------------------------------------

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

# --------------------------------------------------
# split workload across GPUs
# --------------------------------------------------

gpu_count = int(
    os.environ.get(
        "TOTAL_GPUS",
        1
    )
)

gpu_id = int(
    os.environ.get(
        "LOCAL_GPU_ID",
        0
    )
)

if gpu_count > 1:

    chunk_size = (
        len(image_paths)
        + gpu_count
        - 1
    ) // gpu_count

    start = (
        gpu_id
        * chunk_size
    )

    end = min(
        start
        + chunk_size,
        len(image_paths)
    )

    image_paths = image_paths[
        start:end
    ]

print(
    f"GPU {gpu_id}: "
    f"{len(image_paths)} images"
)

for p in image_paths:

    print(
        f"  - {p.name}"
    )

# --------------------------------------------------
# prompts
# --------------------------------------------------

name_list = [
    'road, sidewalk, pavement, parking lot, asphalt, concrete, impervious surface',

    'building, rooftop, house, residential building, commercial building',

    'grass, lawn, shrub, bush, low vegetation',

    'tree, forest, tall vegetation, canopy',

    'car, automobile, vehicle',

    'clutter, background, unknown object, miscellaneous'
]

with open(
    './configs/my_name.txt',
    'w'
) as writers:

    for i in range(
        len(name_list)
    ):

        if i == (
            len(name_list)
            - 1
        ):

            writers.write(
                name_list[i]
            )

        else:

            writers.write(
                name_list[i]
                + '\n'
            )

# --------------------------------------------------
# official Potsdam color map
# --------------------------------------------------

COLOR_MAP = np.array([
    [255, 255, 255],  # 0 impervious
    [0, 0, 255],      # 1 building
    [0, 255, 255],    # 2 low vegetation
    [0, 255, 0],      # 3 tree
    [255, 255, 0],    # 4 car
    [255, 0, 0],      # 5 clutter
], dtype=np.uint8)

# --------------------------------------------------
# create model
# --------------------------------------------------

model = (
    SegEarthOV3Segmentation(
        type='SegEarthOV3Segmentation',
        model_type='SAM3',
        classname_path=
        './configs/my_name.txt',
        prob_thd=0.1,
        confidence_threshold=0.05,
        slide_stride=512,
        slide_crop=512,
    )
)

# --------------------------------------------------
# loop images
# --------------------------------------------------

for idx, img_path in enumerate(
    image_paths,
    1
):

    start_time = (
        time.time()
    )

    print("=" * 60)

    print(
        f"[{idx}/"
        f"{len(image_paths)}] "
        f"{img_path.name}"
    )

    # ---------------------------
    # load RGB
    # ---------------------------

    img = Image.open(
        img_path
    )

    img_tensor = (
        transforms.Compose([
            transforms.ToTensor(),
        ])(img)
        .unsqueeze(0)
        .to('cuda')
    )

    # ---------------------------
    # metadata
    # ---------------------------

    data_sample = (
        SegDataSample()
    )

    img_meta = {
        'img_path':
        str(img_path),

        'ori_shape':
        img.size[::-1]
    }

    data_sample.set_metainfo(
        img_meta
    )

    print(
        "Running prediction..."
    )

    # ---------------------------
    # predict
    # ---------------------------

    seg_pred = (
        model.predict(
            img_tensor,
            data_samples=[
                data_sample
            ]
        )
    )

    seg_pred = (
        seg_pred[0]
        .pred_sem_seg
        .data
        .cpu()
        .numpy()
        .squeeze(0)
    )

    # ---------------------------
    # convert to Potsdam colors
    # ---------------------------

    seg_rgb = COLOR_MAP[
        np.clip(
            seg_pred,
            0,
            5
        )
    ]

    # ---------------------------
    # load GT
    # ---------------------------

    gt_path = (
        img_path.parent /
        img_path.name.replace(
            "_RGB.tif",
            "_label_noBoundary.tif"
        )
    )

    gt_img = Image.open(
        gt_path
    )

    # ---------------------------
    # visualize
    # ---------------------------

    fig, ax = plt.subplots(
        1,
        3,
        figsize=(18, 6)
    )

    # RGB
    ax[0].imshow(img)
    ax[0].axis('off')
    ax[0].set_title(
        "RGB Image"
    )

    # prediction
    ax[1].imshow(
        seg_rgb
    )

    ax[1].axis('off')

    ax[1].set_title(
        "Prediction"
    )

    # GT
    ax[2].imshow(
        gt_img
    )

    ax[2].axis('off')

    ax[2].set_title(
        "Ground Truth"
    )

    plt.tight_layout()

    # ---------------------------
    # save image
    # ---------------------------

    output_path = (
        "output/"
        f"{img_path.stem}"
        "_segmented.png"
    )

    plt.savefig(
        output_path,
        bbox_inches='tight'
    )

    # show only first image
    if idx == 1:

        plt.show()

    plt.close()

    elapsed = (
        time.time()
        - start_time
    )

    # ---------------------------
    # GPU usage
    # ---------------------------

    gpu_info = (
        subprocess
        .check_output([
            "nvidia-smi",
            "--query-gpu="
            "utilization.gpu,"
            "memory.used,"
            "memory.total",
            "--format="
            "csv,noheader,"
            "nounits"
        ])
        .decode()
        .strip()
    )

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