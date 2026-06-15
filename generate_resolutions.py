from pathlib import Path
from PIL import Image


# -----------------------------
# paths
# -----------------------------

INPUT_FOLDER = (
    "/kaggle/input/datasets/"
    "dummyirl/6isprs/"
)

OUTPUT_FOLDER = (
    "/kaggle/working/"
    "potsdam_rescaled/"
)


# -----------------------------
# collect RGB images
# -----------------------------

image_paths = sorted(
    Path(INPUT_FOLDER).glob(
        "*_RGB.tif"
    )
)

print(
    f"Found "
    f"{len(image_paths)} images"
)


# -----------------------------
# create datasets
# -----------------------------

for img_path in image_paths:

    print(
        f"Processing "
        f"{img_path.name}"
    )

    img = Image.open(
        img_path
    )

    w, h = img.size

    # ------------------
    # 10 cm
    # ------------------

    img_10 = img.resize(
        (w // 2, h // 2),
        Image.Resampling.BILINEAR
    )

    img_10_up = img_10.resize(
        (w, h),
        Image.Resampling.BILINEAR
    )

    Path(
        OUTPUT_FOLDER,
        "10cm_small"
    ).mkdir(
        parents=True,
        exist_ok=True
    )

    Path(
        OUTPUT_FOLDER,
        "10cm_upsampled"
    ).mkdir(
        parents=True,
        exist_ok=True
    )

    img_10.save(
        Path(
            OUTPUT_FOLDER,
            "10cm_small",
            img_path.name
        )
    )

    img_10_up.save(
        Path(
            OUTPUT_FOLDER,
            "10cm_upsampled",
            img_path.name
        )
    )

    # ------------------
    # 20 cm
    # ------------------

    img_20 = img.resize(
        (w // 4, h // 4),
        Image.Resampling.BILINEAR
    )

    img_20_up = img_20.resize(
        (w, h),
        Image.Resampling.BILINEAR
    )

    Path(
        OUTPUT_FOLDER,
        "20cm_small"
    ).mkdir(
        parents=True,
        exist_ok=True
    )

    Path(
        OUTPUT_FOLDER,
        "20cm_upsampled"
    ).mkdir(
        parents=True,
        exist_ok=True
    )

    img_20.save(
        Path(
            OUTPUT_FOLDER,
            "20cm_small",
            img_path.name
        )
    )

    img_20_up.save(
        Path(
            OUTPUT_FOLDER,
            "20cm_upsampled",
            img_path.name
        )
    )

print("DONE")
