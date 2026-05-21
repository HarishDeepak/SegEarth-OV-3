# config_local.py

KAGGLE = True

if KAGGLE:

    SAM3_CHECKPOINT = (
        "/kaggle/input/datasets/"
        "dummyirl/sam3-weights/sam3.pt"
    )

    IMAGE_PATH = (
        "/kaggle/input/datasets/"
        "dummyirl/6isprs/"
        "top_potsdam_5_15_RGB.tif"
    )

else:

    SAM3_CHECKPOINT = (
        "weights/sam3/sam3.pt"
    )

    IMAGE_PATH = (
        "resources/oem_koeln_50.tif"
    )