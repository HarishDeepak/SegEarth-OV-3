# config_local.py

import os

# --------------------------------------------------
# environment
# --------------------------------------------------

KAGGLE = (
    "KAGGLE_KERNEL_RUN_TYPE"
    in os.environ
)

# --------------------------------------------------
# inference mode
# --------------------------------------------------

RUN_SINGLE_IMAGE = True

# used only if
# RUN_SINGLE_IMAGE = True

TARGET_IMAGE = (
    "top_potsdam_5_14_RGB.tif"
)

# --------------------------------------------------
# paths
# --------------------------------------------------

if KAGGLE:

    SAM3_CHECKPOINT = (
        "/kaggle/input/datasets/"
        "dummyirl/sam3-weights/"
        "sam3.pt"
    )

    INPUT_FOLDER = (
        "/kaggle/input/datasets/"
        "dummyirl/"
        "potsdam-rescaled/"
        "potsdam_rescaled/"
        "10cm_small/"
    )

else:

    SAM3_CHECKPOINT = (
        "weights/sam3/"
        "sam3.pt"
    )

    INPUT_FOLDER = (
        "resources/"
    )