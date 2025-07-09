# The Kellgren-Lawrence (KL) scale
CLASSES = {
    0: "KL0",
    1: "KL1",
    2: "KL2",
    3: "KL3",
    4: "KL4",
}

# Image size for resizing all input images (height, width)
IMG_SIZE = 224  # You can change this value as needed

# Detection utility
NUM_CLASSES = len(CLASSES) + 1  # +1 for background

# Training hyperparameters
BATCH_SIZE = 16
EPOCHS = 30
K_FOLDS = 5
