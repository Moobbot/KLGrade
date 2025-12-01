# The Kellgren-Lawrence (KL) scale
CLASSES = {
    0: "KL0",
    1: "KL1",
    2: "KL2",
    3: "KL3",
    4: "KL4",
}

# Extended mapping for tag-style labels (e.g., "3a"/"3b") generated in label_new.
# The base id is kept for compatibility while suffixes describe sub-structures (joint space vs osteophyte).
CLASSES_LABEL_NEW = {
    0: "KL0-a",
    1: "KL0-b",
    2: "KL1-a",
    3: "KL1-b",
    4: "KL2-a",
    5: "KL2-b",
    6: "KL3-a",
    7: "KL3-b",
    8: "KL4-a",
    9: "KL4-b",
}

# Image size for resizing all input images (height, width)
IMG_SIZE = 512  # Default input size for classification and transforms

# Detection utility
NUM_CLASSES = len(CLASSES) + 1  # +1 for background

# Training hyperparameters
BATCH_SIZE = 1
EPOCHS = 30