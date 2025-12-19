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

# Filtered class mapping (7 classes) - Rare classes removed (<1% threshold)
# Removed: KL0-b (Class 1: 0.32%), KL1-b (Class 3: 0.74%), KL4-b (Class 9: 0.80%)
# Original -> New mapping: {0->0, 2->1, 4->2, 5->3, 6->4, 7->5, 8->6}
CLASSES_FILTERED = {
    0: "KL0-a",  # Original class 0
    1: "KL1-a",  # Original class 2
    2: "KL2-a",  # Original class 4
    3: "KL2-b",  # Original class 5
    4: "KL3-a",  # Original class 6
    5: "KL3-b",  # Original class 7
    6: "KL4-a",  # Original class 8
}

# Mapping from original labels_new class IDs to filtered class IDs
CLASS_REMAP_FILTERED = {
    0: 0,  # KL0-a
    1: None,  # KL0-b (removed)
    2: 1,  # KL1-a
    3: None,  # KL1-b (removed)
    4: 2,  # KL2-a
    5: 3,  # KL2-b
    6: 4,  # KL3-a
    7: 5,  # KL3-b
    8: 6,  # KL4-a
    9: None,  # KL4-b (removed)
}

# Image size for resizing all input images (height, width)
IMG_SIZE = 512  # Default input size for classification and transforms

# Detection utility
NUM_CLASSES = len(CLASSES) + 1  # +1 for background

# Training hyperparameters
BATCH_SIZE = 1
EPOCHS = 30