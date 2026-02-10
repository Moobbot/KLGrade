"""
Utility functions for KIOCMIL-CADA API.
"""

# Class name mappings for different model configurations

CLASSES_10_CLASS = {
    0: "KL0-a",  # Osteophyte (gai xương)
    1: "KL0-b",  # Joint space (khe khớp)
    2: "KL1-a",
    3: "KL1-b",
    4: "KL2-a",
    5: "KL2-b",
    6: "KL3-a",
    7: "KL3-b",
    8: "KL4-a",
    9: "KL4-b",
}

CLASSES_8_CLASS = {
    0: "KL1-a",
    1: "KL1-b",
    2: "KL2-a",
    3: "KL2-b",
    4: "KL3-a",
    5: "KL3-b",
    6: "KL4-a",
    7: "KL4-b",
}

CLASSES_5_CLASS = {
    0: "KL0",
    1: "KL1",
    2: "KL2",
    3: "KL3",
    4: "KL4",
}

CLASSES_4_CLASS = {
    0: "KL1",
    1: "KL2",
    2: "KL3",
    3: "KL4",
}


def get_class_names(num_classes: int):
    """Get class names for a given number of classes."""
    if num_classes == 10:
        return CLASSES_10_CLASS
    elif num_classes == 8:
        return CLASSES_8_CLASS
    elif num_classes == 5:
        return CLASSES_5_CLASS
    elif num_classes == 4:
        return CLASSES_4_CLASS
    else:
        raise ValueError(f"Unsupported number of classes: {num_classes}")


def get_class_list(num_classes: int):
    """Get ordered list of class names."""
    class_dict = get_class_names(num_classes)
    return [class_dict[i] for i in range(num_classes)]
