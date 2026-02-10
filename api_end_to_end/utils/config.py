"""
Class name mappings for all KL grade classification types.
"""

# 10-class: KL0-a through KL4-b (detailed structure classification)
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

# 8-class: KL1-a through KL4-b (pathological cases only)
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

# 5-class: KL0 through KL4 (traditional KL grading)
CLASSES_5_CLASS = {
    0: "KL0",
    1: "KL1",
    2: "KL2",
    3: "KL3",
    4: "KL4",
}

# 4-class: KL1 through KL4 (pathological cases, no KL0)
CLASSES_4_CLASS = {
    0: "KL1",
    1: "KL2",
    2: "KL3",
    3: "KL4",
}


def get_class_names(num_classes: int) -> dict:
    """
    Get class name mapping for given number of classes.

    Args:
        num_classes: Number of classes (4, 5, 8, or 10)

    Returns:
        Dictionary mapping class IDs to class names

    Raises:
        ValueError: If num_classes is not supported
    """
    if num_classes == 10:
        return CLASSES_10_CLASS
    elif num_classes == 8:
        return CLASSES_8_CLASS
    elif num_classes == 5:
        return CLASSES_5_CLASS
    elif num_classes == 4:
        return CLASSES_4_CLASS
    else:
        raise ValueError(
            f"Unsupported num_classes: {num_classes}. Must be 4, 5, 8, or 10."
        )


def get_class_list(num_classes: int) -> list:
    """
    Get ordered list of class names for given number of classes.

    Args:
        num_classes: Number of classes (4, 5, 8, or 10)

    Returns:
        List of class names in order
    """
    class_dict = get_class_names(num_classes)
    return [class_dict[i] for i in range(num_classes)]
