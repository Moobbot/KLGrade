"""
Instance-Aware Repeat Factor Sampling (IRFS)

Implements repeat factor sampling to address class imbalance in object detection.
Rare classes get higher repeat factors based on instance counts.

Formula: r_i = max(1, sqrt(t / f_i))
where:
- t = target frequency threshold (default 0.001 = 0.1%)
- f_i = frequency of class i in dataset

Reference:
- Repeat Factor Sampling paper (Facebook AI Research)
- Applied to knee OA detection with severe class imbalance (45:1 ratio)
"""

import math
from collections import defaultdict
from torch.utils.data import Sampler
import torch


class RepeatFactorSampler(Sampler):
    """
    Instance-Aware Repeat Factor Sampling for long-tail detection.

    Samples images with higher frequency if they contain rare classes.
    This balances the effective class distribution during training.

    Args:
        dataset: Dataset object with class_labels attribute
        repeat_thresh: Target frequency for rare classes (default: 0.001)
        shuffle: Whether to shuffle indices (default: True)

    Example:
        >>> sampler = RepeatFactorSampler(train_dataset, repeat_thresh=0.001)
        >>> loader = DataLoader(train_dataset, batch_size=16, sampler=sampler)
    """

    def __init__(self, dataset, repeat_thresh=0.001, shuffle=True):
        self.dataset = dataset
        self.repeat_thresh = repeat_thresh
        self.shuffle = shuffle

        # Compute class frequencies
        self.class_freq = self._compute_class_frequencies()

        # Compute repeat factors per image
        self.repeat_factors = self._compute_repeat_factors()

        # Generate repeated indices
        self.indices = self._generate_indices()

        print(f"\n📊 IRFS Statistics:")
        print(f"   Original dataset size: {len(dataset)}")
        print(
            f"   Effective dataset size: {len(self.indices)} ({len(self.indices)/len(dataset):.2f}×)"
        )
        self._print_class_stats()

    def _compute_class_frequencies(self):
        """Count instances per class across all images"""
        freq = defaultdict(int)
        total = 0

        for idx in range(len(self.dataset)):
            try:
                sample = self.dataset[idx]
                for box_class in sample["class_labels"]:
                    freq[int(box_class)] += 1
                    total += 1
            except:
                continue

        # Normalize to [0,1]
        for k in freq:
            freq[k] = freq[k] / total

        return freq

    def _compute_repeat_factors(self):
        """Compute repeat factor for each image based on rarest class"""
        factors = []

        for idx in range(len(self.dataset)):
            try:
                sample = self.dataset[idx]

                # Get max repeat factor among all classes in this image
                max_factor = 1.0
                for box_class in sample["class_labels"]:
                    class_id = int(box_class)
                    class_frac = self.class_freq.get(class_id, 1.0)

                    # Formula: r = max(1, sqrt(t / f))
                    factor = math.sqrt(self.repeat_thresh / max(class_frac, 1e-6))
                    max_factor = max(max_factor, factor)

                factors.append(max_factor)
            except:
                factors.append(1.0)

        return factors

    def _generate_indices(self):
        """Generate repeated indices based on repeat factors"""
        indices = []

        for idx, factor in enumerate(self.repeat_factors):
            # Repeat each image ceiling(factor) times
            repeat_count = math.ceil(factor)
            indices.extend([idx] * repeat_count)

        return indices

    def _print_class_stats(self):
        """Print statistics about class frequencies and repeat factors"""
        print(f"\n   Class Frequencies:")
        for class_id in sorted(self.class_freq.keys()):
            freq = self.class_freq[class_id]
            factor = math.sqrt(self.repeat_thresh / freq)
            print(f"      Class {class_id}: {freq*100:.2f}% → {factor:.2f}× repeat")

    def __iter__(self):
        if self.shuffle:
            # Shuffle the repeated indices
            perm = torch.randperm(len(self.indices)).tolist()
            return iter([self.indices[i] for i in perm])
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


if __name__ == "__main__":
    # Test sampler
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from src.datasets import YoloDataset

    # Load dataset
    dataset = YoloDataset(
        img_dir="processed/knee/images",
        label_dir="processed/knee/labels",
        transform=None,
    )

    # Create sampler
    sampler = RepeatFactorSampler(dataset, repeat_thresh=0.001)

    print(f"\n✅ IRFS Sampler created successfully!")
    print(f"   Dataset length: {len(dataset)}")
    print(f"   Sampler length: {len(sampler)}")
