"""
Script for creating optimal train/test splits for segmentation datasets.

This script analyzes mask images and creates balanced train/test splits that:
1. Ensure all classes appear in both splits (if possible)
2. Maintain similar class ratios between train and test sets
3. Handle rare classes appropriately

The dataset path is hardcoded in the main function.
"""

from pathlib import Path

import cv2
import numpy as np

from petroscope.segmentation.classes import ClassSet, LumenStoneClasses


def calc_single_mask_ratio(
    mask_path: Path,
    classes: ClassSet,
) -> dict[str, int]:
    """
    Calculate the count of each class label in a single mask image.

    Args:
        mask_path (Path): Path to the mask image file.
        classes (ClassSet): An object that provides a mapping from class
        codes to labels.

    Returns:
        dict[str, int]: A dictionary mapping class labels to their
        pixel counts.
    """
    arr = cv2.imread(str(mask_path))
    if arr.ndim == 3:
        arr = arr[:, :, 0]

    values, counts = np.unique(arr, return_counts=True)
    d = {}
    for value, count in zip(values, counts):
        if int(value) in classes.code_to_label:
            label = classes.code_to_label[int(value)]
            d[label] = count
    return d


def create_balanced_split(
    mask_paths: list[Path],
    classes: ClassSet,
    min_test_ratio: float,
    max_test_ratio: float,
    rare_class_threshold: int,
    exclude_from_balance: set[int] = None,
) -> tuple[list[Path], list[Path]]:
    """
    Create an optimal train/test split for segmentation masks that:
    1. Ensures all classes appear in both splits (if possible)
    2. Maintains similar class ratios between train and test
    3. Puts very rare classes (appearing in ≤ rare_class_threshold images)
       in train

    Args:
        mask_paths: List of paths to mask images
        classes: ClassSet object with class mappings
        min_test_ratio: Minimum fraction of images for test set
        max_test_ratio: Maximum fraction of images for test set
        rare_class_threshold: Images with classes appearing ≤ this many times
        go to train
        exclude_from_balance: Set of class codes (int) to exclude from balance
        calculations (default: {0} for background)

    Returns:
        tuple[list[Path], list[Path]]: (train_paths, test_paths)
    """
    if exclude_from_balance is None:
        exclude_from_balance = {0}  # Exclude background by default
    if len(mask_paths) < 4:
        # Too few images for meaningful split
        return mask_paths, []

    print(f"Creating balanced split for {len(mask_paths)} masks...")
    if exclude_from_balance:
        excluded_codes = sorted(exclude_from_balance)
        print(f"Excluding from balance calculation: {excluded_codes}")

    # Step 1: Analyze each mask to get per-image class distributions
    image_class_counts = {}
    class_image_counts = {}  # class -> list of images containing it

    for mask_path in mask_paths:
        class_counts = calc_single_mask_ratio(mask_path, classes)
        image_class_counts[mask_path] = class_counts

        for class_name in class_counts:
            if class_name not in class_image_counts:
                class_image_counts[class_name] = []
            class_image_counts[class_name].append(mask_path)

    # Step 2: Identify rare classes and critical images
    rare_classes = {
        class_name: images
        for class_name, images in class_image_counts.items()
        if len(images) <= rare_class_threshold
    }

    train_paths = []
    test_paths = []
    remaining_paths = list(mask_paths)

    # Step 3: Handle rare classes - put them in train
    for class_name, images in rare_classes.items():
        print(
            f"Rare class '{class_name}' appears in {len(images)} "
            f"image(s) - assigning to train"
        )
        for img_path in images:
            if img_path in remaining_paths:
                train_paths.append(img_path)
                remaining_paths.remove(img_path)

    # Step 4: For remaining images, use greedy assignment
    train_class_counts = {}
    test_class_counts = {}

    # Initialize with rare class counts already in train
    for train_path in train_paths:
        for class_name, count in image_class_counts[train_path].items():
            train_class_counts[class_name] = (
                train_class_counts.get(class_name, 0) + count
            )

    # Step 5: Greedy assignment for remaining images
    # Process diverse images first
    remaining_paths.sort(
        key=lambda x: len(image_class_counts[x]), reverse=True
    )

    for img_path in remaining_paths:
        img_classes = image_class_counts[img_path]

        # Calculate current test ratio
        total_assigned = len(train_paths) + len(test_paths)
        current_test_ratio = len(test_paths) / (total_assigned + 1)

        # Check if we need to ensure classes appear in both sets
        missing_in_test = [
            cls for cls in img_classes if cls not in test_class_counts
        ]
        missing_in_train = [
            cls for cls in img_classes if cls not in train_class_counts
        ]

        # Decision logic
        assign_to_test = False

        if missing_in_train:
            # PRIORITY: This image contains classes not yet in train
            # This is critical - we must have all classes in train for learning
            assign_to_test = False
        elif current_test_ratio < min_test_ratio:
            # Need more test images
            assign_to_test = True
        elif current_test_ratio >= max_test_ratio:
            # Test set is large enough
            assign_to_test = False
        elif missing_in_test:
            # This image contains classes not yet in test
            assign_to_test = True
        else:
            # Calculate balance improvement for both assignments
            train_score = _calculate_balance_score(
                img_classes,
                train_class_counts,
                test_class_counts,
                True,
                exclude_from_balance,
                classes,
            )
            test_score = _calculate_balance_score(
                img_classes,
                train_class_counts,
                test_class_counts,
                False,
                exclude_from_balance,
                classes,
            )
            assign_to_test = test_score < train_score

        # Assign image
        if assign_to_test:
            test_paths.append(img_path)
            for class_name, count in img_classes.items():
                test_class_counts[class_name] = (
                    test_class_counts.get(class_name, 0) + count
                )
        else:
            train_paths.append(img_path)
            for class_name, count in img_classes.items():
                train_class_counts[class_name] = (
                    train_class_counts.get(class_name, 0) + count
                )

    # Step 6: Verify and report
    final_test_ratio = len(test_paths) / len(mask_paths)
    missing_train_classes = [
        cls for cls in class_image_counts if cls not in train_class_counts
    ]
    missing_test_classes = [
        cls for cls in class_image_counts if cls not in test_class_counts
    ]

    print(
        f"Split created: {len(train_paths)} train, {len(test_paths)} test "
        f"(ratio: {final_test_ratio:.2f})"
    )
    if missing_train_classes:
        print(f"WARNING: Classes missing in train: {missing_train_classes}")
    if missing_test_classes:
        print(f"WARNING: Classes missing in test: {missing_test_classes}")

    return train_paths, test_paths


def _calculate_balance_score(
    img_classes: dict[str, int],
    current_train: dict[str, int],
    current_test: dict[str, int],
    assign_to_train: bool,
    exclude_from_balance: set[int],
    classes: ClassSet,
) -> float:
    """Calculate how much the class balance would improve if we assign
    this image to train/test"""
    # Simulate the assignment
    new_train = current_train.copy()
    new_test = current_test.copy()

    if assign_to_train:
        for class_name, count in img_classes.items():
            new_train[class_name] = new_train.get(class_name, 0) + count
    else:
        for class_name, count in img_classes.items():
            new_test[class_name] = new_test.get(class_name, 0) + count

    # Calculate total pixels in each set (excluding excluded classes)
    # Convert class codes back to labels for exclusion check
    excluded_labels = {
        classes.code_to_label.get(code, f"code_{code}")
        for code in exclude_from_balance
        if code in classes.code_to_label
    }

    train_total = sum(
        count
        for class_name, count in new_train.items()
        if class_name not in excluded_labels
    )
    test_total = sum(
        count
        for class_name, count in new_test.items()
        if class_name not in excluded_labels
    )

    if train_total == 0:
        train_total = 1
    if test_total == 0:
        test_total = 1

    # Calculate balance score (lower is better)
    balance_score = 0.0
    all_classes = set(new_train.keys()) | set(new_test.keys())
    # Filter out excluded classes
    all_classes = all_classes - excluded_labels

    for class_name in all_classes:
        train_ratio = new_train.get(class_name, 0) / train_total
        test_ratio = new_test.get(class_name, 0) / test_total
        balance_score += abs(train_ratio - test_ratio)

    return balance_score


def calc_mask_ratio(
    mask_paths: list[Path],
    classes: ClassSet,
) -> tuple[int, dict[str, int]]:
    """
    Calculate the total number of pixels and the count of each class label
    in a list of mask images. Used for statistics display.

    Args:
        mask_paths (list[Path]): A list of paths to mask image files.
        classes (ClassSet): An object that provides a mapping from class
        codes to labels.

    Returns:
        tuple[int, dict[str, int]]: A tuple where the first element is the
        total number of pixels across all mask images, and the second element
        is a dictionary mapping class labels to their respective pixel counts.
    """
    total_pixels = 0
    total_class_counts = {}

    for mask_path in mask_paths:
        # Get class counts for this mask
        class_counts = calc_single_mask_ratio(mask_path, classes)

        # Add pixel count (need to load image to get dimensions)
        arr = cv2.imread(str(mask_path))
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        total_pixels += arr.shape[0] * arr.shape[1]

        # Accumulate class counts
        for class_name, count in class_counts.items():
            total_class_counts[class_name] = (
                total_class_counts.get(class_name, 0) + count
            )

    return total_pixels, total_class_counts


def create_split_from_directory(
    dataset_path: Path,
    classes: ClassSet,
    exclude_from_balance: set[int],
    min_test_ratio: float,
    max_test_ratio: float,
    rare_class_threshold: int,
) -> None:
    """
    Create train/test split for masks in a dataset directory and save the
    split to files.

    Args:
        dataset_path: Path to dataset directory (should contain 'masks' folder)
        classes: ClassSet object with class mappings
        exclude_from_balance: Set of class codes (int) to exclude from balance
        calculations (default: {0} for background)
        min_test_ratio: Minimum fraction of images for test set
        max_test_ratio: Maximum fraction of images for test set
        rare_class_threshold: Images with classes appearing ≤ this many times
        go to train
    """
    masks_dir = dataset_path / "masks"
    if not masks_dir.exists():
        print(f"Error: {masks_dir} does not exist")
        return

    # Get all mask files
    mask_paths = [p for p in masks_dir.rglob("*.png") if p.is_file()]

    if not mask_paths:
        print(f"No PNG mask files found in {masks_dir}")
        return

    print(f"Found {len(mask_paths)} mask files")

    # Create the split
    train_paths, test_paths = create_balanced_split(
        mask_paths,
        classes,
        min_test_ratio=min_test_ratio,
        max_test_ratio=max_test_ratio,
        rare_class_threshold=rare_class_threshold,
        exclude_from_balance=exclude_from_balance,
    )

    # Save split to files
    output_dir = dataset_path / "splits"
    output_dir.mkdir(exist_ok=True)

    train_file = output_dir / "train.txt"
    test_file = output_dir / "test.txt"

    with open(train_file, "w") as f:
        for path in train_paths:
            # Store relative path from dataset root
            rel_path = path.relative_to(dataset_path)
            f.write(str(rel_path) + "\n")

    with open(test_file, "w") as f:
        for path in test_paths:
            # Store relative path from dataset root
            rel_path = path.relative_to(dataset_path)
            f.write(str(rel_path) + "\n")

    print("Split saved to:")
    print(f"  Train: {train_file} ({len(train_paths)} files)")
    print(f"  Test: {test_file} ({len(test_paths)} files)")

    # Print statistics for the split
    print("\n--- Split Statistics ---")

    # Get statistics for both sets
    train_stats = {}
    test_stats = {}

    if train_paths:
        _, train_stats = calc_mask_ratio(train_paths, classes)
    if test_paths:
        _, test_stats = calc_mask_ratio(test_paths, classes)

    # Get all classes that appear in either set, sorted for consistency
    all_classes = sorted(set(train_stats.keys()) | set(test_stats.keys()))

    if train_paths:
        print("Train set:")
        train_total = sum(train_stats.values())
        for class_name in all_classes:
            if class_name in train_stats:
                count = train_stats[class_name]
                ratio = count / train_total * 100
                print(f"  {class_name}: {ratio:.2f}%")

    if test_paths:
        print("Test set:")
        test_total = sum(test_stats.values())
        for class_name in all_classes:
            if class_name in test_stats:
                count = test_stats[class_name]
                ratio = count / test_total * 100
                print(f"  {class_name}: {ratio:.2f}%")


def main():
    """Main function - create split for predefined dataset"""
    # Predefined dataset path
    dataset_path = Path.home() / "dev/LumenStone/S3_v1.5/"

    if not dataset_path.exists():
        print(f"Error: Dataset path {dataset_path} does not exist")
        return

    create_split_from_directory(
        dataset_path,
        LumenStoneClasses.all(),
        exclude_from_balance={0},
        min_test_ratio=0.15,
        max_test_ratio=0.35,
        rare_class_threshold=2,
    )


if __name__ == "__main__":
    main()
