import json
import random
import os

def split_dataset(
    data,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    shuffle=True,
    random_seed=42
):
    """
    Splits the data into train, validation, and test sets according to specified ratios.
    
    :param data: A list of data samples (each sample can be a dict or anything else).
    :param train_ratio: Fraction of samples to include in the train set.
    :param val_ratio: Fraction of samples to include in the validation set.
    :param test_ratio: Fraction of samples to include in the test set.
    :param shuffle: Whether to shuffle the data before splitting.
    :param random_seed: Random seed to ensure reproducibility if shuffle=True.
    :return: (train_data, val_data, test_data) as three lists.
    """

    # Check that the ratios sum to 1 if doing a 3-way split
    # or sum to 1 if val_ratio=0 (meaning 2-way split).
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-8:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    if shuffle:
        random.seed(random_seed)
        random.shuffle(data)

    total_samples = len(data)
    train_end = int(train_ratio * total_samples)
    val_end = train_end + int(val_ratio * total_samples)

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    return train_data, val_data, test_data


def create_dummy_datasets(train_data, val_data, test_data, output_dir):
    """
    Creates dummy datasets by taking the first 64 examples from train, 
    16 from val, and 16 from test.
    
    :param train_data: Original training data list
    :param val_data: Original validation data list
    :param test_data: Original test data list
    :param output_dir: Directory to save the dummy datasets
    """
    # Create dummy data by taking the first N examples
    dummy_train = train_data[:64] if len(train_data) >= 64 else train_data
    dummy_val = val_data[:16] if len(val_data) >= 16 else val_data
    dummy_test = test_data[:16] if len(test_data) >= 16 else test_data
    
    # Create the dummy directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the dummy datasets
    with open(os.path.join(output_dir, "dummy_train_data.json"), "w", encoding="utf-8") as f:
        json.dump(dummy_train, f, ensure_ascii=False, indent=2)
    
    with open(os.path.join(output_dir, "dummy_val_data.json"), "w", encoding="utf-8") as f:
        json.dump(dummy_val, f, ensure_ascii=False, indent=2)
    
    with open(os.path.join(output_dir, "dummy_test_data.json"), "w", encoding="utf-8") as f:
        json.dump(dummy_test, f, ensure_ascii=False, indent=2)
    
    print("\nDummy datasets created!")
    print(f"Dummy train set size: {len(dummy_train)}")
    print(f"Dummy validation set size: {len(dummy_val)}")
    print(f"Dummy test set size: {len(dummy_test)}")


def main():
    # ------------------------
    # 1) Load your data from JSON
    # ------------------------
    input_json_path = "Data_Analysis/final_data_similar_filtered.json"  # <-- Replace with your actual file path
    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)  # data should be a list of samples

    # ------------------------
    # 2) Choose split type and ratios
    #    Example A: 3-way split 80-10-10
    # ------------------------
    train_ratio = 0.80
    val_ratio = 0.1
    test_ratio = 0.1

    # Example B: 2-way split 85-15 (i.e., val_ratio = 0, then test_ratio = 0.15)
    # train_ratio = 0.85
    # val_ratio = 0.0
    # test_ratio = 0.15

    # ------------------------
    # 3) Call the split function
    # ------------------------
    train_data, val_data, test_data = split_dataset(
        data,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        shuffle=True,
        random_seed=42
    )

    # ------------------------
    # 4) Save outputs as JSON
    # ------------------------
    # This step is optional – you can keep them in memory for your training pipeline,
    # or write them to separate files. Below is how to write them to disk.
    
    # Get the directory where the original data will be saved
    output_dir = "Data"
    
    with open(os.path.join(output_dir, "train_data.json"), "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)

    # If val_data is non-empty, you can save it
    if len(val_data) > 0:
        with open(os.path.join(output_dir, "val_data.json"), "w", encoding="utf-8") as f:
            json.dump(val_data, f, ensure_ascii=False, indent=2)

    with open(os.path.join(output_dir, "test_data.json"), "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)

    print("Data split completed!")
    print(f"Train set size: {len(train_data)}")
    print(f"Validation set size: {len(val_data)}")
    print(f"Test set size: {len(test_data)}")

    # ------------------------
    # 5) Create and save dummy datasets
    # ------------------------
    dummy_output_dir = os.path.join(output_dir, "dummy")
    create_dummy_datasets(train_data, val_data, test_data, dummy_output_dir)


if __name__ == "__main__":
    main()