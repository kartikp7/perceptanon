import torch
from generate_splits import *
import pandas as pd
import random
import os

def get_label_transforms(label_type, is_classification):
    '''
        Define the label transformation function.
        Human mean annotation scores 1-10 (float) are thresholded.
    '''
    if label_type == 'mean':
        return transform_mean
    elif label_type == 'labels10':
        return lambda score: transform_10class(score, is_classification)
    elif label_type == 'labels5':
        return lambda score: transform_5class(score, is_classification)
    elif label_type == 'labels3':
        return lambda score: transform_3class(score, is_classification)
    elif label_type == 'binary':
        return lambda score: transform_binary(score, is_classification)
    else:
        raise ValueError(f'Invalid label_type: {label_type}')

def transform_10class(score, is_classification):
    '''
        Transform to 10 class granularity by rounding.
    '''
    label = round(score) - 1
    if not is_classification:
        label = normalize_value(label, 0, 9)
        return torch.tensor(label, dtype=torch.float32).unsqueeze(0)
    return int(label)
    
def transform_3class(score, is_classification):
    '''
        Transform to 3 class granularity. 
    '''
    if score < 5.0:
        label = 0
    elif score >= 5.0 and score < 8.0:
        label = 1
    else:
        label = 2
    if not is_classification:
        label = normalize_value(label, 0, 2)
        return torch.tensor(label, dtype=torch.float32).unsqueeze(0)
    return int(label)

def transform_5class(score, is_classification):
    '''
        Transform to 5 class granularity. 
    '''
    if score < 3.0:
        label = 0
    elif score >= 3.0 and score < 5.0:
        label = 1
    elif score >= 5.0 and score < 7.0:
        label = 2
    elif score >= 7.0 and score < 9.0:
        label = 3
    else:
        label = 4
    if not is_classification:
        label = normalize_value(label, 0, 4)
        return torch.tensor(label, dtype=torch.float32).unsqueeze(0)
    return int(label)

def transform_binary(score, is_classification):
    '''
        Transform to 2 class (binary) granularity. 
    '''
    if score > 5.0:
        label = 0
    else:
        label = 1
    if not is_classification:
        return torch.tensor(label, dtype=torch.float32).unsqueeze(0)
    return label
    
def transform_mean(score):
    '''
        Normalize mean human scores (values 1-10) to be between 0 and 1 for regression. 
    '''
    label = normalize_value(score, 1, 10)
    return torch.tensor(label, dtype=torch.float32).unsqueeze(0)

def normalize_value(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val)


def get_dataset_splits(anno_path, split_type, dataset_splits_path=None):
    '''
        Get all dataset splits - all, loov, task
    '''
    dataset_name = os.path.splitext(os.path.basename(anno_path))[0]  # Extract 'HA1' or 'HA2'

    # Load from dataset_splits_path if provided
    if dataset_splits_path:
        split_folder = os.path.join(dataset_splits_path, dataset_name, split_type)
        if os.path.exists(split_folder):
            print(f"Loading dataset splits from: {split_folder}")
            return {
                "train": pd.read_csv(os.path.join(split_folder, "train.csv")),
                "val": pd.read_csv(os.path.join(split_folder, "val.csv")),
                "test": pd.read_csv(os.path.join(split_folder, "test.csv"))
            }
        else:
            print(f" No precomputed splits found at {split_folder}. Generating splits instead.")

    # Create splits (seeded so same splits!)
    # Load full dataset
    df = pd.read_csv(anno_path)

    # Generate splits dynamically
    # test_other_df = None
    if split_type == "all":
        train_df, val_df, test_df = split_all(df)
    elif split_type.startswith("loov-"):
        dataset_to_exclude = split_type.split("-")[-1]
        train_df, val_df, test_df = split_loov(df, dataset_to_exclude)
    elif split_type.startswith("task-"):
        train_df, val_df, test_df = split_task(df, split_type.split("-")[-1])
    elif split_type.startswith("anon-"):
        train_df, val_df, test_df = split_anon(df, method_name=split_type.split('-')[-1])
    else:
        raise ValueError(f"Invalid split_type {split_type}.")

    return {"train": train_df, "val": val_df, "test": test_df}

# Negatives function -- Not used in paper!
def make_negative_pairs(dataframe, neg_ratio, seed=42):
    """
    Generate negative pairs for training.

    Args:
        dataframe (pd.DataFrame): Original dataset.
        neg_ratio (int): Number of negatives per sample.
        seed (int, optional): Seed for reproducibility.

    Returns:
        pd.DataFrame: Negative pairs dataset.
    """
    if seed is not None:
        random.seed(seed)

    required_columns = ["img1", "img2", "method", "score_mean"]
    negative_pairs_list = []

    for index, row in dataframe.iterrows():
        method = row["method"]
        original_img = row["img1"]

        # Find candidate negative samples
        candidates = dataframe[(dataframe["method"] == method) & (dataframe["img1"] != original_img)]

        for i in range(min(neg_ratio, len(candidates))):
            if not candidates.empty:
                sample_seed = hash((seed, index, i)) % (2**16)  # Unique seed per sample
                neg_pair = candidates.sample(random_state=sample_seed).iloc[0]

                negative_pairs_list.append({
                    "img1": row["img1"],
                    "img2": neg_pair["img2"],
                    "method": method,
                    "score_mean": 10.0
                })

    return pd.DataFrame(negative_pairs_list, columns=required_columns)