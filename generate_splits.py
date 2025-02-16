import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Constants for dataset splits
TRAIN_SIZE = 0.6
VALID_SIZE = 0.2
TEST_SIZE = 0.2
# Seed for dataset split reproducability
DEFAULT_SEED = 42

def stratify_bin(score):
    '''Custom function for binning since annotation labels 1-4 have very few samples.'''
    if score <= 4:
        return 0  # Bin for 1, 2, 3, 4
    else:
        return score - 4  # Bins for 5, 6, 7, 8, 9, 10

def preprocess_labels(df):
    '''Process dataset labels to generate stratify_score for stratified splits.'''
    df["score_round"] = df["score_mean"].round().astype(int)
    df["stratify_score"] = df["score_round"].apply(stratify_bin)
    df.drop("score_round", axis=1, inplace=True)
    return df

def remove_stratify_score(train_df, val_df, test_df):
    '''Removes the 'stratify_score' column from train, val, and test dataframes.'''
    def drop_column(df):
        if "stratify_score" in df.columns:
            return df.drop(columns=["stratify_score"]).copy()
        return df

    return drop_column(train_df), drop_column(val_df), drop_column(test_df)

def split_all(df, seed=DEFAULT_SEED):
    '''Perform stratified train/val/test split on entire dataset.'''
    train_df, test_df = train_test_split(df, test_size=TEST_SIZE, stratify=df["stratify_score"], random_state=seed)
    train_df, val_df = train_test_split(train_df, test_size=VALID_SIZE/(TRAIN_SIZE+VALID_SIZE), 
                                        stratify=train_df["stratify_score"], random_state=seed)
    return train_df, val_df, test_df

def split_loov(df, dataset_name, seed=DEFAULT_SEED):
    '''Perform leave-one-out-validation (LOOV) split based on dataset name (COCO, VOC, CelebA, LFW).'''
    test_df = df[df["dataset"] == dataset_name]
    train_val_df = df[df["dataset"] != dataset_name]
    train_df, val_df = train_test_split(train_val_df, test_size=VALID_SIZE/(TRAIN_SIZE+VALID_SIZE),
                                        stratify=train_val_df["stratify_score"], random_state=seed)
    return train_df, val_df, test_df

def split_anon(df, method_name, seed=DEFAULT_SEED):
    '''Perform leave-one-out-validation (LOOV) split based on anonymization method (mask, blurpix).'''
    # Separate out the test dataset
    test_df = df[df['method'] == method_name]
    train_val_df = df[df['method'] != method_name]
    # Split the remaining data into train and validation sets
    X_train_val = train_val_df.drop('stratify_score', axis=1)
    y_train_val = train_val_df['stratify_score']
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=VALID_SIZE/(TRAIN_SIZE + VALID_SIZE),
        stratify=y_train_val,
        random_state=seed
    )
    return train_df, val_df, test_df

def split_task(df, task, seed=DEFAULT_SEED):
    '''
    Perform task-based train/val/test split
    - If `task = "person"`, trains on COCO+VOC, ignores CelebA+LFW.
    - If `task = "face"`, trains on CelebA+LFW, ignores COCO+VOC.
    '''
    if task == "person":
        task_datasets = ["coco", "voc"]
    elif task == "face":
        task_datasets = ["celeba", "lfw"]
    else:
        raise ValueError("Task must be 'person' or 'face'")
    # Select rows for the given task
    task_df = df[df["dataset"].isin(task_datasets)]
    # Store everything NOT in the selected task
    # other_task_df = df[~df["dataset"].isin(task_datasets)]
    # Split the task dataset into train, validation, and test sets
    train_df, val_test_df = train_test_split(task_df, test_size=VALID_SIZE + TEST_SIZE,
                                             stratify=task_df["stratify_score"], random_state=seed)
    val_test_df = adjust_low_sample_classes(val_test_df)
    val_df, test_df = train_test_split(val_test_df, test_size=TEST_SIZE/(VALID_SIZE+TEST_SIZE),
                                       stratify=val_test_df["stratify_score"], random_state=seed)
    return train_df, val_df, test_df  

def adjust_low_sample_classes(df):
    '''
    Adjusts `stratify_score` values for any class with ≤ 1 sample by merging it into the next higher class.
    This is done to ensure we can split dataset evenly. the actual mean scores are unchanged.
    '''
    strat_counts = df["stratify_score"].value_counts()
    # Find classes with ≤ 1 sample and merge them into the next class
    for cls in sorted(strat_counts.index):
        if strat_counts[cls] <= 1:
            df.loc[df["stratify_score"] == cls, "stratify_score"] = cls + 1
    return df

def save_splits(base_folder, dataset_name, split_name, train_df, val_df, test_df, test_other_df=None):
    '''Save dataset splits into corresponding CSV files within dataset_splits.'''
    dataset_splits_folder = os.path.join(base_folder, "dataset_splits")
    dataset_output_dir = os.path.join(dataset_splits_folder, dataset_name, split_name)
    os.makedirs(dataset_output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(dataset_output_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(dataset_output_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(dataset_output_dir, "test.csv"), index=False)
    if test_other_df is not None:
        test_other_df.to_csv(os.path.join(dataset_output_dir, "test_other.csv"), index=False)

def find_csv_files(base_folder):
    '''Find HA1.csv and HA2.csv in the base directory.'''
    csv_files = [f for f in os.listdir(base_folder) if f.endswith(".csv") and f in ["HA1.csv", "HA2.csv"]]
    return [os.path.join(base_folder, f) for f in csv_files]

def main():
    parser = argparse.ArgumentParser(description="Generate dataset splits automatically.")
    parser.add_argument("--base_folder", "-f", default='data/', required=True, help="Base folder where HA1.csv and HA2.csv are located")
    args = parser.parse_args()

    input_csvs = find_csv_files(args.base_folder)

    if not input_csvs:
        print("No HA1.csv or HA2.csv files found in the specified folder.")
        return

    split_types = ["all", "task-person", "task-face", "loov-coco", "loov-voc", "loov-lfw", "loov-celeba", "anon-mask", "anon-inpaint", "anon-blurpix"]

    for input_csv in input_csvs:
        df = pd.read_csv(input_csv)
        # Process labels to make new column to do stratified splits
        df = preprocess_labels(df)
        # Get 'HA1' or 'HA2'
        dataset_name = os.path.splitext(os.path.basename(input_csv))[0]
        for split_type in split_types:
            if split_type == "all":
                train_df, val_df, test_df = split_all(df)
            elif split_type.startswith("loov-"):
                dataset_to_exclude = split_type.split("-")[1]
                train_df, val_df, test_df = split_loov(df, dataset_to_exclude)
            elif split_type.startswith("anon-"):
                anon_to_exclude = split_type.split('-')[-1]
                train_df, val_df, test_df = split_anon(df, method_name=anon_to_exclude)
            elif split_type.startswith("task-"):
                task = split_type.split("-")[1]
                train_df, val_df, test_df = split_task(df, task)
            train_df, val_df, test_df = remove_stratify_score(train_df, val_df, test_df)
            save_splits(args.base_folder, dataset_name, split_type, train_df, val_df, test_df)
    print(f"Dataset splits saved in {os.path.join(args.base_folder, 'dataset_splits')}")

if __name__ == "__main__":
    main()