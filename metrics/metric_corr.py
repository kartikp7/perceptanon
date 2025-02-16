import argparse
import pandas as pd
from scipy.stats import pearsonr, spearmanr, kendalltau
# from ..dataset.dataset_splits import *
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from dataset.dataset_splits import *
from utils import *

def get_corr(df, score_col, metric_col):
    '''
        Compute correlations given human annotations (y_true) and metric scores (y_pred)
    '''
    y_true = df[score_col].to_numpy()
    y_pred = df[metric_col].to_numpy()

    # Pearson is not used as it is not rank based, will give unreliable results.
    correlations = {
        'pearson': pearsonr(y_true, y_pred)[0],
        'spearman': spearmanr(y_true, y_pred)[0],
        'kendall': kendalltau(y_true, y_pred)[0]
    }
    return correlations

def get_transformed_correlations(df, label_type, is_classification, metrics):
    '''
        Transform human annotations to class granularity needed (10, 5, 3, 2 class)
    '''
    # Transform the labels
    label_transform = get_label_transforms(label_type, is_classification)
    transformed_labels = df['score_mean'].apply(lambda x: label_transform(x).item() if isinstance(label_transform(x), torch.Tensor) else label_transform(x))

    # Calculate correlations for each metric
    correlations = {}
    for metric in metrics:
        transformed_corr = get_corr(df.assign(transformed_label=transformed_labels), 'transformed_label', metric)
        correlations[metric] = transformed_corr

    return correlations


def print_all_correlations(all_correlations):
    '''
        Display results
    '''
    for label_type, metrics in all_correlations.items():
        print(f"Label Type: {label_type}")
        for metric, correlations in metrics.items():
            print(f"  Metric: {metric}")
            for corr_type, value in correlations.items():
                print(f"    {corr_type.capitalize()} correlation: {value:.4f}")
        print("")


def main():
    parser = argparse.ArgumentParser(description="Calculate correlations between image assessment metrics and human scores.")
    parser.add_argument('--csv_file', type=str, help='Path to the CSV file containing the metrics and scores.')
    parser.add_argument('--split_type', type=str, required=True, choices=['all', 'loov-coco', 'loov-voc', 'loov-lfw', 'loov-celeba', 'task-person', 'task-face'], help='Dataset split experiment type')
    args = parser.parse_args()

    # Get required train/test split -- only test will be used -- df_splits['test']
    df_splits = get_dataset_splits(args.csv_file, split_type=args.split_type)

    metrics = ['PSNR', 'MSE', 'LPIPS', 'SSIM', 'FID']

    # Correlations with human mean score results
    print('\n'+'-'*20)
    print('Mean')
    print('-'*20+'\n')

    score_mean_corr = {}
    for metric in metrics:
        score_mean_corr[metric] = get_corr(df_splits['test'], 'score_mean', metric)
        print(f'{metric}: ', score_mean_corr[metric])

    # Correlations with human score results that are thresholded and normalized (for regression analysis)
    print('\n'+'-'*20)
    print('REG')
    print('-'*20+'\n')

    label_types = ['mean', 'labels10', 'labels5', 'labels3', 'binary']
    is_classification = False  # set to regression, after thresholding, scores will be normalized 0 to 1

    all_correlations = {}
    for label_type in label_types:
        all_correlations[label_type] = get_transformed_correlations(df_splits['test'], label_type, is_classification, metrics)

    print_all_correlations(all_correlations)

    # Correlations with human score results that are thresholded for class granularities
    print('\n'+'-'*20)
    print('CLF')
    print('-'*20+'\n')

    label_types = ['labels10', 'labels5', 'labels3', 'binary']
    is_classification = True  # this is for classification, human scores will be 0-9, 0-4, 0-2, 0-1

    all_correlations = {}
    for label_type in label_types:
        all_correlations[label_type] = get_transformed_correlations(df_splits['test'], label_type, is_classification, metrics)

    print_all_correlations(all_correlations)

if __name__ == '__main__':
    main()
