import metric_api as ma
import numpy as np
from PIL import Image
import pandas as pd
import os
from tqdm import tqdm
import argparse

#? Note: run from inside metrics/ folder

def image_to_numpy(image_path, root='../data'):
    '''
        Convert images to numpy format needed by metric_api.py
    '''
    image_path = os.path.join(root, image_path)
    # Open the image file
    image = Image.open(image_path).convert('RGB')  # Ensure image is in RGB
    # Convert it to a numpy array and ensure the type is float32
    image_np = np.array(image, dtype=np.float32)
    # Normalize pixel values to [0, 1]
    image_np /= 255.0
    # Add an extra dimension to the start of the array
    # shape of (1, height, width, channels)
    image_np = np.expand_dims(image_np, axis=0)
    return image_np

def compute_metrics_for_csv(csv_path, output_path):
    '''
        For each image pair, compute metric and store result in dataframe under metric name column heading
    '''
    df = pd.read_csv(csv_path)
    
    metrics_cols = ["MSE", "PSNR", "SSIM", "LPIPS", "FID"]
    for metric in metrics_cols:
        df[metric] = 0.0
    
    # for index, row in df.iterrows():
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        img1 = image_to_numpy(row['img1'])
        img2 = image_to_numpy(row['img2'])

        metrics = ma.compute_all_metrics(img1, img2)
        for metric, value in metrics.items():
            df.at[index, metric] = value
            
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    # Initialize the argparse object
    parser = argparse.ArgumentParser(description='Compute metrics from input CSV and save to output CSV.')
    # Add arguments for input and output CSV file paths
    parser.add_argument('input_csv', type=str, help='The file path for the input CSV.')
    parser.add_argument('output_csv', type=str, help='The file path for the output CSV.')

    # Parse the arguments
    args = parser.parse_args()

    # Call the function with the provided arguments
    compute_metrics_for_csv(args.input_csv, args.output_csv)