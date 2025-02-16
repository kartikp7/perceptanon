# Image Assessment Metrics

Code is adapted from [DeepPrivacy](
https://github.com/hukkelas/DeepPrivacy/tree/master/deep_privacy/metrics)

### 1. Generate metric scores for image pairs
Usage: 

```bash
get_metrics.py <input_data_csv> <output_csv_name>
```
Generates an output dataframe with the metrics computer per image pair.
`<input_data_csv>` contains img1 and img2, paths to dataset for original and anonymized images.
`<output_csv_name>` is desired output csv filename containing all computed metrics for all image pairs in dataset.

### 2. Compute correlations 

Usage: 
```bash
metric_corr.py <metric_csv_file> <split_type>
```
Computes the correlations with human scores. `<metric_csv_file>` is output from get_metrics.py which contains the human annotations, and metric scores for all metrics
`<split_type>` determines the test split - all, loov-voc, loov-coco, loov-lfw, loov-celeba, task-face, task-person