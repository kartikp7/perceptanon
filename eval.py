import numpy as np
# from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, accuracy_score
from scipy.stats import pearsonr, spearmanr, kendalltau
import torch

def calculate_correlations(y_true, y_pred, method='spearman'):
    '''
        Compute correlations.
        Inputs: y_true (human annotation scores), y_pred (PerceptAnon model predictions)
    '''
    if method == 'pearson':
        correlation, _ = pearsonr(y_true, y_pred)
    elif method == 'spearman':
        correlation, _ = spearmanr(y_true, y_pred)
    elif method == 'kendall':
        correlation, _ = kendalltau(y_true, y_pred)
    else:
        raise ValueError(f"Unknown correlation method: {method}")
    
    return correlation

def evaluate_regression(model, test_loader, device, ha_mode="HA1"):
    '''
        Get MSE, predictions, labels (human annotations)
    '''
    model.eval()
    total_loss = 0.0
    total_count = 0
    predictions, labels = [], []
    
    with torch.no_grad():
        for batch in test_loader:
            if ha_mode == "HA1":
                _, inputs, targets = batch  # HA1 uses single anonymized images
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
            else:
                imgs1, imgs2, targets = batch  # HA2 uses original-anonymized pairs
                imgs1, imgs2, targets = imgs1.to(device), imgs2.to(device), targets.to(device)
                outputs = model(imgs1, imgs2)
            loss = torch.nn.functional.mse_loss(outputs.view(-1), targets.view(-1), reduction='sum')
            total_loss += loss.item()
            total_count += targets.size(0)
            predictions.extend(outputs.view(-1).cpu().numpy())
            labels.extend(targets.view(-1).cpu().numpy())

    mse = total_loss / total_count
    return mse, np.array(predictions), np.array(labels)

def evaluate_classification(model, test_loader, device, ha_mode="HA1"):
    '''
        Get Accuracy, predictions, labels (human annotations)
    '''
    model.eval()
    correct = 0
    total = 0
    predictions, labels = [], []
    
    with torch.no_grad():
        for batch in test_loader:
            if ha_mode == "HA1":
                _, inputs, targets = batch  # HA1 uses single anonymized images
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
            else:
                imgs1, imgs2, targets = batch  # HA2 uses original-anonymized pairs
                imgs1, imgs2, targets = imgs1.to(device), imgs2.to(device), targets.to(device)
                outputs = model(imgs1, imgs2)
            _, preds = torch.max(outputs, 1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            predictions.extend(preds.view(-1).cpu().numpy())
            labels.extend(targets.view(-1).cpu().numpy())

    accuracy = correct / total
    return accuracy, np.array(predictions), np.array(labels)

def evaluate_model(model, dataloader, device, is_classification, ha_mode="HA1", label_names=None):
    '''
        Evaluate model for both HA1 and HA2 in regression or classification tasks
    '''
    if is_classification:
        acc, predictions, true_labels = evaluate_classification(model, dataloader, device, ha_mode)
        acc *= 100
        precision = precision_score(true_labels, predictions, average='macro')
        recall = recall_score(true_labels, predictions, average='macro')
        # conf_matrix = confusion_matrix(true_labels, predictions, labels=label_names)
        # clf_report = classification_report(true_labels, predictions, labels=label_names, zero_division=0)
    else:
        mse, predictions, true_labels = evaluate_regression(model, dataloader, device, ha_mode)

    pearson_corr = calculate_correlations(true_labels, predictions, method='pearson')
    spearman_corr = calculate_correlations(true_labels, predictions, method='spearman')
    kendall_corr = calculate_correlations(true_labels, predictions, method='kendall')

    if is_classification:
        return {'accuracy': acc, 'precision': precision, 'recall': recall,
                'pearson': pearson_corr, 'spearman': spearman_corr, 'kendall': kendall_corr}
    else:
        return {'mse': mse, 'pearson': pearson_corr, 'spearman': spearman_corr, 'kendall': kendall_corr}
