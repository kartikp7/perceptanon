import argparse
import os
import numpy as np
# import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from anon_dataset import AnonDataset
from utils import *
from models import *
from train import train_model 
from eval import evaluate_model
from collections import Counter

def get_transforms(img_size):
    '''
        Image transforms
    '''
    train_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_transforms, test_transforms

def get_loaders(data_splits, label_type, is_classification, batch_size, num_workers, root_dir, img_size=224):
    '''
        Get dataloaders given data splits
    '''
    train_transforms, test_transforms = get_transforms(img_size)
    label_transforms = get_label_transforms(label_type, is_classification)

    # Create dataset
    trainset = AnonDataset(annotations=data_splits['train'], root_dir=root_dir, transform=train_transforms, transform_label=label_transforms)
    valset = AnonDataset(annotations=data_splits['val'], root_dir=root_dir, transform=test_transforms, transform_label=label_transforms)
    testset = AnonDataset(annotations=data_splits['test'], root_dir=root_dir, transform=test_transforms, transform_label=label_transforms)

    dataloaders = {
        'train': DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        'val': DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        'test': DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        'trainset': trainset
    }

    return dataloaders

def get_num_outputs(label_type, is_classification):
    if not is_classification:
        return 1 # regression
    
    if label_type == 'labels10':
        return 10
    elif label_type == 'labels5':
        return 5
    elif label_type == 'labels3':
        return 3
    elif label_type == 'binary':
        return 2
    else:
        raise ValueError(f'Invalid label_type: {label_type}')
    
def get_ce_weights(train_dataset, num_classes=10):
    '''
        Get weights tensor for weighted CE.
        Input: train_dataset
    '''
    # Count label distribution
    class_counts = Counter()
    for _, _, label in train_dataset:
        class_counts[label] += 1

    total_counts = sum(class_counts.values())
    class_weights = {class_id: total_counts/count for class_id, count in class_counts.items()}

    # Handle missing classes by assigning a default weight
    default_weight = 1.0
    weights_tensor = torch.tensor([class_weights.get(class_id, default_weight) for class_id in range(num_classes)])

    return weights_tensor

def main(args):
    device = torch.device("cuda:0")

    # Load dataset splits (from precomputed or generate dynamically)
    data_splits = get_dataset_splits(
        os.path.join(args.data_root,f'{args.ha_mode}.csv'), 
        split_type=args.split_type, 
        dataset_splits_path=args.dataset_splits
    )

    # Get dataloaders
    dataloaders = get_loaders(
        data_splits, label_type=args.label_type, is_classification=args.is_classification,
        batch_size=args.batch_size, num_workers=args.num_workers, root_dir=args.data_root
    )

    # Determine model output dims
    num_outputs = get_num_outputs(label_type=args.label_type, is_classification=args.is_classification)

    # Determine model architecture 
    if args.ha_mode == "HA1":
        model = PerceptAnonHA1(num_outputs, args.pretrained, args.is_classification).get_model(args.model_name)
    else:
        model = PerceptAnonHA2(args.model_name, num_outputs, args.pretrained, args.is_classification)

    # Define loss function
    loss_fn = nn.CrossEntropyLoss() if args.is_classification else nn.MSELoss()
    if args.weightedce:
        weights = get_ce_weights(dataloaders['trainset'], num_outputs).to(device)
        loss_fn = nn.CrossEntropyLoss(weight=weights)

    # Optimizer & Scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4) if args.optimizer == 'adam' else optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1) if args.scheduler == 'steplr' else optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs) if args.scheduler == 'cosine' else None

    # Paths
    # log_path = f'logs/{args.ha_mode}/{args.model_name}/{args.split_type}/{args.label_type}.log'
    # ckpt_path = f'ckpts/{args.ha_mode}/{args.model_name}/{args.split_type}/{args.label_type}.pth.tar'
    # os.makedirs(os.path.dirname(log_path), exist_ok=True)
    # os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    # Saving checkpoints and log files
    os.makedirs(args.savepath, exist_ok=True)
    log_path = os.path.join(args.savepath, f'logs/{args.ha_mode}/{args.model_name}/{args.split_type}/{args.label_type}.log')
    ckpt_path = os.path.join(args.savepath, f'ckpts/{args.ha_mode}/{args.model_name}/{args.split_type}/{args.label_type}.pth.tar')
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    
    # Log experiment meta-data
    with open(log_path, 'a') as log_file:
        log_file.write(f"{args.ha_mode} | Model: {args.model_name} | Split: {args.split_type} | Label: {args.label_type} | Classification: {args.is_classification}\n")

    # Train model
    trained_model = train_model(args.ha_mode, model, dataloaders, loss_fn, optimizer, scheduler, args.num_epochs, device, checkpoint_path=ckpt_path, log_path=log_path)

    # Evaluate
    label_names = np.arange(num_outputs) if args.is_classification else None
    eval_results = evaluate_model(trained_model, dataloaders['test'], device, is_classification=args.is_classification, ha_mode=args.ha_mode, label_names=label_names)

    # Log results
    with open(log_path, 'a') as log_file:
        log_file.write('-'*10 + '\n')
        log_file.write(f"Results: {eval_results}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run HA1 or HA2 experiments')
    parser.add_argument('--ha_mode', type=str, default='HA1', choices=['HA1', 'HA2'], help='Specify experiment type (HA1 or HA2)')
    # parser.add_argument('--anno_path', type=str, required=True, default='data/HA1.csv', help='Path to annotations file')
    parser.add_argument('--data_root', type=str, default='data', help='Path to data folder with dataset and annotation files')
    parser.add_argument('--split_type', type=str, default='all', choices=['all', 'task-person', 'task-face', 'loov-coco', 'loov-voc', 'loov-lfw', 'loov-celeba', 'anon-mask', 'anon-inpaint', 'anon-blurpix'], help='Dataset split experiment type')
    parser.add_argument('--label_type', type=str, default='mean', choices=['mean', 'labels10', 'labels5', 'labels3', 'binary'], help='Score type')
    parser.add_argument('--is_classification', action='store_true')
    parser.add_argument('--model_name', type=str, default='resnet18', choices=['resnet18', 'resnet50', 'resnet152', 'densenet121', 'vgg11', 'alexnet', 'vit_b_16'], help='model architecture')
    parser.add_argument('--savepath', type=str, default='results', help='Base folder to save logs and checkpoints')

    parser.add_argument('--dataset_splits', type=str, default=None, help='Path to precomputed dataset splits folder (optional)')
    parser.add_argument('--num_workers', type=int, default=4, help='num workers for dataloaders')
    # parser.add_argument('--make_negs', action='store_true')
    parser.add_argument('--no-pretrained', dest='pretrained', action='store_false', help='Include this flag to not use pretrained Imagenet weights')
    parser.set_defaults(pretrained=True)
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['adam', 'sgd'], help='Optimizer to use')
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['steplr', 'cosine'], help='Learning rate scheduler to use')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--weightedce', dest='weightedce', action='store_true', help='Include this flag to use weighted cross entropy')
    args = parser.parse_args()
    main(args)