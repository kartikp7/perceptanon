import torch
import os
import datetime
import time
from tqdm import tqdm

def save_checkpoint(state, filename):
    '''
        Saving model
    '''
    torch.save(state, filename)

def load_weights(model, checkpoint_path, optimizer=None, scheduler=None, device=None):
    '''
        Load & return model from checkpoint path
    '''
    if device is None:
        device = torch.device("cuda:0")

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at '{checkpoint_path}'")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])

    if scheduler is not None and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])

    epoch = checkpoint.get('epoch', 'Unknown')
    # print(f"Loaded weights from checkpoint (Epoch: {epoch})")
    return model, epoch, optimizer, scheduler

def train_epoch(model, dataloader, criterion, optimizer, device, ha_mode):
    '''
        Train single epoch
    '''
    model.train()
    running_loss = 0.0
    for batch in dataloader:
        if ha_mode == "HA1":
            _, images, labels = batch  # HA1 uses only anonymized images
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
        elif ha_mode == "HA2":
            orig_images, anon_images, labels = batch  # HA2 uses both original and anonymized images
            orig_images, anon_images, labels = orig_images.to(device), anon_images.to(device), labels.to(device)
            outputs = model(orig_images, anon_images)

        optimizer.zero_grad()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * labels.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

def validate_epoch(model, dataloader, criterion, device, ha_mode):
    '''
        Compute validation set loss
    '''
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            if ha_mode == "HA1":
                _, images, labels = batch
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
            elif ha_mode == "HA2":
                orig_images, anon_images, labels = batch
                orig_images, anon_images, labels = orig_images.to(device), anon_images.to(device), labels.to(device)
                outputs = model(orig_images, anon_images)

            loss = criterion(outputs, labels)
            test_loss += loss.item() * labels.size(0)

    test_loss /= len(dataloader.dataset)
    return test_loss

def train_model(ha_mode, model, dataloaders, criterion, optimizer, scheduler, num_epochs=25, device=None, checkpoint_path='checkpoint.pth.tar', log_path='training.log'):
    '''
        Train model
    '''
    if device is None:
        device = torch.device("cuda:0")

    model.to(device)
    best_loss = float('inf')
    best_epoch = 0

    # Logging experiment
    with open(log_path, 'a') as log_file:
        log_file.write(f"Training started at {datetime.datetime.now()}\n")

    for epoch in tqdm(range(num_epochs), desc='Training'):
        train_loss = train_epoch(model, dataloaders['train'], criterion, optimizer, device, ha_mode)
        val_loss = validate_epoch(model, dataloaders['val'], criterion, device, ha_mode)
        current_lr = optimizer.param_groups[0]['lr']

        if scheduler:
            scheduler.step()

        # Checkpointing
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch+1
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict() if scheduler else None,
                'val_loss': val_loss,
                'lr': current_lr,
            }, checkpoint_path)

        # Logging
        with open(log_path, 'a') as log_file:
            log_file.write(f"Epoch: {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.6f}\n")

    with open(log_path, 'a') as log_file:
        log_file.write(f"Training ended at {datetime.datetime.now()}. Best Epoch: {best_epoch}\n")

    model, loaded_epoch, optimizer, scheduler = load_weights(model, checkpoint_path, optimizer, scheduler)

    with open(log_path, 'a') as log_file:                                        
        log_file.write(f"Loaded Epoch {loaded_epoch}\n")

    return model
