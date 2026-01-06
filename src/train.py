import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from data import create_dataloaders
from models import create_model
from config import Config
from pathlib import Path
import json, time


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    """Train for one epoch, return average loss and accuracy."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda', enabled=scaler is not None):  
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        if scaler is not None:                          
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)
    
    return total_loss / total, correct / total


def validate(model, loader, criterion, device):
    """Validate model, return average loss and accuracy."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=torch.cuda.is_available()): 
        for images, labels in loader:  
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += images.size(0)
    
    return total_loss / total, correct / total


def plot_history(history, save_path):
    """Plot training and validation loss/accuracy in one figure with 4 lines."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    ax.plot(epochs, history['train_loss'], color='blue', label='Train Loss', linewidth=2)
    ax.plot(epochs, history['val_loss'], color='red', label='Val Loss', linewidth=2)
    ax.plot(epochs, history['train_acc'], color='green', label='Train Acc', linewidth=2)
    ax.plot(epochs, history['val_acc'], color='orange', label='Val Acc', linewidth=2)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss / Accuracy')
    ax.set_title('Training History')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def train(config, model_id='model5', class_mode='4-class'):
    """
    Main training function.
    
    Args:
        config: Config class with all parameters
        model_id: from MODELS config
        class_mode: '4-class', '3-class', or '2-class'
    """
    
    model_config = config.get_model_config(model_id)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"\nModel: {model_id}")
    print(f"  Architecture: {model_config['architecture']}")
    print(f"  Contrast: {model_config['use_contrast']}")
    print(f"  Augmentation: {model_config['use_augmentation']}")
    print(f"  Transfer Learning: {model_config['use_transfer_learning']}")

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    num_classes = model_config['num_classes']
    class_mode = f"{num_classes}-class"

    if class_mode == '4-class':
        batch_size = config.BATCH_SIZE_SINGLE
        epochs = config.EPOCHS_SINGLE
        lr = config.LR_SINGLE
    else:  # 3-class or 2-class
        batch_size = config.BATCH_SIZE_HYBRID
        epochs = config.EPOCHS_HYBRID
        lr = config.LR_HYBRID

    save_dir = Path(f"{config.CHECKPOINT_DIR}/{model_id}2_{model_config['architecture']}")
    save_dir.mkdir(parents=True, exist_ok=True)

    
    # Data use model specific settings 
    print("\nLoading data...")
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dir=config.TRAIN_DIR,
        val_dir=config.VALIDATION_DIR,
        test_dir=config.TEST_DIR,
        batch_size=batch_size,
        class_mode=class_mode,
        img_size=config.IMG_SIZE,
        use_zoom=model_config['use_augmentation'],
        use_contrast=model_config['use_contrast'],
        use_translation=model_config['use_augmentation']
    )
    print(f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}")
    
    # Model - use model-specific settings
    print(f"\nCreating {model_config['architecture']} model ({num_classes} classes)...")
    model = create_model(
        model_name=model_config['architecture'],
        num_classes=num_classes,
        pretrained=model_config['use_transfer_learning']
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
    
    # Training loop
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0

    print(f"\nConfig: epochs={epochs}, batch={batch_size}, lr={lr}")
    print("-" * 60)

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        elapsed = time.time() - start_time
        print(f"Epoch {epoch:3d}/{epochs} | "
              f"Train: {train_loss:.4f}/{train_acc:.4f} | "
              f"Val: {val_loss:.4f}/{val_acc:.4f} | {elapsed:.1f}s")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch, 
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc, 
                'val_loss': val_loss,
                'model_id': model_id,
                'model_config': model_config,
                'num_classes': num_classes
            }, save_dir / 'best_model.pth')
            print(f"  -> Best model saved! (Val Acc: {val_acc:.4f})")
        
        if epoch % 5 == 0:
            plot_history(history, save_dir / 'training_history.png')
    
     # Save last model after training completes
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'val_loss': val_loss,
        'model_id': model_id,
        'model_config': model_config,
        'num_classes': num_classes
    }, save_dir / 'last_model.pth')
    print(f"  -> Last model saved!")

    plot_history(history, save_dir / 'training_history.png')
    with open(save_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print("-" * 60)
    print(f"Training complete! Best Val Acc: {best_val_acc:.4f}")
    print(f"Saved to: {save_dir}")
    
    return model, history, save_dir


if __name__ == "__main__":
    
    train(Config, model_id='model1')
    train(Config, model_id='model2')
    train(Config, model_id='model3')
    train(Config, model_id='model4')
    train(Config, model_id='model5')
    train(Config, model_id='model6')
    train(Config, model_id='model7')
    train(Config, model_id='model8', class_mode='2-class')  
    train(Config, model_id='model9', class_mode='3-class')  