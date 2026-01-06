import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from models import create_model
from data import create_dataloaders
from config import Config
import os


class HybridSystem:
    def __init__(self, model_3c_path, model_2c_path, model_name='xception', device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        self.model_3 = create_model(model_name=model_name, num_classes=3, pretrained=False).to(self.device)
        self.model_3.load_state_dict(torch.load(model_3c_path, map_location=self.device)['model_state_dict'])
        self.model_3.eval()
        
        self.model_2 = create_model(model_name=model_name, num_classes=2, pretrained=False).to(self.device)
        self.model_2.load_state_dict(torch.load(model_2c_path, map_location=self.device)['model_state_dict'])
        self.model_2.eval()

    def predict_batch(self, images):
        """Hybrid prediction: 3-class -> 2-class for pneumonia."""
        batch_size = images.size(0)
        
        with torch.no_grad():
            logits_3 = self.model_3(images)
            pred_3 = logits_3.argmax(1)
            
            pred_4class = torch.zeros(batch_size, dtype=torch.long, device=self.device)
            pred_4class[pred_3 == 0] = 0  # COVID
            pred_4class[pred_3 == 1] = 1  # Normal
            
            pneumonia_mask = pred_3 == 2
            if pneumonia_mask.any():
                logits_2 = self.model_2(images[pneumonia_mask])
                pred_2 = logits_2.argmax(1)
                
                pneumonia_indices = pneumonia_mask.nonzero(as_tuple=True)[0]
                pred_4class[pneumonia_indices[pred_2 == 0]] = 2  # Bacterial
                pred_4class[pneumonia_indices[pred_2 == 1]] = 3  # Viral
        
        return pred_4class


def evaluate(model_path, class_mode, model_name='xception', hybrid_paths=None):
    """
    Unified evaluation function for Model 8, 9, and 10.
    
    Args:
        model_path: Path to model checkpoint (ignored if hybrid_paths is set)
        class_mode: '2-class', '3-class', or '4-class'
        model_name: Architecture name
        hybrid_paths: Dict with 'model_3c' and 'model_2c' paths for hybrid evaluation
    
    Returns:
        Dictionary with accuracy, precision, recall, f1_score
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = {'2-class': 2, '3-class': 3, '4-class': 4}[class_mode]
    
    # Load model or hybrid system
    if hybrid_paths:
        predictor = HybridSystem(
            model_3c_path=hybrid_paths['model_3c'],
            model_2c_path=hybrid_paths['model_2c'],
            model_name=model_name
        )
        test_class_mode = '4-class'  # Hybrid evaluates on 4-class data
    else:
        model = create_model(model_name=model_name, num_classes=num_classes, pretrained=False).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device)['model_state_dict'])
        model.eval()
        test_class_mode = class_mode
    
    if model_name == 'inception_v3':
        img_size = 299
    else:
        img_size = Config.IMG_SIZE

    # Load test data
    _, _, test_loader = create_dataloaders(
        train_dir=Config.TRAIN_DIR,
        val_dir=Config.VALIDATION_DIR,
        test_dir=Config.TEST_DIR,
        batch_size=32,
        class_mode=test_class_mode,
        img_size=Config.IMG_SIZE,
        use_zoom=False,
        use_contrast=False,
        use_translation=False,
        num_workers=0
    )
    
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            
            if hybrid_paths:
                preds = predictor.predict_batch(images).cpu().numpy()
            else:
                preds = model(images).argmax(1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    
    y_true, y_pred = np.array(all_labels), np.array(all_preds)
    
    return {
        'accuracy': accuracy_score(y_true, y_pred) * 100,
        'precision': precision_score(y_true, y_pred, average='weighted') * 100,
        'recall': recall_score(y_true, y_pred, average='weighted') * 100,
        'f1_score': f1_score(y_true, y_pred, average='weighted') * 100
    }


def print_metrics_table(results):
    """Print metrics table like the paper."""
    print("\n" + "=" * 80)
    print(f"{'Method':<15} {'Accuracy (%)':<15} {'F-measure (%)':<15} {'Precision (%)':<15} {'Recall (%)':<15}")
    print("=" * 80)
    
    for method, metrics in results.items():
        print(f"{method:<15} {metrics['accuracy']:<15.2f} {metrics['f1_score']:<15.2f} {metrics['precision']:<15.2f} {metrics['recall']:<15.2f}")
    
    print("=" * 80)


def run_full_evaluation():
    """Evaluate all models."""
    results = {}
    
    # 4-class models (Model 1-7)
    print("Evaluating Model 1-7 (4-class)...")
    for i in range(1, 8):
        model_id = f'model{i}'
        model_config = Config.MODELS[model_id]
        checkpoint_path = f"checkpoints/{model_id}_{model_config['architecture']}/best_model.pth"
        
        # Check if checkpoint exists
        if os.path.exists(checkpoint_path):
            print(f"  Evaluating {model_id}...")
            results[f'Model {i}'] = evaluate(
                model_path=checkpoint_path,
                class_mode='4-class',
                model_name=model_config['architecture']
            )
        else:
            print(f"  Skipping {model_id} - checkpoint not found")
    
    print("Evaluating Model 8 (2-class)...")
    results['Model 8'] = evaluate(
        model_path='checkpoints/model8_xception/best_model.pth',
        class_mode='2-class'
    )
    
    print("Evaluating Model 9 (3-class)...")
    results['Model 9'] = evaluate(
        model_path='checkpoints/model9_xception/best_model.pth',
        class_mode='3-class'
    )
    
    print("Evaluating Model 10 (Hybrid)...")
    results['Model 10'] = evaluate(
        model_path=None,
        class_mode='4-class',
        hybrid_paths={
            'model_3c': 'checkpoints/model9_xception/best_model.pth',
            'model_2c': 'checkpoints/model8_xception/best_model.pth'
        }
    )
    
    print_metrics_table(results)
    return results


if __name__ == "__main__":
    run_full_evaluation()
   