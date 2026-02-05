import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import os

from models import create_model
from data import get_visualization_transforms
from config import Config


def visualize_figure(model_path, test_dir, model_name='xception', num_classes=4, save_path='gradcam_figure.png'):
    """
    Create a figure like Fig. 14 in the paper.
    Shows 2 samples per class: original image + Grad-CAM heatmap.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Model
    model = create_model(model_name=model_name, num_classes=num_classes, pretrained=False).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Select Target Layer
    if model_name == 'xception':
        target_layers = [model.base_model.act4]
    elif model_name == 'inception_v3':
        target_layers = [model.base_model.Mixed_7c]
    elif model_name == 'inception_resnet_v2':
        target_layers = [model.base_model.conv2d_7b]

    cam = GradCAM(model=model, target_layers=target_layers)
    preprocess = get_visualization_transforms()
    
    # Class folders
    class_folders = {
        'COVID-19': 'Covid-19',
        'Normal': 'Normal',
        'Bacterial PNA': 'Bacterial Pneumonia',
        'Viral PNA': 'Viral Pneumonia'
    }
    
    # Find 2 samples per class
    samples = {}
    for display_name, folder_name in class_folders.items():
        folder_path = os.path.join(test_dir, folder_name)
        if os.path.exists(folder_path):
            images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg'))]
            if len(images) >= 2:
                samples[display_name] = [
                    os.path.join(folder_path, images[0]),
                    os.path.join(folder_path, images[1])
                ]
    
    class_names = list(samples.keys())
    
    # Create figure: 4 rows, 4 columns
    fig, axes = plt.subplots(4, len(class_names), figsize=(4 * len(class_names), 12))
    
    for col, class_name in enumerate(class_names):
        img_paths = samples[class_name]
        
        for img_idx, img_path in enumerate(img_paths[:2]):
            # Load image
            img = Image.open(img_path).convert('RGB')

            # Preprocess for model
            input_tensor = preprocess(img).unsqueeze(0).to(device)
            
            # Convert tensor to numpy for visualization (0-1 range)
            rgb_img_float = input_tensor[0].cpu().permute(1, 2, 0).numpy()
            rgb_img_float = (rgb_img_float - rgb_img_float.min()) / (rgb_img_float.max() - rgb_img_float.min())
            
            # Generate Grad-CAM
            grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
            visualization = show_cam_on_image(rgb_img_float, grayscale_cam, use_rgb=True)
            
            # Row indices
            row_orig = img_idx * 2
            row_cam = img_idx * 2 + 1
            
            # Plot original
            axes[row_orig, col].imshow(rgb_img_float)
            axes[row_orig, col].axis('off')
            if img_idx == 0:
                axes[row_orig, col].set_title(f"({chr(97 + col)}) {class_name}", fontsize=12, fontweight='bold')
            
            # Plot heatmap
            axes[row_cam, col].imshow(visualization)
            axes[row_cam, col].axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"Saved to {save_path}")
    plt.close()


if __name__ == "__main__":
    visualize_figure(
        model_path='checkpoints/model5_xception/best_model.pth',
        test_dir=Config.TEST_DIR,
        model_name='xception',
        num_classes=4,
        save_path='gradcam_figure.png'
    )