import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from models import PerceptAnonNetwork
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Generate Grad-CAM visualizations for PerceptAnonNetwork.")
parser.add_argument("--checkpoint", type=str, default="/content/drive/MyDrive/perceptanon/rn50-all-ckpts/labels10-clf_all_resnet50.pth.tar", help="Path to model checkpoint")
parser.add_argument("--model_name", type=str, default="resnet50", choices=["resnet18", "resnet50", "resnet152", "densenet121", "vgg11", "alexnet", "vit_b_16"], help="Model architecture")
parser.add_argument("--img_folder", type=str, default="/content/drive/MyDrive/perceptanon/sample_imgs/", help="Folder containing input images")
parser.add_argument("--resized_folder", type=str, default="/content/drive/MyDrive/perceptanon/resized/", help="Folder to save resized images")
parser.add_argument("--maps_folder", type=str, default="/content/drive/MyDrive/perceptanon/maps/", help="Folder to save Grad-CAM visualizations")
parser.add_argument("--num_outputs", type=int, default=10, help="Number of model output classes")
parser.add_argument("--is_classification", action="store_true", help="Use classification mode (default: regression)")
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for inference (cuda or cpu)")
args = parser.parse_args()

DEVICE = torch.device(args.device)

os.makedirs(args.resized_folder, exist_ok=True)
os.makedirs(args.maps_folder, exist_ok=True)

print(f"Loading model: {args.model_name} from checkpoint: {args.checkpoint}")
model = PerceptAnonNetwork(num_outputs=args.num_outputs, is_classification=args.is_classification).get_model(args.model_name)
checkpoint = torch.load(args.checkpoint, map_location=DEVICE)
model.load_state_dict(checkpoint["state_dict"])
model.to(DEVICE)
model.eval()

# Select Grad-CAM target layer
target_layers = [model.layer4[-1]]  # For ResNet-based models
cam = GradCAM(model=model, target_layers=target_layers)

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

for filename in tqdm(os.listdir(args.img_folder)):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # image pathing
        img_path = os.path.join(args.img_folder, filename)
        img_name = os.path.splitext(filename)[0]
        image = Image.open(img_path).convert("RGB")
        # resize to model input size
        image_resized = image.resize((224, 224))
        resized_path = os.path.join(args.resized_folder, f"{img_name}-resized.jpg")
        # original image resized
        image_resized.save(resized_path)
        input_tensor = test_transforms(image).unsqueeze(0).to(DEVICE)
        # gradcam
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
        image_array = np.array(image_resized).astype(np.float32) / 255.0
        visualization = show_cam_on_image(image_array, grayscale_cam, use_rgb=True)
        output_path = os.path.join(args.maps_folder, f"{img_name}-gradcam.jpg")
        # gradcam image save
        plt.imsave(output_path, visualization)
