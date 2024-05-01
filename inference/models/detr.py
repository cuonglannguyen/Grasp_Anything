import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from transformers import CLIPModel, CLIPProcessor
import clip
from inference.models.grasp_model import GraspModel, ResidualBlock

class DETRwithCLIP(GraspModel):
    def __init__(self, input_channels=3, dropout=False, prob=0.0, channel_size = 32):
        super(DETRwithCLIP, self).__init__()
        # Load CLIP model
        self.clip_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
        self.processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
                # Image preprocessing transformation
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to CLIP's input size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet stats
        ])
        # Define the initial convolution layer to adjust input channels if necessary
        # Normally, CLIP handles 3 RGB channels; if your input differs, you might need to adjust here.
        self.initial_conv = nn.Conv2d(input_channels, 3, kernel_size=1) if input_channels != 3 else nn.Identity()
        
        # Define the rest of the network
        self.conv1 = nn.Conv2d(512, 128, kernel_size=3, stride=1, padding=1)  # Adjusted for CLIP's feature dimension
        self.bn1 = nn.BatchNorm2d(128)
        
        self.res1 = ResidualBlock(128, 128)
        self.res2 = ResidualBlock(128, 128)
        self.res3 = ResidualBlock(128, 128)
        self.res4 = ResidualBlock(128, 128)
        self.res5 = ResidualBlock(128, 128)
        
        # Output layers
        self.pos_output = nn.Conv2d(128, 2, kernel_size=1)  # x, y position
        self.cos_output = nn.Conv2d(128, 1, kernel_size=1)  # cosine of angle
        self.sin_output = nn.Conv2d(128, 1, kernel_size=1)  # sine of angle
        self.width_output = nn.Conv2d(128, 1, kernel_size=1)  # width
        
        self.dropout1 = nn.Dropout(p=prob) if dropout else nn.Identity()

    def forward(self, images):
        # Adjust input channels if necessary
        #images = self.initial_conv(images)
        if not isinstance(images, torch.Tensor):
            images = [self.preprocess(Image.fromarray(img)) if isinstance(img, np.ndarray) else self.preprocess(img) for img in images]
            images = torch.stack(images)  # Create a batch from a list of tensors

        # Process images through CLIP processor to prepare for model input
        inputs = {'pixel_values': images}  # Directly use preprocessed tensor images
        
        # Extract features using CLIP
        features = self.clip_model.get_image_features(**inputs)
        features = features.view(-1, 512, 1, 1)
        # Process features through the network
        x = F.relu(self.bn1(self.conv1(features)))
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        
        # Apply dropout if enabled
        x = self.dropout1(x)
        
        # Compute outputs
        pos_output = self.pos_output(x)
        cos_output = self.cos_output(x)
        sin_output = self.sin_output(x)
        width_output = self.width_output(x)
        
        return pos_output, cos_output, sin_output, width_output
