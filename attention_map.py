import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
from transformers import ViTModel, ViTImageProcessor

# Load the pre-trained ViT model and image processor
model_name = 'google/vit-base-patch16-224'
model = ViTModel.from_pretrained(model_name, output_attentions=True)
model.eval()
image_processor = ViTImageProcessor.from_pretrained(model_name)

# Move model to device (GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Function to preprocess the image
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    # Resize the image to the expected size
    image = image.resize((224, 224), Image.BILINEAR)
    inputs = image_processor(images=image, return_tensors="pt")
    return inputs['pixel_values'].to(device), image

# Function to visualize attention maps
def visualize_attention(model, input_tensor, image, layer_number=-1, head_number=0):
    # Forward pass through the model to get attention weights
    with torch.no_grad():
        outputs = model(input_tensor)

    # Get the attention weights from the specified layer
    attentions = outputs.attentions  # List of tensors from each layer
    attention = attentions[layer_number]  # Shape: (batch_size, num_heads, seq_len, seq_len)

    # Select the attention weights from the specified head
    attention = attention[0, head_number]  # Shape: (seq_len, seq_len)

    # Compute the average attention weights across all heads (optional)
    # attention = attention.mean(dim=0)  # Uncomment to average over heads

    # Get the attention weights corresponding to the class token
    cls_attention = attention[0, 1:]  # Exclude the class token's attention to itself
    cls_attention = cls_attention.reshape(14, 14).cpu().numpy()  # Reshape to 14x14 grid

    # Normalize the attention map
    cls_attention = cls_attention - cls_attention.min()
    cls_attention = cls_attention / cls_attention.max()

    # Upsample the attention map to the image size
    cls_attention = cv2.resize(cls_attention, image.size, interpolation=cv2.INTER_CUBIC)

    # Convert the attention map to a heatmap
    heatmap = plt.cm.jet(cls_attention)[:, :, :3]  # Use only RGB channels

    # Overlay the heatmap on the image
    image_np = np.array(image) / 255.0
    overlay = image_np * 0.5 + heatmap * 0.5
    overlay = np.clip(overlay, 0, 1)

    # Display the image
    plt.figure(figsize=(8, 8))
    plt.imshow(overlay)
    plt.axis('off')
    plt.title('Attention Map Overlay')
    plt.show()

# Path to your image
image_path = '/content/ViT-pytorch/img/phone_resized.jpeg'
# Preprocess the image
input_tensor, orig_image = preprocess_image(image_path)

# Visualize the attention map
visualize_attention(model, input_tensor, orig_image, layer_number=-1, head_number=0)