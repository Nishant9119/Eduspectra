#code to generate Figure 3 in paper: Visual depiction of images of training dataset with their class labels
import os
import random
import matplotlib.pyplot as plt
from PIL import Image

#dataset path
base_path = "C:\\Users\\Adros\\Desktop\\Dataset\\Train"
categories = ["Dropouts", "NonDropouts"]

image_label_list = []

for label in categories:
    folder_path = os.path.join(base_path, label)
    for file in os.listdir(folder_path):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_label_list.append((os.path.join(folder_path, file), label))

# Shuffle and select 49 images for 7x7 grid
random.shuffle(image_label_list)
selected_images = image_label_list[:49]

# Create a 7x7 image grid
fig, axes = plt.subplots(7, 7, figsize=(12, 12))
fig.subplots_adjust(hspace=0.05, wspace=0.2)

for ax, (img_path, label) in zip(axes.flat, selected_images):
    img = Image.open(img_path)
    ax.imshow(img)
    ax.set_title(label, fontsize=8)
    ax.axis('off')
plt.savefig("final_image.jpg", dpi=300, bbox_inches='tight')
plt.tight_layout()
plt.show()
