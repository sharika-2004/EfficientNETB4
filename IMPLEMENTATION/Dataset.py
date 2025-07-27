from google.colab import files
import os
import zipfile

# Upload kaggle.json (upload manually when prompted)
files.upload()

# Move kaggle.json to the correct location
os.makedirs('/root/.kaggle', exist_ok=True)
!mv kaggle.json /root/.kaggle/
!chmod 600 /root/.kaggle/kaggle.json

# Download the Flowers Recognition dataset
!kaggle datasets download -d alxmamaev/flowers-recognition

# Unzip the dataset
with zipfile.ZipFile("flowers-recognition.zip", 'r') as zip_ref:
    zip_ref.extractall("flowers_dataset")

import matplotlib.pyplot as plt

data_dir = "flowers_dataset/flowers"
class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
print("Flower classes found:", class_names)

# Display sample images
plt.figure(figsize=(15, 8))
for i, class_name in enumerate(class_names):
    class_dir = os.path.join(data_dir, class_name)
    image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if image_files:    
        img_path = os.path.join(class_dir, image_files[0])    
        img = plt.imread(img_path)    
        plt.subplot(2, 3, i + 1)    
        plt.imshow(img)    
        plt.title(f"{class_name} ({len(image_files)} images)")    
        plt.axis('off')    

plt.tight_layout()
plt.show()
