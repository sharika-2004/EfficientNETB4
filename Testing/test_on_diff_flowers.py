# Prediction function for EfficientNetB4
def predict_flower_efficientnet(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = preprocess_input(img_array)  # EfficientNet-specific preprocessing
    img_array = tf.expand_dims(img_array, 0)  # Add batch dimension

    predictions = model.predict(img_array)
    predicted_class = class_names[tf.argmax(predictions[0]).numpy()]
    confidence = tf.reduce_max(predictions[0]).numpy()

    plt.imshow(img)
    plt.title(f"EfficientNetB4 Prediction: {predicted_class} ({confidence:.2%})")
    plt.axis('off')
    plt.show()
    
    return predicted_class, confidence

# Test on a rose
rose_image = os.path.join(data_dir, 'rose', os.listdir(os.path.join(data_dir, 'rose'))[0])
print("Testing on Rose:")
predict_flower_efficientnet(rose_image)

# Test on a dandelion
dandelion_image = os.path.join(data_dir, 'dandelion', os.listdir(os.path.join(data_dir, 'dandelion'))[0])
print("Testing on Dandelion:")
predict_flower_efficientnet(dandelion_image)

# Test on a sunflower
sunflower_image = os.path.join(data_dir, 'sunflower', os.listdir(os.path.join(data_dir, 'sunflower'))[0])
print("Testing on Sunflower:")
predict_flower_efficientnet(sunflower_image)

# Test on a tulip
tulip_image = os.path.join(data_dir, 'tulip', os.listdir(os.path.join(data_dir, 'tulip'))[0])
print("Testing on Tulip:")
predict_flower_efficientnet(tulip_image)

# Test on a daisy (if available in your dataset)
if 'daisy' in class_names:
    daisy_image = os.path.join(data_dir, 'daisy', os.listdir(os.path.join(data_dir, 'daisy'))[0])
    print("Testing on Daisy:")
    predict_flower_efficientnet(daisy_image)
else:
    print("Daisy class not found in dataset")
