def predict_flower(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=IMG_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = tf.expand_dims(img_array, 0)  # Add batch dimension

    predictions = model.predict(img_array)
    predicted_class = class_names[tf.argmax(predictions[0]).numpy()]
    confidence = tf.reduce_max(predictions[0]).numpy()

    plt.imshow(img)
    plt.title(f"Predicted: {predicted_class} ({confidence:.2%})")
    plt.axis('off')
    plt.show()

    return predicted_class, confidence

# Test on a sample image
sample_image = os.path.join(data_dir, class_names[0], os.listdir(os.path.join(data_dir, class_names[0]))[0])
predict_flower(sample_image)
