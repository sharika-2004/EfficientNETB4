from google.colab import files
from IPython.display import display, Image
import numpy as np

def upload_and_predict():
    # Upload the file
    uploaded = files.upload()
    
    # Get the first uploaded file
    for filename in uploaded.keys():
        # Display the uploaded image
        print(f"\nUploaded Image: {filename}")
        display(Image(filename, width=300))
        
        # Make prediction
        img = tf.keras.preprocessing.image.load_img(filename, target_size=(224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        
        # Predict
        predictions = model.predict(img_array)
        predicted_class = class_names[np.argmax(predictions[0])]
        confidence = np.max(predictions[0])
        
        # Show results
        plt.imshow(img)
        plt.title(f"Prediction: {predicted_class}\nConfidence: {confidence:.2%}")
        plt.axis('off')
        plt.show()
        
        return predicted_class, confidence
print("Available flower classes in model:", class_names)
print("\nPlease upload a flower image (jpg, jpeg, png):")
pred_class, confidence = upload_and_predict()
print(f"\nFinal Prediction: {pred_class} ({confidence:.2%} confidence)")
  
