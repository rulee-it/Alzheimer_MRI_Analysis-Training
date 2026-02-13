# Alzheimer MRI Analysis - Training

This repository includes a TensorFlow/Keras training script to classify Alzheimer's MRI brain scans into 4 categories:

- MildDemented
- ModerateDemented
- VeryMildDemented
- NonDemented

## Dataset structure

Place your images in the following structure relative to the repo root:

```
dataset/
  train/
    MildDemented/
    ModerateDemented/
    VeryMildDemented/
    NonDemented/
  test/
    MildDemented/
    ModerateDemented/
    VeryMildDemented/
    NonDemented/
```

## Train the model

1) Ensure dependencies are installed from `requirements.txt`.
2) Run the training script:

```
python Alzheimer_MRI_Analysis/Alzheimer_MRI_Analysis/train_cnn.py
```

The script will:

- Load and augment images using `ImageDataGenerator`
- Train for 20 epochs
- Save the model to `models/alzheimer_model.h5`
- Save accuracy and loss plots to `static/accuracy.png` and `static/loss.png`
- Save `models/class_indices.json` for class label mapping

## Predict on a single MRI image

After training, run the prediction script with a single image path:

```
python Alzheimer_MRI_Analysis/Alzheimer_MRI_Analysis/predict_mri.py <path_to_image>
```

What it does:

- Loads `models/alzheimer_model.h5`
- Resizes the image to the model's required input size, normalizes to [0,1]
- Predicts probabilities for classes in this order:
  - MildDemented, ModerateDemented, NonDemented, VeryMildDemented
- Displays a bar chart of probabilities
- Prints the predicted class and the probabilities per class
