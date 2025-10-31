# 🚀 Pneumonia Detection CNN – Full One-Cell Colab Project (Optimized)
# https://www.kaggle.com/datasets/pcbreviglieri/pneumonia-xray-images

!pip install -q kaggle tensorflow matplotlib seaborn scikit-learn pydot graphviz

# ========== KAGGLE SETUP ==========
from google.colab import files
print("📁 Upload your kaggle.json file (from Kaggle > Account > Create API Token)")
files.upload()

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# ========== DOWNLOAD DATA ==========
!kaggle datasets download -d pcbreviglieri/pneumonia-xray-images
!unzip -q pneumonia-xray-images.zip -d dataset

# ========== IMPORT LIBRARIES ==========
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator, image
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ========== PATHS ==========
train_path = '/content/dataset/cnn/pneumonia_revamped/train'
valid_path = '/content/dataset/cnn/pneumonia_revamped/val'
test_path  = '/content/dataset/cnn/pneumonia_revamped/test'

# ========== IMAGE SETTINGS ==========
img_height = 128
img_width = 128
batch_size = 16

# ========== DATA GENERATORS ==========
train_gen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
).flow_from_directory(
    train_path,
    target_size=(img_height, img_width),
    color_mode="grayscale",
    class_mode="binary",
    batch_size=batch_size
)

valid_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    valid_path,
    target_size=(img_height, img_width),
    color_mode="grayscale",
    class_mode="binary",
    batch_size=batch_size
)

test_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    test_path,
    target_size=(img_height, img_width),
    color_mode="grayscale",
    class_mode="binary",
    batch_size=batch_size,
    shuffle=False
)

# ========== BUILD CNN (Optimized Architecture) ==========
cnn = Sequential([
    Input(shape=(img_height, img_width, 1)),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    GlobalAveragePooling2D(),

    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
cnn.summary()

# ========== CLASS WEIGHTS ==========
weights = compute_class_weight(class_weight='balanced',
                               classes=np.unique(train_gen.classes),
                               y=train_gen.classes)
cw = dict(zip(np.unique(train_gen.classes), weights))
print("\n⚖️ Class Weights:", cw)

# ========== CALLBACKS ==========
callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, min_lr=1e-6, verbose=1)
]

# ========== TRAIN ==========
history = cnn.fit(
    train_gen,
    epochs=5,
    validation_data=valid_gen,
    class_weight=cw,
    callbacks=callbacks
)

# ========== EVALUATE ==========
test_loss, test_acc = cnn.evaluate(test_gen)
print(f"\n✅ Test Accuracy: {test_acc*100:.2f}%")
print(f"✅ Test Loss: {test_loss:.4f}")

# ========== CONFUSION MATRIX ==========
preds = cnn.predict(test_gen)
pred_labels = (preds > 0.5).astype(int)
cm = confusion_matrix(test_gen.classes, pred_labels)
sns.heatmap(pd.DataFrame(cm,
                         index=["Normal","Pneumonia"],
                         columns=["Pred Normal","Pred Pneumonia"]),
            annot=True, fmt="d", cmap="Blues")
plt.title("🧠 Confusion Matrix")
plt.show()

print(classification_report(test_gen.classes, pred_labels, target_names=["Normal","Pneumonia"]))

# ========== SAVE MODEL ==========
cnn.save("cnn_pneumonia_model_optimized.h5")
print("\n💾 Model saved as cnn_pneumonia_model_optimized.h5")

# ========== UPLOAD IMAGE FOR PREDICTION ==========
print("\n📤 Upload a chest X-ray image for prediction")
uploaded = files.upload()

for fn in uploaded.keys():
    img = image.load_img(fn, target_size=(img_height, img_width), color_mode='grayscale')
    img_arr = image.img_to_array(img)/255.0
    img_arr = np.expand_dims(img_arr, axis=0)

    pred = cnn.predict(img_arr)[0][0]
    label = "PNEUMONIA" if pred > 0.5 else "NORMAL"

    plt.imshow(img, cmap='gray')
    plt.title(f"Prediction: {label}\nConfidence: {pred:.2f}")
    plt.axis('off')
    plt.show()
