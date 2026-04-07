import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt
import numpy as np
import os
import zipfile
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc  # ← ADDED
from sklearn.preprocessing import label_binarize  # ← ADDED
import cv2

# ==========================================
# 1. SETUP & EXTRACTION
# ==========================================
zip_path = r'C:\Users\Sneha Jha\Downloads\lungs cancer dataset.zip'
extract_path = r'C:\Users\Sneha Jha\Downloads\lungs_cancer_extracted'

if not os.path.exists(extract_path):
    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print("Extraction complete!")

# ==========================================
# 2. DATA LOADING
# ==========================================
IMG_SIZE = (256, 256)
BATCH_SIZE = 32

train_ds = tf.keras.utils.image_dataset_from_directory(
    extract_path,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    extract_path,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
print(f"\nDetected Cancer Types: {class_names}")

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# ==========================================
# 3. CNN MODEL ARCHITECTURE
# ==========================================
inputs = layers.Input(shape=(256, 256, 3))
x = layers.Rescaling(1./255)(inputs)
x = layers.Conv2D(32, (3, 3), activation='relu', name="final_conv_layer_1")(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(64, (3, 3), activation='relu', name="final_conv_layer_2")(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(128, (3, 3), activation='relu', name="target_conv_layer")(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Flatten()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(len(class_names), activation='softmax')(x)

model = models.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# ==========================================
# 4. TRAINING
# ==========================================
print("\nTraining starting...")
history = model.fit(train_ds, validation_data=val_ds, epochs=15)

# ==========================================
# 5. GRAD-CAM FUNCTION
# ==========================================
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# ==========================================
# 6. FINAL VISUALIZATION & EVALUATION
# ==========================================

# A. Visual Batch Check
plt.figure(figsize=(20, 12))
for images, labels in val_ds.take(1):
    predictions = model.predict(images)
    sample_img = images[0]
    sample_label = labels[0]

    num_images_to_show = min(32, len(images))
    for i in range(num_images_to_show):
        ax = plt.subplot(4, 8, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        actual = class_names[labels[i]]
        pred = class_names[np.argmax(predictions[i])]
        confidence = 100 * np.max(predictions[i])
        color = "green" if actual == pred else "red"
        plt.title(f"A: {actual}\nP: {pred}\n({confidence:.0f}%)", color=color, fontsize=8)
        plt.axis("off")
plt.tight_layout()
plt.show()

# B. Collect predictions for metrics
y_true, y_pred, y_scores = [], [], []   # ← y_scores ADDED for ROC
for images, labels in val_ds:
    preds = model.predict(images, verbose=0)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))
    y_scores.extend(preds)              # ← softmax probabilities collected

y_true   = np.array(y_true)
y_scores = np.array(y_scores)

# C. Confusion Matrix
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix: Statistical Accuracy')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()

# D. Classification Report
print("\n" + "="*40)
print("      MEDICAL CLASSIFICATION REPORT")
print("="*40)
print(classification_report(y_true, y_pred, target_names=class_names))

# ==========================================
# E. ROC CURVE  ← NEW SECTION
# ==========================================
# E. ROC CURVE (works for 2 or more classes)
n_classes = len(class_names)

if n_classes == 2:
    # Binary case — directly use column 1 (positive class probability)
    y_true_bin = label_binarize(y_true, classes=[0, 1])
    fpr, tpr, _ = roc_curve(y_true_bin, y_scores[:, 1])
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Classifier')
    plt.xlabel('False Positive Rate', fontsize=13)
    plt.ylabel('True Positive Rate', fontsize=13)
    plt.title('ROC Curve', fontsize=15)
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

else:
    # Multi-class: one-vs-rest per class
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
    plt.figure(figsize=(10, 7))
    colors = plt.cm.tab10.colors

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[i], lw=2,
                 label=f"{class_names[i]}  (AUC = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=13)
    plt.ylabel('True Positive Rate', fontsize=13)
    plt.title('ROC Curve – One-vs-Rest per Cancer Type', fontsize=15)
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
# F. Grad-CAM Explainable AI
img_for_cam = tf.expand_dims(sample_img, axis=0)
heatmap = make_gradcam_heatmap(img_for_cam, model, "target_conv_layer")

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title(f"Original Scan\n(True: {class_names[sample_label]})")
plt.imshow(sample_img.numpy().astype("uint8"))
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("AI Focus Area (Grad-CAM)")
heatmap_rescaled = cv2.resize(heatmap, (256, 256))
plt.imshow(sample_img.numpy().astype("uint8"))
plt.imshow(heatmap_rescaled, cmap='jet', alpha=0.4)
plt.axis("off")

plt.tight_layout()
plt.show()