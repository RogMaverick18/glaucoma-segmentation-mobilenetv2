"""
RIM-ONE r3 Final Base: MobileNetV2-UNet (Full Image)
====================================================================
Dataset: RIM-ONE r3
Strategy:
1. Standard MobileNetV2-UNet (No MSCA/LBFR/PPM).
2. Full image resizing to 512x512 (No high-res ROI extraction).
3. Stereo Image Cropping: Disregards right half of the 2144x1424 stereo images.
4. Expert 1 Masks (Healthy & Glaucoma folders).
5. Standard Augmentation (Online only).
6. Standard Training: Adam 1e-3, ReduceLROnPlateau, 100 epochs.
7. Standard Inference (No TTA).
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import albumentations as A
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, backend as K
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')

def configure_gpu():
    print("Configuring GPU settings...")
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
            print(f"✓ GPU configured: {len(physical_devices)} device(s) available")
            return True
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    return False

# ============================================================================
# RIM-ONE DATA LOADING (STEREO CROPPING + FULL IMAGE)
# ============================================================================
def discover_rimone_roots():
    roots = []
    kaggle_candidates = [
        "/kaggle/input/datasets/rogmaverick/rim-one-r3/RIM-ONE r3",
    ]
    for p in kaggle_candidates:
        if os.path.exists(p):
            # Check if this root contains Healthy or Glaucoma folders directly
            for sub in os.listdir(p):
                if sub.lower() in ['healthy', 'glaucoma and suspects', 'glaucoma']:
                    return [os.path.abspath(p)]
            roots.append(os.path.abspath(p))
    return roots

def find_rimone_mask_for_image(img_path):
    img_dir = os.path.dirname(img_path)
    parent_dir = os.path.dirname(img_dir)
    expert1_dir = os.path.join(parent_dir, 'Expert1_masks')
    
    img_basename = os.path.splitext(os.path.basename(img_path))[0]
    cup_path, disc_path = None, None
    
    if os.path.exists(expert1_dir):
        for f in os.listdir(expert1_dir):
            if not f.endswith('.png'): continue
            if img_basename in f:
                f_lower = f.lower()
                if 'cup' in f_lower: cup_path = os.path.join(expert1_dir, f)
                elif 'disc' in f_lower: disc_path = os.path.join(expert1_dir, f)
    return cup_path, disc_path

def load_rimone_data(root_dirs, img_size=(512, 512)):
    print(f"\nRIM-ONE DATA LOADING (STEREO CROPPING -> {img_size})")
    
    # Find all stereo images
    image_paths = []
    for base_root in root_dirs:
        for root, _, files in os.walk(base_root):
            if 'stereo' in root.lower() and 'image' in root.lower():
                for file in files:
                    if file.lower().endswith((".png", ".jpg", ".jpeg")):
                        image_paths.append(os.path.join(root, file))
    
    images, masks = [], []
    for img_path in sorted(list(set(image_paths))):
        cup_path, disc_path = find_rimone_mask_for_image(img_path)
        if not cup_path or not disc_path: continue
        
        img = cv2.imread(img_path)
        if img is None: continue
        
        cup = cv2.imread(cup_path, cv2.IMREAD_GRAYSCALE)
        disc = cv2.imread(disc_path, cv2.IMREAD_GRAYSCALE)
        if cup is None or disc is None: continue
        
        # Stereo Crop: The right part of the stereo image is disregarded
        # We ensure to only crop if the image width is much larger than the mask
        if img.shape[1] >= 1.8 * cup.shape[1]:
            img = img[:, :img.shape[1]//2]
            
        img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), img_size)
        img = img.astype(np.float32) / 255.0
        
        cup = cv2.resize(cup, img_size, interpolation=cv2.INTER_NEAREST)
        disc = cv2.resize(disc, img_size, interpolation=cv2.INTER_NEAREST)
        
        _, cup_bin = cv2.threshold(cup, 127, 255, cv2.THRESH_BINARY)
        _, disc_bin = cv2.threshold(disc, 127, 255, cv2.THRESH_BINARY)
        
        # RIM-ONE masks might be white background with black ROI. We invert if needed.
        if np.sum(cup_bin == 255) > np.sum(cup_bin == 0): cup_bin = 255 - cup_bin
        if np.sum(disc_bin == 255) > np.sum(disc_bin == 0): disc_bin = 255 - disc_bin
        
        mask = np.zeros(img_size, dtype=np.uint8)
        mask[disc_bin > 0] = 1
        mask[cup_bin > 0] = 2
        
        images.append(img)
        masks.append(mask)
        
    print(f"Loaded {len(images)} samples out of expected 159.")
    return np.array(images, dtype=np.float32), np.array(masks, dtype=np.uint8)

# ============================================================================
# AUGMENTATION
# ============================================================================
def setup_augmentations():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.10, rotate_limit=15, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.6),
    ])

def data_generator(images, masks, batch_size, augment=None):
    idxs = np.arange(len(images))
    while True:
        np.random.shuffle(idxs)
        for i in range(0, len(images), batch_size):
            batch_idxs = idxs[i:i+batch_size]
            batch_x, batch_y = [], []
            for idx in batch_idxs:
                img, mask = images[idx], masks[idx]
                if augment:
                    aug = augment(image=(img * 255).astype(np.uint8), mask=mask)
                    img_aug = aug['image'].astype(np.float32) / 255.0
                    mask_aug = aug['mask']
                else:
                    img_aug, mask_aug = img, mask
                batch_x.append(img_aug)
                batch_y.append(to_categorical(mask_aug, num_classes=3))
            yield np.stack(batch_x).astype(np.float32), np.stack(batch_y).astype(np.float32)

# ============================================================================
# ARCHITECTURE: Standard MobileNetV2-UNet
# ============================================================================
def build_mobilenet_unet(input_shape=(512,512,3), num_classes=3):
    inputs = layers.Input(input_shape, dtype='float32')
    backbone = MobileNetV2(input_tensor=inputs, weights='imagenet', include_top=False)
    
    skips = [
        backbone.get_layer('block_1_expand_relu').output,
        backbone.get_layer('block_3_expand_relu').output,
        backbone.get_layer('block_6_expand_relu').output,
        backbone.get_layer('block_13_expand_relu').output,
    ]
    bridge = backbone.output
    
    x = layers.UpSampling2D((2, 2))(bridge)
    x = layers.Concatenate()([x, skips[3]])
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Concatenate()([x, skips[2]])
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Concatenate()([x, skips[1]])
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Concatenate()([x, skips[0]])
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    outputs = layers.Conv2D(num_classes, (1, 1), activation='softmax', dtype='float32')(x)
    return models.Model(inputs, outputs)

# ============================================================================
# METRICS
# ============================================================================
def dice_coef_class(y_true, y_pred, class_index, smooth=1e-7):
    y_true_c = K.cast(K.equal(K.argmax(y_true, axis=-1), class_index), 'float32')
    y_pred_c = K.cast(K.equal(K.argmax(y_pred, axis=-1), class_index), 'float32')
    inter = K.sum(y_true_c * y_pred_c)
    return (2. * inter + smooth) / (K.sum(y_true_c) + K.sum(y_pred_c) + smooth)

def dice_class_1(y_true, y_pred): return dice_coef_class(y_true, y_pred, 1)
def dice_class_2(y_true, y_pred): return dice_coef_class(y_true, y_pred, 2)

def compute_and_print_metrics(y_true, y_pred_labels, classes=[1, 2], class_names=["Disc", "Cup"]):
    print("\n--- Detailed Metrics ---")
    metrics_dict = {}
    
    for c, name in zip(classes, class_names):
        true_c = (y_true == c).astype(int)
        pred_c = (y_pred_labels == c).astype(int)
        
        tp = np.sum(true_c * pred_c)
        fp = np.sum((1 - true_c) * pred_c)
        fn = np.sum(true_c * (1 - pred_c))
        
        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        f1 = 2 * precision * recall / (precision + recall + 1e-7)
        iou = tp / (tp + fp + fn + 1e-7)
        dice = 2 * tp / (2 * tp + fp + fn + 1e-7)
        
        metrics_dict[name] = {"Precision": precision, "Recall": recall, "F1-Score": f1, "Dice": dice, "IoU": iou}
        
        print(f"[{name}]")
        print(f"  Precision: {precision:.4f} | Recall: {recall:.4f} | F1-Score: {f1:.4f} | Dice: {dice:.4f} | IoU: {iou:.4f}")
        
    print("[Overall (Macro Avg)]")
    overall_strs = []
    for metric in ["Precision", "Recall", "F1-Score", "Dice", "IoU"]:
        avg_val = np.mean([metrics_dict[name][metric] for name in class_names])
        overall_strs.append(f"{metric}: {avg_val:.4f}")
    print("  " + " | ".join(overall_strs))
    print("------------------------")

# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    configure_gpu()
    roots = discover_rimone_roots()
    if not roots: raise ValueError("No RIM-ONE r3 data found!")
    
    images, masks = load_rimone_data(roots)
    X_temp, X_test, y_temp, y_test = train_test_split(images, masks, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)
    
    model = build_mobilenet_unet()
    model.compile(optimizer=optimizers.Adam(learning_rate=1e-3),
                  loss="categorical_crossentropy",
                  metrics=['accuracy', dice_class_1, dice_class_2])
    
    callbacks = [
        ModelCheckpoint('rimone_base_final.keras', save_best_only=True, monitor='val_loss'),
        ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3, min_lr=1e-7),
        EarlyStopping(patience=8, restore_best_weights=True)
    ]
    
    train_aug = setup_augmentations()
    
    print("\nTraining BASE model on RIM-ONE...")
    model.fit(
        data_generator(X_train, y_train, batch_size=4, augment=train_aug),
        validation_data=data_generator(X_val, y_val, batch_size=4, augment=None),
        steps_per_epoch=len(X_train)//4,
        validation_steps=len(X_val)//4,
        epochs=100, callbacks=callbacks, verbose=1
    )
    
    print("\nEvaluating...")
    pred = model.predict(X_test, batch_size=4, verbose=0)
    y_pred_labels = np.argmax(pred, axis=-1)
    
    compute_and_print_metrics(y_test, y_pred_labels, classes=[0, 1, 2], class_names=["Background", "Disc", "Cup"])
    print("Done!")
