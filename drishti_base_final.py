"""
Drishti-GS Final Base: MobileNetV2-UNet (Full Image)
====================================================================
Dataset: Drishti-GS
Strategy:
1. Standard MobileNetV2-UNet (No MSCA/LBFR/PPM).
2. Full image resizing to 512x512 (No high-res ROI extraction).
3. Standard Augmentation (Online only).
4. Standard Training: Adam 1e-3, ReduceLROnPlateau, 100 epochs.
5. Standard Inference (No TTA).
"""

import os
import re
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
# DRISHTI DATA LOADING (FULL IMAGE)
# ============================================================================
def _has_image_files(path, max_check=400):
    checked = 0
    for root, _, files in os.walk(path):
        for f in files:
            checked += 1
            if f.lower().endswith((".png", ".jpg", ".jpeg")): return True
            if checked >= max_check: return False
    return False

def discover_drishti_roots():
    roots = []
    kaggle_candidates = [
        "/kaggle/input/datasets/lokeshsaipureddi/drishtigs-retina-dataset-for-onh-segmentation",
        "/kaggle/input/datasets/lokeshsaipureddi/drishtigs-retina-dataset-for-onh-segmentation/Training",
        "/kaggle/input/datasets/lokeshsaipureddi/drishtigs-retina-dataset-for-onh-segmentation/Test",
    ]
    for p in kaggle_candidates:
        if os.path.exists(p):
            roots.append(os.path.abspath(p))
    unique = []
    seen = set()
    for p in roots:
        norm = os.path.normpath(p)
        if norm not in seen and _has_image_files(p):
            unique.append(p)
            seen.add(norm)
    return unique

def find_mask_for_image(img_path, root_dirs):
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    match = re.search(r'(\d+)', img_name)
    img_id = match.group(1) if match else img_name
    cup_path, disc_path = None, None
    for base_root in root_dirs:
        for root, _, files in os.walk(base_root):
            root_lower = root.lower()
            if 'gt' not in root_lower and 'groundtruth' not in root_lower: continue
            for f in files:
                f_lower = f.lower()
                if img_id in f or img_name.lower() in f_lower:
                    full_path = os.path.join(root, f)
                    if ('cup' in f_lower or 'oc' in f_lower) and cup_path is None: cup_path = full_path
                    elif ('disc' in f_lower or 'od' in f_lower) and disc_path is None: disc_path = full_path
            if cup_path and disc_path: return cup_path, disc_path
    return cup_path, disc_path

def load_drishti_data(root_dirs, img_size=(512, 512)):
    print(f"\nDRISHTI DATA LOADING (FULL IMAGE {img_size})")
    image_paths = []
    for base_root in root_dirs:
        for root, _, files in os.walk(base_root):
            if 'image' not in root.lower(): continue
            for file in files:
                if file.lower().endswith((".png", ".jpg", ".jpeg")):
                    if not any(x in file.lower() for x in ['seg', 'map', 'gt', 'cup', 'disc', 'od', 'oc', 'mask']):
                        image_paths.append(os.path.join(root, file))
    
    images, masks = [], []
    for img_path in sorted(list(set(image_paths))):
        cup_path, disc_path = find_mask_for_image(img_path, root_dirs)
        if not cup_path or not disc_path: continue
        
        img = cv2.imread(img_path)
        if img is None: continue
        img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), img_size)
        img = img.astype(np.float32) / 255.0
        
        cup = cv2.imread(cup_path, cv2.IMREAD_GRAYSCALE)
        disc = cv2.imread(disc_path, cv2.IMREAD_GRAYSCALE)
        if cup is None or disc is None: continue
        
        cup = cv2.resize(cup, img_size, interpolation=cv2.INTER_NEAREST)
        disc = cv2.resize(disc, img_size, interpolation=cv2.INTER_NEAREST)
        
        _, cup_bin = cv2.threshold(cup, 127, 255, cv2.THRESH_BINARY)
        _, disc_bin = cv2.threshold(disc, 127, 255, cv2.THRESH_BINARY)
        
        mask = np.zeros(img_size, dtype=np.uint8)
        mask[disc_bin > 0] = 1
        mask[cup_bin > 0] = 2
        
        images.append(img)
        masks.append(mask)
        
    print(f"Loaded {len(images)} samples.")
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
    roots = discover_drishti_roots()
    if not roots: raise ValueError("No Drishti data found!")
    
    images, masks = load_drishti_data(roots)
    X_temp, X_test, y_temp, y_test = train_test_split(images, masks, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)
    
    model = build_mobilenet_unet()
    model.compile(optimizer=optimizers.Adam(learning_rate=1e-3),
                  loss="categorical_crossentropy",
                  metrics=['accuracy', dice_class_1, dice_class_2])
    
    callbacks = [
        ModelCheckpoint('drishti_base_final.keras', save_best_only=True, monitor='val_loss'),
        ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3, min_lr=1e-7),
        EarlyStopping(patience=8, restore_best_weights=True)
    ]
    
    train_aug = setup_augmentations()
    
    print("\nTraining BASE model on Drishti...")
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
