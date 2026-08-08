"""
RIM-ONE r3 Final Phase 4: V5 BASE AUG + HIGH-RES ROI + MSCA/LBFR/PPM + TTA + PP + OFFLINE AUG
====================================================================
Dataset: RIM-ONE r3 (~100 train images, very small)
Strategy:
1. Stereo Image Cropping: Disregards right half of the 2144x1424 stereo images.
2. High-Res ROI Extraction: Zoom into disc area (1.2x padding)
3. Offline Augmentation (8x copies): Because 100 images is too few, we expand to ~900
   using the PROVEN V5 augmentation (no destructive GridDistortion).
4. Online Augmentation: Light augmentations during training.
5. Model: MobileNetV2-UNet + 12 attention modules (MSCA, LBFR, PPM).
6. Phase 4 Training: Cosine LR, Label Smoothing (0.05), Disc Focal α=1.3, 100 epochs.
7. Inference: 4-pass TTA + MBG-Net style Post-Processing.
"""

import os
import gc
import math
import warnings
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import albumentations as A
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, backend as K
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from PIL import Image
import cv2

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

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
    return False

# ============================================================================
# RIM-ONE DATA LOADING (STEREO CROP + HIGH-RES ROI)
# ============================================================================
def discover_rimone_roots():
    roots = []
    kaggle_candidates = [
        "/kaggle/input/datasets/rogmaverick/rim-one-r3/RIM-ONE r3",
    ]
    for p in kaggle_candidates:
        if os.path.exists(p):
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

def process_rimone_mask_highres(cup_path, disc_path):
    cup = cv2.imread(cup_path, cv2.IMREAD_GRAYSCALE)
    disc = cv2.imread(disc_path, cv2.IMREAD_GRAYSCALE)
    if cup is None or disc is None: return None
    if cup.shape != disc.shape:
        disc = cv2.resize(disc, (cup.shape[1], cup.shape[0]), interpolation=cv2.INTER_NEAREST)
        
    _, cup_bin = cv2.threshold(cup, 127, 255, cv2.THRESH_BINARY)
    _, disc_bin = cv2.threshold(disc, 127, 255, cv2.THRESH_BINARY)
    
    if np.sum(cup_bin == 255) > np.sum(cup_bin == 0): cup_bin = 255 - cup_bin
    if np.sum(disc_bin == 255) > np.sum(disc_bin == 0): disc_bin = 255 - disc_bin
        
    mask = np.zeros(cup.shape, dtype=np.uint8)
    mask[disc_bin > 0] = 1
    mask[cup_bin > 0] = 2
    return mask

def extract_roi(image, mask, padding_factor=1.2, roi_size=(512, 512)):
    roi_mask = ((mask == 1) | (mask == 2)).astype(np.uint8)
    if roi_mask.sum() == 0:
        h, w = image.shape[:2]
        return image.copy(), mask.copy(), (0, 0, h, w)
    coords = np.where(roi_mask > 0)
    y_min, y_max, x_min, x_max = coords[0].min(), coords[0].max(), coords[1].min(), coords[1].max()
    center_y, center_x = (y_min + y_max) // 2, (x_min + x_max) // 2
    radius = max(y_max - y_min, x_max - x_min) // 2
    h, w = image.shape[:2]
    roi_half = max(int(radius * padding_factor), 60)
    y1, y2 = max(0, center_y - roi_half), min(h, center_y + roi_half)
    x1, x2 = max(0, center_x - roi_half), min(w, center_x + roi_half)
    crop_h, crop_w = y2 - y1, x2 - x1
    if crop_h > crop_w:
        diff = crop_h - crop_w
        x1, x2 = max(0, x1 - diff // 2), min(w, x1 + crop_h)
        x1 = max(0, x2 - crop_h)
    elif crop_w > crop_h:
        diff = crop_w - crop_h
        y1, y2 = max(0, y1 - diff // 2), min(h, y1 + crop_w)
        y1 = max(0, y2 - crop_w)
    roi_image = image[y1:y2, x1:x2]
    roi_mask = mask[y1:y2, x1:x2]
    roi_image_resized = np.array(Image.fromarray((roi_image * 255).astype(np.uint8)).resize(roi_size, Image.BILINEAR)) / 255.0
    roi_mask_resized = np.array(Image.fromarray(roi_mask).resize(roi_size, Image.NEAREST))
    return roi_image_resized.astype(np.float32), roi_mask_resized.astype(np.uint8), (y1, x1, y2, x2)

def load_rimone_highres_roi(root_dirs, roi_size=(512, 512), padding_factor=1.2):
    print(f"\nRIM-ONE HIGH-RES ROI LOADING (Padding={padding_factor}x)")
    image_paths = []
    for base_root in root_dirs:
        for root, _, files in os.walk(base_root):
            if 'stereo' in root.lower() and 'image' in root.lower():
                for file in files:
                    if file.lower().endswith((".png", ".jpg", ".jpeg")):
                        image_paths.append(os.path.join(root, file))
    
    all_roi_images, all_roi_masks, all_full_masks = [], [], []
    for img_path in sorted(list(set(image_paths))):
        cup_path, disc_path = find_rimone_mask_for_image(img_path)
        if not cup_path or not disc_path: continue
        
        img = cv2.imread(img_path)
        if img is None: continue
        
        mask_highres = process_rimone_mask_highres(cup_path, disc_path)
        if mask_highres is None: continue
        
        # Stereo Crop (Left part only)
        if img.shape[1] >= 1.8 * mask_highres.shape[1]:
            img = img[:, :img.shape[1]//2]
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        
        if img.shape[:2] != mask_highres.shape[:2]:
            mask_highres = cv2.resize(mask_highres, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        roi_img, roi_msk, _ = extract_roi(img, mask_highres, padding_factor, roi_size)
        full_msk_512 = np.array(Image.fromarray(mask_highres).resize((512, 512), Image.NEAREST))
        all_roi_images.append(roi_img)
        all_roi_masks.append(roi_msk)
        all_full_masks.append(full_msk_512)
    
    print(f"Loaded {len(all_roi_images)} RIM-ONE samples.")
    return np.array(all_roi_images, dtype=np.float32), np.array(all_roi_masks, dtype=np.uint8), np.array(all_full_masks, dtype=np.uint8)

# ============================================================================
# AUGMENTATION (Proven V5 Augmentation for Offline Copies)
# ============================================================================
def setup_offline_augmentation():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.10, rotate_limit=20, border_mode=0, p=0.6),
        A.ElasticTransform(alpha=30, sigma=5, p=0.2),
        A.RandomBrightnessContrast(brightness_limit=0.20, contrast_limit=0.20, p=0.7),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
        A.GaussianBlur(blur_limit=3, p=0.3),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),
        A.GaussNoise(var_limit=(5.0, 25.0), p=0.2),
    ])

def offline_augment_dataset(images, masks, n_copies=8):
    aug_pipeline = setup_offline_augmentation()
    n_orig = len(images)
    print(f"\nOFFLINE AUGMENTATION: {n_orig} original images -> expanding by {n_copies}x")
    aug_images_list = list(images)
    aug_masks_list = list(masks)
    for i in range(n_orig):
        img_uint8 = (images[i] * 255).astype(np.uint8)
        msk = masks[i]
        for _ in range(n_copies):
            augmented = aug_pipeline(image=img_uint8, mask=msk)
            aug_images_list.append(augmented['image'].astype(np.float32) / 255.0)
            aug_masks_list.append(augmented['mask'])
    
    aug_images = np.array(aug_images_list, dtype=np.float32)
    aug_masks = np.array(aug_masks_list, dtype=np.uint8)
    print(f"✓ Expanded to {len(aug_images)} images.")
    return aug_images, aug_masks

def setup_online_augmentation():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.05, rotate_limit=10, p=0.5),
    ])

def smooth_labels(y_cat, smoothing=0.05):
    num_classes = y_cat.shape[-1]
    return y_cat * (1.0 - smoothing) + smoothing / num_classes

def data_generator(images, masks, batch_size, augment=None, label_smoothing=0.05):
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
                mask_cat = to_categorical(mask_aug, num_classes=3)
                if label_smoothing > 0:
                    mask_cat = smooth_labels(mask_cat, smoothing=label_smoothing)
                batch_x.append(img_aug)
                batch_y.append(mask_cat)
            yield np.stack(batch_x).astype(np.float32), np.stack(batch_y).astype(np.float32)

# ============================================================================
# ARCHITECTURE (MobileNetV2 + MSCA + LBFR + PPM)
# ============================================================================
def msca_block(x, channels, dilation_rates=[1, 2, 3, 5], block_name='msca'):
    multi_scale_features = []
    filters_per_branch = channels // len(dilation_rates)
    for rate in dilation_rates:
        branch = layers.Conv2D(filters_per_branch, (3, 3), padding='same', dilation_rate=rate, use_bias=False, name=f'{block_name}_dil{rate}_conv')(x)
        branch = layers.BatchNormalization()(branch)
        branch = layers.Activation('relu')(branch)
        multi_scale_features.append(branch)
    concat = layers.Concatenate()(multi_scale_features)
    fused = layers.Conv2D(channels, (1, 1), use_bias=False)(concat)
    fused = layers.BatchNormalization()(fused)
    fused = layers.Activation('relu')(fused)
    return layers.Add()([x, fused])

def lbfr_block(x, channels, reduction=16, block_name='lbfr'):
    avg_pool = layers.Reshape((1, 1, channels))(layers.GlobalAveragePooling2D()(x))
    max_pool = layers.Reshape((1, 1, channels))(layers.GlobalMaxPooling2D()(x))
    fc1 = layers.Conv2D(channels // reduction, (1, 1), use_bias=False)
    fc2 = layers.Conv2D(channels, (1, 1), use_bias=False)
    avg_out = fc2(layers.Activation('relu')(fc1(avg_pool)))
    max_out = fc2(layers.Activation('relu')(fc1(max_pool)))
    channel_att = layers.Activation('sigmoid')(layers.Add()([avg_out, max_out]))
    x_channel = layers.Multiply()([x, channel_att])
    avg_spatial = layers.Lambda(lambda x: K.mean(x, axis=-1, keepdims=True))(x_channel)
    max_spatial = layers.Lambda(lambda x: K.max(x, axis=-1, keepdims=True))(x_channel)
    spatial_concat = layers.Concatenate()([avg_spatial, max_spatial])
    spatial_att = layers.Activation('sigmoid')(layers.Conv2D(1, (7, 7), padding='same', use_bias=False)(spatial_concat))
    x_spatial = layers.Multiply()([x_channel, spatial_att])
    recalibrated = layers.BatchNormalization()(layers.Conv2D(channels, (1, 1), use_bias=False)(x_spatial))
    return layers.Add()([x, recalibrated])

def ppm_block(x, channels, pool_scales=[1, 2, 3, 6], block_name='ppm'):
    h, w = x.shape[1], x.shape[2]
    ppm_features = [x]
    filters_per_scale = channels // len(pool_scales)
    for scale in pool_scales:
        pool_h, pool_w = max(h // scale, 1), max(w // scale, 1)
        pooled = layers.AveragePooling2D((h//pool_h, w//pool_w), strides=(h//pool_h, w//pool_w))(x)
        conv = layers.Activation('relu')(layers.BatchNormalization()(layers.Conv2D(filters_per_scale, (1, 1), use_bias=False)(pooled)))
        upsampled = layers.Resizing(h, w, interpolation='bilinear')(conv)
        ppm_features.append(upsampled)
    concat = layers.Concatenate()(ppm_features)
    return layers.Activation('relu')(layers.BatchNormalization()(layers.Conv2D(channels, (3, 3), padding='same', use_bias=False)(concat)))

def enhanced_decoder_block(x, skip_connection, filters, block_name, dropout_rate=0.3):
    x = layers.UpSampling2D((2, 2))(x)
    if skip_connection.shape[-1] != x.shape[-1]:
        skip_connection = layers.Conv2D(x.shape[-1], (1, 1), padding='same')(skip_connection)
    x = layers.Concatenate()([x, skip_connection])
    x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    residual = x
    x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, residual])
    x = layers.Conv2D(filters, (1, 1), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = msca_block(x, filters, block_name=f'{block_name}_msca')
    x = lbfr_block(x, filters, block_name=f'{block_name}_lbfr')
    x = ppm_block(x, filters, block_name=f'{block_name}_ppm')
    return x

def build_unet(input_shape=(512, 512, 3), num_classes=3, dropout_rate=0.3):
    inputs = layers.Input(input_shape, dtype='float32')
    backbone = MobileNetV2(input_tensor=inputs, weights='imagenet', include_top=False)
    skip_1 = backbone.get_layer('block_1_expand_relu').output
    skip_2 = backbone.get_layer('block_3_expand_relu').output
    skip_3 = backbone.get_layer('block_6_expand_relu').output
    skip_4 = backbone.get_layer('block_13_expand_relu').output
    bridge = backbone.output
    bridge = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(bridge)
    bridge = layers.BatchNormalization()(bridge)
    bridge = layers.Dropout(dropout_rate)(bridge)
    x = enhanced_decoder_block(bridge, skip_4, 512, 'dec1', dropout_rate)
    x = enhanced_decoder_block(x, skip_3, 256, 'dec2', dropout_rate)
    x = enhanced_decoder_block(x, skip_2, 128, 'dec3', dropout_rate * 0.7)
    x = enhanced_decoder_block(x, skip_1, 64, 'dec4', dropout_rate * 0.5)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    outputs = layers.Conv2D(num_classes, (1, 1), activation='softmax', dtype='float32')(x)
    return models.Model(inputs, outputs)

# ============================================================================
# LOSS & METRICS
# ============================================================================
def focal_loss(y_true, y_pred, gamma=2.0, alpha=None, epsilon=1e-7):
    if alpha is None: alpha = [0.25, 1.3, 1.0] # Phase 4: disc alpha 1.3
    y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
    focal_loss_value = 0.0
    for c in range(3):
        y_true_c, y_pred_c = y_true[:, :, :, c], y_pred[:, :, :, c]
        bce = -(y_true_c * K.log(y_pred_c) + (1 - y_true_c) * K.log(1 - y_pred_c))
        pt = y_true_c * y_pred_c + (1 - y_true_c) * (1 - y_pred_c)
        focal_loss_value += alpha[c] * K.mean(K.pow(1 - pt, gamma) * bce)
    return focal_loss_value

def enhanced_iou_loss(y_true, y_pred, smooth=1e-7):
    eiou_loss_value = 0.0
    for c in range(3):
        y_true_c, y_pred_c = y_true[:, :, :, c], y_pred[:, :, :, c]
        inter = K.sum(y_true_c * y_pred_c, axis=[1, 2])
        union = K.sum(y_true_c, axis=[1, 2]) + K.sum(y_pred_c, axis=[1, 2]) - inter
        iou = (inter + smooth) / (union + smooth)
        eiou_loss_value += K.mean(1.0 - iou)
    return eiou_loss_value / 3.0

def combined_loss(y_true, y_pred):
    return focal_loss(y_true, y_pred) + enhanced_iou_loss(y_true, y_pred)

def dice_coef_class(y_true, y_pred, class_index, smooth=1e-7):
    y_true_c = K.cast(K.equal(K.argmax(y_true, axis=-1), class_index), 'float32')
    y_pred_c = K.cast(K.equal(K.argmax(y_pred, axis=-1), class_index), 'float32')
    inter = K.sum(y_true_c * y_pred_c)
    return (2. * inter + smooth) / (K.sum(y_true_c) + K.sum(y_pred_c) + smooth)

def dice_class_1(y_true, y_pred): return dice_coef_class(y_true, y_pred, 1)
def dice_class_2(y_true, y_pred): return dice_coef_class(y_true, y_pred, 2)

class CosineAnnealingWarmRestarts(Callback):
    def __init__(self, T_0=15, T_mult=2, eta_max=2e-4, eta_min=1e-7):
        super().__init__()
        self.T_0, self.T_mult, self.eta_max, self.eta_min = T_0, T_mult, eta_max, eta_min
        self.T_cur, self.T_i, self.cycle = 0, T_0, 0
    def on_epoch_begin(self, epoch, logs=None):
        lr = self.eta_min + (self.eta_max - self.eta_min) * (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
        if hasattr(self.model.optimizer.learning_rate, 'assign'):
            self.model.optimizer.learning_rate.assign(lr)
        else:
            self.model.optimizer.learning_rate = lr
        self.T_cur += 1
        if self.T_cur >= self.T_i:
            self.T_cur = 0
            self.T_i = int(self.T_i * self.T_mult)
            self.cycle += 1

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
# MAIN TRAINING & EVAL
# ============================================================================
if __name__ == "__main__":
    configure_gpu()
    roots = discover_rimone_roots()
    if not roots: raise ValueError("No RIM-ONE r3 data found!")
    
    # 1. Load Data
    X_roi, y_roi, y_full = load_rimone_highres_roi(roots, padding_factor=1.2)
    
    # Tiny dataset split
    X_temp, X_test, y_temp, y_test = train_test_split(X_roi, y_roi, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)
    
    # 2. Offline Augmentation for Train
    X_train_aug, y_train_aug = offline_augment_dataset(X_train, y_train, n_copies=8)
    
    # 3. Train
    model = build_unet()
    model.compile(optimizer=optimizers.Adam(learning_rate=2e-4),
                  loss=combined_loss,
                  metrics=['accuracy', dice_class_1, dice_class_2])
    
    callbacks = [
        ModelCheckpoint('rimone_phase4_best.keras', save_best_only=True, monitor='val_loss'),
        CosineAnnealingWarmRestarts(T_0=15, T_mult=2),
        EarlyStopping(patience=20, restore_best_weights=True)
    ]
    
    online_aug = setup_online_augmentation()
    
    print("\nTraining Phase 4 model on RIM-ONE...")
    model.fit(
        data_generator(X_train_aug, y_train_aug, batch_size=4, augment=online_aug, label_smoothing=0.05),
        validation_data=data_generator(X_val, y_val, batch_size=4, augment=None, label_smoothing=0.0),
        steps_per_epoch=len(X_train_aug)//4,
        validation_steps=len(X_val)//4,
        epochs=80, callbacks=callbacks, verbose=1
    )
    
    # 4. Predict
    print("\nEvaluating with TTA...")
    pred_orig = model.predict(X_test, batch_size=4, verbose=0)
    pred_hflip = np.flip(model.predict(np.flip(X_test, axis=2), batch_size=4, verbose=0), axis=2)
    pred_vflip = np.flip(model.predict(np.flip(X_test, axis=1), batch_size=4, verbose=0), axis=1)
    pred_hvflip = np.flip(np.flip(model.predict(np.flip(np.flip(X_test, axis=2), axis=1), batch_size=4, verbose=0), axis=2), axis=1)
    
    avg_pred = (pred_orig + pred_hflip + pred_vflip + pred_hvflip) / 4.0
    y_pred_labels = np.argmax(avg_pred, axis=-1)
    
    compute_and_print_metrics(y_test, y_pred_labels, classes=[0, 1, 2], class_names=["Background", "Disc", "Cup"])
    print("Done!")
