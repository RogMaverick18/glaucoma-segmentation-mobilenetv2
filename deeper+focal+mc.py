"""
COMBINED ADVANCED MOBILENET-UNET EXPERIMENT WITH VISUALIZATION
================================================================
Enhanced with:
1. Sample prediction visualization
2. Training history plots (loss, dice curves)
3. Comparative analysis tables
4. Parameter count comparison
5. Performance improvement metrics
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, backend as K
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import albumentations as A
import warnings
import gc  # For memory management in MC Dropout
warnings.filterwarnings('ignore')

# Set plot style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# [Previous functions remain the same: GPU config, data loading, augmentation]
# ============================================================================

def configure_gpu():
    """Configure GPU settings for optimal performance"""
    print("Configuring GPU settings...")
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            for gpu in physical_devices:
                tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.set_visible_devices(physical_devices[0], 'GPU')
            print(f"✓ GPU found: {physical_devices}")
            return True
        except RuntimeError as e:
            print(f"✗ GPU configuration error: {e}")
            return False
    else:
        print("✗ No GPU found, using CPU")
        return False

def load_images_masks(images_folder, masks_folder, img_size=(256,256)):
    """Load images and masks from folders"""
    images, masks = [], []
    image_files = sorted(os.listdir(images_folder))
    mask_files = sorted(os.listdir(masks_folder))
    mask_dict = {os.path.splitext(f)[0]: f for f in mask_files}
    
    for img_file in image_files:
        base = os.path.splitext(img_file)[0]
        if base in mask_dict:
            img = cv2.imread(os.path.join(images_folder, img_file))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, img_size) / 255.0
            
            mask = cv2.imread(os.path.join(masks_folder, mask_dict[base]), cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, img_size)
            unique_vals = np.unique(mask)
            if len(unique_vals) > 3 or np.max(unique_vals) > 2:
                mask = np.clip(mask // 85, 0, 2).astype(np.uint8)
            
            images.append(img)
            masks.append(mask)
    
    return np.array(images, dtype=np.float32), np.array(masks, dtype=np.uint8)

def load_all_data(root_dir):
    """Load all data from existing splits and merge"""
    all_images, all_masks = [], []
    for split in ["train", "val", "test"]:
        print(f"Loading {split} data...")
        imgs, msks = load_images_masks(
            os.path.join(root_dir, split, "Images"),
            os.path.join(root_dir, split, "Masks")
        )
        all_images.append(imgs)
        all_masks.append(msks)
        print(f"  {split}: {imgs.shape[0]} samples")
    
    all_images = np.concatenate(all_images, axis=0)
    all_masks = np.concatenate(all_masks, axis=0)
    return all_images, all_masks

def create_new_split(all_images, all_masks):
    """Create new split: 600 train, 200 val, 400 test"""
    X_temp, X_test, y_temp, y_test = train_test_split(
        all_images, all_masks, test_size=400, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=200, random_state=42
    )
    print(f"Split - Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def setup_augmentations():
    return A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.7),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.5),
        A.GaussianBlur(blur_limit=3, p=0.3),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
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
                    aug = augment(image=img, mask=mask)
                    img_aug, mask_aug = aug['image'], aug['mask']
                else:
                    img_aug, mask_aug = img, mask
                
                mask_cat = to_categorical(mask_aug, num_classes=3)
                batch_x.append(img_aug)
                batch_y.append(mask_cat)
            
            yield np.stack(batch_x).astype(np.float32), np.stack(batch_y).astype(np.float32)

# ============================================================================
# [Previous loss functions remain the same]
# ============================================================================

def focal_loss(y_true, y_pred, gamma=2.0, alpha=None, epsilon=1e-7):
    if alpha is None:
        alpha = [0.25, 1.0, 1.0]
    
    y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
    focal_loss_value = 0.0
    
    for c in range(3):
        y_true_c = y_true[:, :, :, c]
        y_pred_c = y_pred[:, :, :, c]
        pt = y_pred_c
        focal_weight = K.pow(1.0 - pt, gamma)
        cross_entropy = -K.log(pt)
        focal_loss_c = alpha[c] * focal_weight * cross_entropy * y_true_c
        focal_loss_value += K.mean(focal_loss_c)
    
    return focal_loss_value

def enhanced_iou_loss(y_true, y_pred, smooth=1e-7):
    eiou_loss_value = 0.0
    
    for c in range(3):
        y_true_c = y_true[:, :, :, c]
        y_pred_c = y_pred[:, :, :, c]
        
        intersection = K.sum(y_true_c * y_pred_c, axis=[1, 2])
        union = K.sum(y_true_c, axis=[1, 2]) + K.sum(y_pred_c, axis=[1, 2]) - intersection
        iou = (intersection + smooth) / (union + smooth)
        
        height = K.cast(K.shape(y_true_c)[1], 'float32')
        width = K.cast(K.shape(y_true_c)[2], 'float32')
        
        y_coords = K.arange(0, height)
        x_coords = K.arange(0, width)
        y_grid = K.tile(K.reshape(y_coords, (-1, 1)), (1, K.cast(width, 'int32')))
        x_grid = K.tile(K.reshape(x_coords, (1, -1)), (K.cast(height, 'int32'), 1))
        y_grid = K.cast(y_grid, 'float32')
        x_grid = K.cast(x_grid, 'float32')
        
        true_mass = K.sum(y_true_c, axis=[1, 2]) + smooth
        true_center_y = K.sum(y_true_c * y_grid, axis=[1, 2]) / true_mass
        true_center_x = K.sum(y_true_c * x_grid, axis=[1, 2]) / true_mass
        
        pred_mass = K.sum(y_pred_c, axis=[1, 2]) + smooth
        pred_center_y = K.sum(y_pred_c * y_grid, axis=[1, 2]) / pred_mass
        pred_center_x = K.sum(y_pred_c * x_grid, axis=[1, 2]) / pred_mass
        
        center_distance = K.square(true_center_y - pred_center_y) + K.square(true_center_x - pred_center_x)
        diagonal = K.square(height) + K.square(width)
        center_penalty = center_distance / (diagonal + smooth)
        
        true_width = K.sqrt(K.sum(K.sum(y_true_c, axis=1), axis=1) + smooth)
        true_height = K.sqrt(K.sum(K.sum(y_true_c, axis=2), axis=1) + smooth)
        pred_width = K.sqrt(K.sum(K.sum(y_pred_c, axis=1), axis=1) + smooth)
        pred_height = K.sqrt(K.sum(K.sum(y_pred_c, axis=2), axis=1) + smooth)
        
        width_diff = K.square(pred_width - true_width)
        height_diff = K.square(pred_height - true_height)
        aspect_penalty = width_diff / (K.square(width) + smooth) + height_diff / (K.square(height) + smooth)
        
        eiou_c = 1.0 - iou + center_penalty + aspect_penalty
        eiou_loss_value += K.mean(eiou_c)
    
    return eiou_loss_value / 3.0

def focal_eiou_combined_loss(y_true, y_pred, focal_weight=1.0, eiou_weight=1.0, gamma=2.0, alpha=None):
    fl = focal_loss(y_true, y_pred, gamma=gamma, alpha=alpha)
    eiou = enhanced_iou_loss(y_true, y_pred)
    return focal_weight * fl + eiou_weight * eiou

# ============================================================================
# [Previous model architectures remain the same]
# ============================================================================

def enhanced_decoder_block(x, skip_connection, filters, block_name, dropout_rate=0.3):
    """
    Enhanced decoder block with residual connections and dropout
    
    FIXED: Uses single dropout per block (after first conv+BN) for MC Dropout consistency
    - Dropout placement: After Conv1+BN1 (early in block)
    - Dropout rate: Configurable (0.3 for deep layers, reduced for shallow)
    - This matches MC_dropout.py implementation for proper uncertainty estimation
    """
    x = layers.UpSampling2D((2, 2), name=f'{block_name}_upsample')(x)
    
    if skip_connection.shape[-1] != x.shape[-1]:
        skip_connection = layers.Conv2D(x.shape[-1], (1, 1), activation='relu', 
                                      padding='same', name=f'{block_name}_skip_align')(skip_connection)
        skip_connection = layers.BatchNormalization(name=f'{block_name}_skip_bn')(skip_connection)
    
    x = layers.Concatenate(name=f'{block_name}_concat')([x, skip_connection])
    
    x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same', name=f'{block_name}_conv1')(x)
    x = layers.BatchNormalization(name=f'{block_name}_bn1')(x)
    # FIXED: Single dropout per block like MC_dropout.py for consistent MC Dropout
    x = layers.Dropout(dropout_rate, name=f'{block_name}_dropout')(x)
    
    x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same', name=f'{block_name}_conv2')(x)
    x = layers.BatchNormalization(name=f'{block_name}_bn2')(x)
    
    x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same', name=f'{block_name}_conv3')(x)
    x = layers.BatchNormalization(name=f'{block_name}_bn3')(x)
    
    residual = x
    x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same', name=f'{block_name}_conv4')(x)
    x = layers.BatchNormalization(name=f'{block_name}_bn4')(x)
    
    x = layers.Add(name=f'{block_name}_residual')([x, residual])
    x = layers.Conv2D(filters, (1, 1), activation='relu', padding='same', name=f'{block_name}_conv_final')(x)
    x = layers.BatchNormalization(name=f'{block_name}_bn_final')(x)
    
    return x

def build_combined_unet(input_shape=(256, 256, 3), num_classes=3, 
                       use_deeper=False, dropout_rate=0.3):
    inputs = layers.Input(input_shape, dtype='float32')
    backbone = MobileNetV2(input_tensor=inputs, weights='imagenet', include_top=False)
    
    skip_1 = backbone.get_layer('block_1_expand_relu').output
    skip_2 = backbone.get_layer('block_3_expand_relu').output
    skip_3 = backbone.get_layer('block_6_expand_relu').output
    skip_4 = backbone.get_layer('block_13_expand_relu').output
    bridge = backbone.output
    
    if use_deeper:
        bridge = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(bridge)
        bridge = layers.BatchNormalization()(bridge)
    
    if dropout_rate > 0:
        bridge = layers.Dropout(dropout_rate, name='bridge_dropout')(bridge)
    
    if use_deeper:
        # FIXED: Use consistent dropout rates like MC_dropout.py
        # decoder_1 and decoder_2: full dropout_rate (0.3)
        # decoder_3: 0.7x (0.21)
        # decoder_4: 0.5x (0.15)
        x = enhanced_decoder_block(bridge, skip_4, 512, 'decoder_1', dropout_rate)
        x = enhanced_decoder_block(x, skip_3, 256, 'decoder_2', dropout_rate)
        x = enhanced_decoder_block(x, skip_2, 128, 'decoder_3', dropout_rate * 0.7)
        x = enhanced_decoder_block(x, skip_1, 64, 'decoder_4', dropout_rate * 0.5)
    else:
        # Baseline decoder (non-deeper) with consistent dropout placement
        x = layers.UpSampling2D((2, 2))(bridge)
        x = layers.Concatenate()([x, skip_4])
        x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate)(x)  # After BN, before next conv
        x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Concatenate()([x, skip_3])
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate)(x)
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Concatenate()([x, skip_2])
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate * 0.7)(x)  # Reduced dropout
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Concatenate()([x, skip_1])
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate * 0.5)(x)  # Further reduced
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
    
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    outputs = layers.Conv2D(num_classes, (1, 1), activation='softmax', dtype='float32')(x)
    
    model = models.Model(inputs, outputs)
    
    # Print dropout configuration for MC Dropout
    if dropout_rate > 0:
        dropout_layers = [layer for layer in model.layers if isinstance(layer, layers.Dropout)]
        print(f"\nMC Dropout Configuration:")
        print(f"  Total Dropout Layers: {len(dropout_layers)}")
        print(f"  Dropout Rates: {[f'{layer.rate:.2f}' for layer in dropout_layers]}")
        print(f"  Deeper Architecture: {use_deeper}")
    
    return model

# ============================================================================
# [Previous metrics remain the same]
# ============================================================================

def dice_coef_multiclass(y_true, y_pred, smooth=1e-7):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_class(y_true, y_pred, class_index, smooth=1e-7):
    y_true_class = K.cast(K.equal(K.argmax(y_true, axis=-1), class_index), 'float32')
    y_pred_class = K.cast(K.equal(K.argmax(y_pred, axis=-1), class_index), 'float32')
    intersection = K.sum(y_true_class * y_pred_class)
    return (2. * intersection + smooth) / (K.sum(y_true_class) + K.sum(y_pred_class) + smooth)

def dice_class_0(y_true, y_pred): return dice_coef_class(y_true, y_pred, 0)
def dice_class_1(y_true, y_pred): return dice_coef_class(y_true, y_pred, 1)
def dice_class_2(y_true, y_pred): return dice_coef_class(y_true, y_pred, 2)

def iou_coef_multiclass(y_true, y_pred, smooth=1e-7):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

def iou_coef_class(y_true, y_pred, class_index, smooth=1e-7):
    y_true_class = K.cast(K.equal(K.argmax(y_true, axis=-1), class_index), 'float32')
    y_pred_class = K.cast(K.equal(K.argmax(y_pred, axis=-1), class_index), 'float32')
    intersection = K.sum(y_true_class * y_pred_class)
    union = K.sum(y_true_class) + K.sum(y_pred_class) - intersection
    return (intersection + smooth) / (union + smooth)

def iou_class_0(y_true, y_pred): return iou_coef_class(y_true, y_pred, 0)
def iou_class_1(y_true, y_pred): return iou_coef_class(y_true, y_pred, 1)
def iou_class_2(y_true, y_pred): return iou_coef_class(y_true, y_pred, 2)

def mc_dropout_predict(model, X, num_samples=15, batch_size=4, verbose=True):
    """
    MC Dropout prediction using training=True flag
    
    This enables dropout layers during inference to get stochastic predictions
    for uncertainty estimation (following Gal & Ghahramani 2016)
    """
    if verbose:
        print(f"\nRunning MC Dropout with {num_samples} stochastic forward passes...")
    
    predictions = []
    
    # Process in batches to avoid memory issues
    num_batches = int(np.ceil(len(X) / batch_size))
    
    for i in range(num_samples):
        if verbose and (i + 1) % 5 == 0:
            print(f"  Completed: {i+1}/{num_samples} forward passes")
        
        batch_preds = []
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(X))
            X_batch = X[start_idx:end_idx]
            
            # Call model with training=True to enable dropout
            pred_batch = model(X_batch, training=True)
            batch_preds.append(pred_batch.numpy())
        
        # Concatenate batch predictions
        pred = np.concatenate(batch_preds, axis=0)
        predictions.append(pred)
        
        # Clear memory periodically
        if (i + 1) % 5 == 0:
            gc.collect()
    
    # Stack predictions: shape (num_samples, batch, height, width, classes)
    all_preds = np.array(predictions)
    
    # Calculate mean and standard deviation across samples
    mean_pred = np.mean(all_preds, axis=0)
    std_pred = np.std(all_preds, axis=0)
    
    # Clear memory
    del all_preds
    gc.collect()
    
    if verbose:
        print(f"✓ MC Dropout completed")
        print(f"  Mean prediction shape: {mean_pred.shape}")
        print(f"  Std prediction shape: {std_pred.shape}")
    
    return mean_pred, std_pred

# ============================================================================
# NEW: VISUALIZATION FUNCTIONS
# ============================================================================

def plot_sample_predictions(model, X_test, y_test, model_name, num_samples=6, save_dir='results'):
    """Visualize sample predictions with original image, GT, segmented mask, and overlay"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Get predictions
    y_pred = model.predict(X_test[:num_samples], verbose=0, batch_size=4)
    y_pred_labels = np.argmax(y_pred, axis=-1)
    
    # Color map for classes
    colors = {
        0: [0, 0, 0],        # Background - Black
        1: [0, 255, 0],      # Disc - Green
        2: [255, 0, 0]       # Cup - Red
    }
    
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
    
    for i in range(num_samples):
        # Original image
        axes[i, 0].imshow(X_test[i])
        axes[i, 0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[i, 0].axis('off')
        
        # Ground truth mask (colored)
        gt_colored = np.zeros((256, 256, 3), dtype=np.uint8)
        for class_id, color in colors.items():
            gt_colored[y_test[i] == class_id] = color
        axes[i, 1].imshow(gt_colored)
        axes[i, 1].set_title('Ground Truth (GT)', fontsize=12, fontweight='bold')
        axes[i, 1].axis('off')
        
        # Prediction mask (colored) - Segmented Mask
        pred_colored = np.zeros((256, 256, 3), dtype=np.uint8)
        for class_id, color in colors.items():
            pred_colored[y_pred_labels[i] == class_id] = color
        axes[i, 2].imshow(pred_colored)
        axes[i, 2].set_title('Segmented Mask', fontsize=12, fontweight='bold')
        axes[i, 2].axis('off')
        
        # Overlay - prediction on original image
        overlay = X_test[i].copy()
        for class_id, color in colors.items():
            if class_id > 0:  # Skip background
                mask = (y_pred_labels[i] == class_id).astype(np.float32)
                mask_rgb = np.stack([mask * color[0]/255, mask * color[1]/255, mask * color[2]/255], axis=-1)
                overlay = overlay * 0.6 + mask_rgb * 0.4
        axes[i, 3].imshow(overlay)
        axes[i, 3].set_title('Overlay', fontsize=12, fontweight='bold')
        axes[i, 3].axis('off')
        
        # Calculate dice for this sample
        y_test_cat = to_categorical(y_test[i], num_classes=3)
        y_pred_cat = to_categorical(y_pred_labels[i], num_classes=3)
        
        disc_dice = 2 * np.sum(y_test_cat[:,:,1] * y_pred_cat[:,:,1]) / (np.sum(y_test_cat[:,:,1]) + np.sum(y_pred_cat[:,:,1]) + 1e-7)
        cup_dice = 2 * np.sum(y_test_cat[:,:,2] * y_pred_cat[:,:,2]) / (np.sum(y_test_cat[:,:,2]) + np.sum(y_pred_cat[:,:,2]) + 1e-7)
        
        axes[i, 0].text(0.5, -0.1, f'Sample {i+1}\nDisc: {disc_dice:.3f} | Cup: {cup_dice:.3f}', 
                       transform=axes[i, 0].transAxes, ha='center', fontsize=10, 
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(f'{model_name} - Predictions: Original | GT | Segmented Mask | Overlay', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/{model_name}_predictions.png', dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {save_dir}/{model_name}_predictions.png")
    plt.show()
    plt.close()

def plot_training_history(history, model_name, save_dir='results'):
    """Plot training history (loss and dice curves)"""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss curves
    axes[0, 0].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[0, 0].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[0, 0].set_title('Loss Curves', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Overall Dice
    axes[0, 1].plot(history.history['dice_coef_multiclass'], label='Train Dice', linewidth=2)
    axes[0, 1].plot(history.history['val_dice_coef_multiclass'], label='Val Dice', linewidth=2)
    axes[0, 1].set_title('Overall Dice Coefficient', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Dice', fontsize=12)
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Disc Dice (Class 1)
    axes[1, 0].plot(history.history['dice_class_1'], label='Train Disc Dice', linewidth=2, color='green')
    axes[1, 0].plot(history.history['val_dice_class_1'], label='Val Disc Dice', linewidth=2, color='darkgreen')
    axes[1, 0].set_title('Disc Segmentation (Class 1)', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Dice', fontsize=12)
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Cup Dice (Class 2)
    axes[1, 1].plot(history.history['dice_class_2'], label='Train Cup Dice', linewidth=2, color='red')
    axes[1, 1].plot(history.history['val_dice_class_2'], label='Val Cup Dice', linewidth=2, color='darkred')
    axes[1, 1].set_title('Cup Segmentation (Class 2)', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Dice', fontsize=12)
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(f'{model_name} - Training History', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/{model_name}_history.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {save_dir}/{model_name}_history.png")

def plot_training_curves(history, model_name, save_dir='results'):
    """Plot training loss and accuracy curves"""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    axes[0].plot(history.history['loss'], label='Training Loss', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0].set_title(f'{model_name} - Loss Over Epochs', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Plot accuracy
    axes[1].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[1].set_title(f'{model_name} - Accuracy Over Epochs', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{model_name}_training_curves.png", dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {save_dir}/{model_name}_training_curves.png")
    plt.show()
    plt.close()

def plot_confusion_matrices(y_true, y_pred, model_name, save_dir='results'):
    """Plot confusion matrix and normalized confusion matrix"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert to class labels
    y_true_labels = np.argmax(y_true, axis=-1).flatten()
    y_pred_labels = np.argmax(y_pred, axis=-1).flatten()
    
    # Compute confusion matrices
    cm = confusion_matrix(y_true_labels, y_pred_labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    class_names = ['Background', 'Disc', 'Cup']
    
    # Plot regular confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[0], cbar_kws={'label': 'Count'})
    axes[0].set_title(f'{model_name}\nConfusion Matrix', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('True Label', fontsize=12)
    axes[0].set_xlabel('Predicted Label', fontsize=12)
    
    # Plot normalized confusion matrix
    sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Greens',
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[1], cbar_kws={'label': 'Proportion'})
    axes[1].set_title(f'{model_name}\nNormalized Confusion Matrix', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('True Label', fontsize=12)
    axes[1].set_xlabel('Predicted Label', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{model_name}_confusion_matrices.png", dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {save_dir}/{model_name}_confusion_matrices.png")
    plt.show()
    plt.close()
    
    # Print metrics
    print(f"\nConfusion Matrix:")
    print(cm)
    print(f"\nNormalized Confusion Matrix:")
    print(np.round(cm_normalized, 3))
    
    return cm, cm_normalized

def create_comparison_table(all_results, all_params, save_dir='results'):
    """Create comprehensive comparison table with improvements"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Prepare data
    data = []
    baseline_disc = None
    baseline_cup = None
    
    for model_name, results in all_results.items():
        disc_dice = results['standard']['Disc']['dice']
        cup_dice = results['standard']['Cup']['dice']
        overall = (disc_dice + cup_dice) / 2
        
        if 'Baseline' in model_name:
            baseline_disc = disc_dice
            baseline_cup = cup_dice
        
        disc_improvement = ((disc_dice - baseline_disc) / baseline_disc * 100) if baseline_disc else 0
        cup_improvement = ((cup_dice - baseline_cup) / baseline_cup * 100) if baseline_cup else 0
        
        params = all_params.get(model_name, 0)
        
        data.append({
            'Model': model_name,
            'Parameters': f'{params:,}',
            'Disc Dice': f'{disc_dice:.4f}',
            'Disc Δ%': f'{disc_improvement:+.2f}%' if baseline_disc else '-',
            'Cup Dice': f'{cup_dice:.4f}',
            'Cup Δ%': f'{cup_improvement:+.2f}%' if baseline_cup else '-',
            'Overall': f'{overall:.4f}'
        })
    
    df = pd.DataFrame(data)
    
    # Plot table
    fig, ax = plt.subplots(figsize=(16, len(df)*0.8 + 2))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center',
                    colWidths=[0.20, 0.12, 0.10, 0.08, 0.10, 0.08, 0.10])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style rows
    for i in range(1, len(df) + 1):
        if i % 2 == 0:
            for j in range(len(df.columns)):
                table[(i, j)].set_facecolor('#f0f0f0')
        
        # Highlight best model
        if 'FULL' in df.iloc[i-1]['Model']:
            for j in range(len(df.columns)):
                table[(i, j)].set_facecolor('#FFEB3B')
                table[(i, j)].set_text_props(weight='bold')
    
    plt.title('Comprehensive Model Comparison\n', fontsize=16, fontweight='bold', pad=20)
    plt.savefig(f'{save_dir}/comparison_table.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {save_dir}/comparison_table.png")
    
    return df

def plot_comparative_bar_chart(all_results, save_dir='results'):
    """Create comparative bar chart"""
    os.makedirs(save_dir, exist_ok=True)
    
    models = []
    disc_scores = []
    cup_scores = []
    
    for model_name, results in all_results.items():
        models.append(model_name.replace('1. ', '').replace('2. ', '').replace('3. ', '')
                     .replace('4. ', '').replace('5. ', '').replace('6. ', ''))
        disc_scores.append(results['standard']['Disc']['dice'])
        cup_scores.append(results['standard']['Cup']['dice'])
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(14, 6))
    bars1 = ax.bar(x - width/2, disc_scores, width, label='Disc Dice', color='#4CAF50')
    bars2 = ax.bar(x + width/2, cup_scores, width, label='Cup Dice', color='#FF5722')
    
    ax.set_xlabel('Model Configuration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Dice Coefficient', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/comparison_barchart.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {save_dir}/comparison_barchart.png")

# ============================================================================
# MODIFIED TRAINING FUNCTION WITH VISUALIZATION
# ============================================================================

def train_model(X_train, y_train, X_val, y_val, X_test, y_test, model_name, 
               use_focal_eiou=False, use_deeper=False, dropout_rate=0.0,
               augment=None, save_dir='results'):
    """Train model with visualization"""
    print(f"\n{'='*80}")
    print(f"TRAINING: {model_name}")
    print(f"{'='*80}")
    print(f"  Focal-EIoU Loss: {'✓' if use_focal_eiou else '✗'}")
    print(f"  Deeper Decoder: {'✓' if use_deeper else '✗'}")
    print(f"  MC Dropout: {'✓ (rate=' + str(dropout_rate) + ')' if dropout_rate > 0 else '✗'}")
    
    K.clear_session()
    
    # Build model
    model = build_combined_unet(use_deeper=use_deeper, dropout_rate=dropout_rate)
    
    # Get parameter count
    param_count = model.count_params()
    print(f"  Total Parameters: {param_count:,}")
    
    # Choose loss
    if use_focal_eiou:
        loss_fn = focal_eiou_combined_loss
    else:
        loss_fn = "categorical_crossentropy"
    
    # Compile
    initial_lr = 5e-4 if use_deeper else 1e-3
    model.compile(
        optimizer=optimizers.Adam(learning_rate=initial_lr),
        loss=loss_fn,
        metrics=[dice_coef_multiclass, dice_class_0, dice_class_1, dice_class_2,
                "accuracy", iou_coef_multiclass, iou_class_0, iou_class_1, iou_class_2]
    )
    
    # Callbacks
    checkpoint = ModelCheckpoint(f"{save_dir}/{model_name}.keras", save_best_only=True, verbose=1, 
                                monitor='val_loss', mode='min')
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3, 
                                  verbose=1, mode='min', min_lr=1e-7)
    early_stop = EarlyStopping(monitor="val_loss", patience=10, verbose=1, 
                              restore_best_weights=True, mode='min')
    
    # Training
    batch_size = 4 if use_deeper else 8
    steps_train = X_train.shape[0] // batch_size
    steps_val = X_val.shape[0] // batch_size
    
    try:
        history = model.fit(
            data_generator(X_train, y_train, batch_size, augment),
            validation_data=data_generator(X_val, y_val, batch_size, None),
            steps_per_epoch=steps_train,
            validation_steps=steps_val,
            epochs=50,
            callbacks=[checkpoint, reduce_lr, early_stop],
            verbose=1
        )
    except Exception as e:
        print(f"✗ Error: {e}. Retrying with smaller batch...")
        batch_size = 2 if use_deeper else 4
        steps_train = X_train.shape[0] // batch_size
        steps_val = X_val.shape[0] // batch_size
        
        history = model.fit(
            data_generator(X_train, y_train, batch_size, augment),
            validation_data=data_generator(X_val, y_val, batch_size, None),
            steps_per_epoch=steps_train,
            validation_steps=steps_val,
            epochs=50,
            callbacks=[checkpoint, reduce_lr, early_stop],
            verbose=1
        )
    
    print(f"✓ Training completed")
    
    # Visualizations
    print("\nGenerating visualizations...")
    plot_training_curves(history, model_name, save_dir)
    plot_training_history(history, model_name, save_dir)
    plot_sample_predictions(model, X_test, y_test, model_name, num_samples=6, save_dir=save_dir)
    
    return model, history, param_count

def evaluate_model(model, X_test, y_test, model_name, run_mc_dropout=False):
    """Evaluate model"""
    print(f"\n{'='*80}")
    print(f"EVALUATION: {model_name}")
    print(f"{'='*80}")
    
    y_test_cat = np.array([to_categorical(mask, num_classes=3) for mask in y_test])
    
    # Standard evaluation
    test_metrics = model.evaluate(X_test, y_test_cat, verbose=0, batch_size=8)
    y_pred = model.predict(X_test, verbose=0, batch_size=8)
    
    # Plot confusion matrices
    plot_confusion_matrices(y_test_cat, y_pred, model_name, save_dir='results')
    
    # Calculate precision, recall, F1 score
    y_true_labels = np.argmax(y_test_cat, axis=-1).flatten()
    y_pred_labels = np.argmax(y_pred, axis=-1).flatten()
    
    precision = precision_score(y_true_labels, y_pred_labels, average=None, zero_division=0)
    recall = recall_score(y_true_labels, y_pred_labels, average=None, zero_division=0)
    f1 = f1_score(y_true_labels, y_pred_labels, average=None, zero_division=0)
    
    print(f"\nPrecision, Recall, F1 Score per class:")
    class_names = ["Background", "Disc", "Cup"]
    for idx, cname in enumerate(class_names):
        print(f"  {cname}: Precision={precision[idx]:.4f}, Recall={recall[idx]:.4f}, F1={f1[idx]:.4f}")
    
    print(f"\nMacro-averaged metrics:")
    print(f"  Precision: {np.mean(precision):.4f}")
    print(f"  Recall: {np.mean(recall):.4f}")
    print(f"  F1 Score: {np.mean(f1):.4f}")
    
    # Per-class metrics
    class_names = ["Background", "Disc", "Cup"]
    results = {'standard': {}}
    
    for cidx, cname in enumerate(class_names):
        y_true_c = (np.argmax(y_test_cat, axis=-1) == cidx).astype(np.float32)
        y_pred_c = (np.argmax(y_pred, axis=-1) == cidx).astype(np.float32)
        
        intersection = np.sum(y_true_c * y_pred_c)
        dice = (2. * intersection + 1e-7) / (np.sum(y_true_c) + np.sum(y_pred_c) + 1e-7)
        union = np.sum(y_true_c) + np.sum(y_pred_c) - intersection
        iou = (intersection + 1e-7) / (union + 1e-7)
        
        results['standard'][cname] = {'dice': dice, 'iou': iou}
        print(f"  {cname}: Dice={dice:.4f}, IoU={iou:.4f}")
    
    # MC Dropout
    if run_mc_dropout:
        print("\nMC Dropout Evaluation:")
        mean_pred, std_pred = mc_dropout_predict(model, X_test[:100], num_samples=10, verbose=True)
        results['mc_dropout'] = {}
        for cidx, cname in enumerate(class_names):
            mean_std = np.mean(std_pred[:, :, :, cidx])
            results['mc_dropout'][cname] = {'mean_uncertainty': mean_std}
            print(f"  {cname} uncertainty: {mean_std:.4f}")
    
    return results

# ============================================================================
# MAIN EXECUTION WITH FULL VISUALIZATION
# ============================================================================

if __name__ == "__main__":
    # Setup
    gpu_available = configure_gpu()
    root_dir = "/kaggle/input/refuge/REFUGE/"
    train_aug = setup_augmentations()
    save_dir = "results"
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n{'='*100}")
    print(f"{'COMBINED ADVANCED MOBILENET-UNET EXPERIMENT WITH VISUALIZATION':^100s}")
    print(f"{'='*100}")
    
    # Load data
    all_images, all_masks = load_all_data(root_dir)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = create_new_split(all_images, all_masks)
    
    # Storage
    all_results = {}
    all_params = {}
    all_histories = {}
    
    # ========================================================================
    # EXPERIMENT 1: Baseline
    # ========================================================================
    model, history, params = train_model(
        X_train, y_train, X_val, y_val, X_test, y_test,
        "1_Baseline",
        use_focal_eiou=False, use_deeper=False, dropout_rate=0.0,
        augment=train_aug, save_dir=save_dir
    )
    all_results['1. Baseline'] = evaluate_model(model, X_test, y_test, "Baseline")
    all_params['1. Baseline'] = params
    all_histories['1. Baseline'] = history
    
    # ========================================================================
    # EXPERIMENT 2: Deeper + MC Dropout 
    # ========================================================================
    model, history, params = train_model(
        X_train, y_train, X_val, y_val, X_test, y_test,
        "2_Deeper_MC_Dropout",
        use_focal_eiou=False, use_deeper=True, dropout_rate=0.3,
        augment=train_aug, save_dir=save_dir
    )
    all_results['2. Deeper + MC Dropout'] = evaluate_model(model, X_test, y_test, "Deeper + MC Dropout", run_mc_dropout=True)
    all_params['2. Deeper + MC Dropout'] = params
    all_histories['2. Deeper + MC Dropout'] = history
    
    # ========================================================================
    # EXPERIMENT 3: Focal-EIoU + Deeper Decoder
    # ========================================================================
    model, history, params = train_model(
        X_train, y_train, X_val, y_val, X_test, y_test,
        "3_Focal_EIoU_Deeper",
        use_focal_eiou=True, use_deeper=True, dropout_rate=0.0,
        augment=train_aug, save_dir=save_dir
    )
    all_results['3. Focal-EIoU + Deeper'] = evaluate_model(model, X_test, y_test, "Focal-EIoU + Deeper")
    all_params['3. Focal-EIoU + Deeper'] = params
    all_histories['3. Focal-EIoU + Deeper'] = history
    
    # ========================================================================
    # EXPERIMENT 4: FULL COMBINATION
    # ========================================================================
    model, history, params = train_model(
        X_train, y_train, X_val, y_val, X_test, y_test,
        "4_FULL_Combination",
        use_focal_eiou=True, use_deeper=True, dropout_rate=0.3,
        augment=train_aug, save_dir=save_dir
    )
    all_results['4. FULL (MC+Focal+Deeper)'] = evaluate_model(model, X_test, y_test, "Full Combination", run_mc_dropout=True)
    all_params['4. FULL (MC+Focal+Deeper)'] = params
    all_histories['4. FULL (MC+Focal+Deeper)'] = history
    
    # ========================================================================
    # FINAL VISUALIZATIONS
    # ========================================================================
    print(f"\n{'='*100}")
    print(f"{'GENERATING FINAL COMPARISON VISUALIZATIONS':^100s}")
    print(f"{'='*100}")
    
    # Comparison table
    comparison_df = create_comparison_table(all_results, all_params, save_dir)
    print("\n" + comparison_df.to_string(index=False))
    
    # Bar chart
    plot_comparative_bar_chart(all_results, save_dir)
    
    print(f"\n{'='*100}")
    print(f"{'✓ ALL EXPERIMENTS COMPLETED!':^100s}")
    print(f"{'='*100}")
    print(f"\nAll results saved in: {save_dir}/")
    print(f"  - Training history plots: *_history.png")
    print(f"  - Sample predictions: *_predictions.png")
    print(f"  - Comparison table: comparison_table.png")
    print(f"  - Bar chart: comparison_barchart.png")
    print(f"  - Saved models: *.h5")
    print(f"\n{'='*100}\n")