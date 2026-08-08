"""
MOBILENET-UNET WITH MSCA + LBFR + PPM + GLAUCOMA PROGRESSION SCORING
====================================================================
Enhanced with three complementary modules from ODFormer/PSPNet for comprehensive feature enhancement
Plus glaucoma progression analysis with CDR-based severity classification

Triple Enhancement Strategy:
- MSCA (Multi-scale Context Aggregator): Multi-scale dilated convolutions [1,2,3,5]
- LBFR (Lightweight Bidirectional Feature Recalibrator): Channel + Spatial attention
- PPM (Pyramid Pooling Module): Multi-scale global context aggregation [1×1, 2×2, 3×3, 6×6]
- Sequential application: MSCA → LBFR → PPM on each decoder block
- Combines local multi-scale context + attention + global context

Glaucoma Scoring:
- CDR-based severity classification: Normal, Suspect, Moderate, Critical
- Area-based CDR calculation (robust & clinically relevant)
- Progression analysis with accuracy metrics
- Manual review recommendations for high-risk cases

Configuration:
- Base: Deeper Decoder + Focal-EIoU + MC Dropout
- Enhancement: MSCA + LBFR + PPM on decoder outputs
- Placement: All three modules after each decoder block (4 total)
- Total Attention Blocks: 12 (4 MSCA + 4 LBFR + 4 PPM)
"""

import os
import gc
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import albumentations as A
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, backend as K
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from PIL import Image

warnings.filterwarnings('ignore')

# Set plot style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# GPU CONFIGURATION
# ============================================================================

def configure_gpu():
    """Configure GPU settings for optimal performance"""
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
    else:
        print("⚠ No GPU detected, running on CPU")
        return False

# ============================================================================
# DATA LOADING
# ============================================================================

def load_images_masks(images_folder, masks_folder, img_size=(256,256)):
    """Load images and masks from folders"""
    images, masks = [], []
    image_files = sorted(os.listdir(images_folder))
    mask_files = sorted(os.listdir(masks_folder))
    mask_dict = {os.path.splitext(f)[0]: f for f in mask_files}
    
    for img_file in image_files:
        img_path = os.path.join(images_folder, img_file)
        img = np.array(Image.open(img_path).resize(img_size)) / 255.0
        images.append(img)
        
        img_name = os.path.splitext(img_file)[0]
        if img_name in mask_dict:
            mask_path = os.path.join(masks_folder, mask_dict[img_name])
            mask = np.array(Image.open(mask_path).resize(img_size, Image.NEAREST))
            masks.append(mask)
        else:
            print(f"Warning: No mask found for {img_file}")
    
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

# ============================================================================
# DATA AUGMENTATION
# ============================================================================

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
# MSCA MODULE (FROM ODFORMER)
# ============================================================================

def msca_block(x, channels, dilation_rates=[1, 2, 3, 5], block_name='msca'):
    """
    Multi-scale Context Aggregator (MSCA) from ODFormer
    
    Features:
    - Multi-scale feature extraction using dilated convolutions
    - Parallel branches with different dilation rates [1, 2, 3, 5]
    - Feature fusion through concatenation and 1×1 convolution
    - Residual connection for gradient flow
    
    Args:
        x: Input feature map
        channels: Number of output channels
        dilation_rates: List of dilation rates for multi-scale context
        block_name: Name prefix for layers
    """
    multi_scale_features = []
    filters_per_branch = channels // len(dilation_rates)
    
    # Multi-scale branches with different dilation rates
    for i, rate in enumerate(dilation_rates):
        branch = layers.Conv2D(
            filters_per_branch,
            (3, 3),
            padding='same',
            dilation_rate=rate,
            use_bias=False,
            name=f'{block_name}_dil{rate}_conv'
        )(x)
        branch = layers.BatchNormalization(name=f'{block_name}_dil{rate}_bn')(branch)
        branch = layers.Activation('relu', name=f'{block_name}_dil{rate}_relu')(branch)
        multi_scale_features.append(branch)
    
    # Concatenate all branches
    concat = layers.Concatenate(name=f'{block_name}_concat')(multi_scale_features)
    
    # Fusion layer to combine multi-scale features
    fused = layers.Conv2D(channels, (1, 1), use_bias=False, name=f'{block_name}_fusion')(concat)
    fused = layers.BatchNormalization(name=f'{block_name}_fusion_bn')(fused)
    fused = layers.Activation('relu', name=f'{block_name}_fusion_relu')(fused)
    
    # Residual connection
    output = layers.Add(name=f'{block_name}_residual')([x, fused])
    
    return output

# ============================================================================
# LBFR MODULE (FROM ODFORMER)
# ============================================================================

def lbfr_block(x, channels, reduction=16, block_name='lbfr'):
    """
    Lightweight Bidirectional Feature Recalibrator (LBFR) from ODFormer
    
    Features:
    - Channel attention using both avg and max pooling
    - Spatial attention using 2-channel feature concatenation
    - Recalibration layer with BatchNormalization
    - Residual connection for gradient flow
    
    Args:
        x: Input feature map
        channels: Number of channels
        reduction: Channel reduction ratio for attention bottleneck
        block_name: Name prefix for layers
    """
    # Channel Attention
    # Average pooling branch
    avg_pool = layers.GlobalAveragePooling2D(name=f'{block_name}_avg_pool')(x)
    avg_pool = layers.Reshape((1, 1, channels), name=f'{block_name}_avg_reshape')(avg_pool)
    
    # Max pooling branch
    max_pool = layers.GlobalMaxPooling2D(name=f'{block_name}_max_pool')(x)
    max_pool = layers.Reshape((1, 1, channels), name=f'{block_name}_max_reshape')(max_pool)
    
    # Shared MLP for channel attention
    fc1 = layers.Conv2D(channels // reduction, (1, 1), use_bias=False, name=f'{block_name}_fc1')
    fc2 = layers.Conv2D(channels, (1, 1), use_bias=False, name=f'{block_name}_fc2')
    
    avg_out = fc1(avg_pool)
    avg_out = layers.Activation('relu', name=f'{block_name}_fc1_relu_avg')(avg_out)
    avg_out = fc2(avg_out)
    
    max_out = fc1(max_pool)
    max_out = layers.Activation('relu', name=f'{block_name}_fc1_relu_max')(max_out)
    max_out = fc2(max_out)
    
    # Combine and apply channel attention
    channel_att = layers.Add(name=f'{block_name}_channel_add')([avg_out, max_out])
    channel_att = layers.Activation('sigmoid', name=f'{block_name}_channel_sigmoid')(channel_att)
    x_channel = layers.Multiply(name=f'{block_name}_channel_multiply')([x, channel_att])
    
    # Spatial Attention
    # Average and max across channels
    avg_spatial = layers.Lambda(lambda x: K.mean(x, axis=-1, keepdims=True), 
                                name=f'{block_name}_spatial_avg')(x_channel)
    max_spatial = layers.Lambda(lambda x: K.max(x, axis=-1, keepdims=True),
                                name=f'{block_name}_spatial_max')(x_channel)
    
    # Concatenate and apply spatial attention
    spatial_concat = layers.Concatenate(name=f'{block_name}_spatial_concat')([avg_spatial, max_spatial])
    spatial_att = layers.Conv2D(1, (7, 7), padding='same', use_bias=False,
                                name=f'{block_name}_spatial_conv')(spatial_concat)
    spatial_att = layers.Activation('sigmoid', name=f'{block_name}_spatial_sigmoid')(spatial_att)
    x_spatial = layers.Multiply(name=f'{block_name}_spatial_multiply')([x_channel, spatial_att])
    
    # Recalibration
    recalibrated = layers.Conv2D(channels, (1, 1), use_bias=False, 
                                 name=f'{block_name}_recalibrate')(x_spatial)
    recalibrated = layers.BatchNormalization(name=f'{block_name}_recalibrate_bn')(recalibrated)
    
    # Residual connection
    output = layers.Add(name=f'{block_name}_residual')([x, recalibrated])
    
    return output

# ============================================================================
# PPM MODULE (FROM PSPNET/ODFORMER)
# ============================================================================

def ppm_block(x, channels, pool_scales=[1, 2, 3, 6], block_name='ppm'):
    """
    Pyramid Pooling Module (PPM) from PSPNet
    
    Features:
    - Multi-scale adaptive pooling at different pyramid levels
    - Captures global context at scales: 1×1, 2×2, 3×3, 6×6
    - Lightweight 1×1 convolutions for each scale
    - Feature fusion through concatenation
    
    Args:
        x: Input feature map
        channels: Number of output channels
        pool_scales: List of pooling scales for pyramid levels
        block_name: Name prefix for layers
    """
    h, w = x.shape[1], x.shape[2]
    
    # Store original features
    ppm_features = [x]
    
    # Multi-scale pooling branches
    filters_per_scale = channels // len(pool_scales)
    
    for i, scale in enumerate(pool_scales):
        # Global average pooling to fixed scale size
        pool_h = max(h // scale, 1)
        pool_w = max(w // scale, 1)
        
        pooled = layers.AveragePooling2D(
            pool_size=(h // pool_h, w // pool_w),
            strides=(h // pool_h, w // pool_w),
            name=f'{block_name}_pool_{scale}x{scale}'
        )(x)
        
        # 1×1 convolution to reduce channels
        conv = layers.Conv2D(
            filters_per_scale,
            (1, 1),
            use_bias=False,
            name=f'{block_name}_conv_{scale}x{scale}'
        )(pooled)
        conv = layers.BatchNormalization(name=f'{block_name}_bn_{scale}x{scale}')(conv)
        conv = layers.Activation('relu', name=f'{block_name}_relu_{scale}x{scale}')(conv)
        
        # Upsample back to original size using exact target dimensions
        upsampled = layers.Resizing(
            h, w,
            interpolation='bilinear',
            name=f'{block_name}_upsample_{scale}x{scale}'
        )(conv)
        
        ppm_features.append(upsampled)
    
    # Concatenate all pyramid features
    concat = layers.Concatenate(name=f'{block_name}_concat')(ppm_features)
    
    # Fusion convolution
    fused = layers.Conv2D(
        channels,
        (3, 3),
        padding='same',
        use_bias=False,
        name=f'{block_name}_fusion'
    )(concat)
    fused = layers.BatchNormalization(name=f'{block_name}_fusion_bn')(fused)
    fused = layers.Activation('relu', name=f'{block_name}_fusion_relu')(fused)
    
    return fused

# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

def focal_loss(y_true, y_pred, gamma=2.0, alpha=None, epsilon=1e-7):
    if alpha is None:
        alpha = [0.25, 0.25, 0.5]
    
    y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
    focal_loss_value = 0.0
    
    for c in range(3):
        y_true_c = y_true[:, :, :, c]
        y_pred_c = y_pred[:, :, :, c]
        
        bce = -(y_true_c * K.log(y_pred_c) + (1 - y_true_c) * K.log(1 - y_pred_c))
        pt = y_true_c * y_pred_c + (1 - y_true_c) * (1 - y_pred_c)
        focal_weight = K.pow(1 - pt, gamma)
        
        focal_loss_value += alpha[c] * K.mean(focal_weight * bce)
    
    return focal_loss_value

def enhanced_iou_loss(y_true, y_pred, smooth=1e-7):
    eiou_loss_value = 0.0
    
    for c in range(3):
        y_true_c = y_true[:, :, :, c]
        y_pred_c = y_pred[:, :, :, c]
        
        intersection = K.sum(y_true_c * y_pred_c, axis=[1, 2])
        union = K.sum(y_true_c, axis=[1, 2]) + K.sum(y_pred_c, axis=[1, 2]) - intersection
        iou = (intersection + smooth) / (union + smooth)
        
        y_true_bbox = K.stack([
            K.min(K.cast(y_true_c > 0.5, 'float32'), axis=1),
            K.min(K.cast(y_true_c > 0.5, 'float32'), axis=2),
            K.max(K.cast(y_true_c > 0.5, 'float32'), axis=1),
            K.max(K.cast(y_true_c > 0.5, 'float32'), axis=2)
        ], axis=-1)
        
        y_pred_bbox = K.stack([
            K.min(K.cast(y_pred_c > 0.5, 'float32'), axis=1),
            K.min(K.cast(y_pred_c > 0.5, 'float32'), axis=2),
            K.max(K.cast(y_pred_c > 0.5, 'float32'), axis=1),
            K.max(K.cast(y_pred_c > 0.5, 'float32'), axis=2)
        ], axis=-1)
        
        center_true = K.stack([
            (y_true_bbox[:, 0] + y_true_bbox[:, 2]) / 2.0,
            (y_true_bbox[:, 1] + y_true_bbox[:, 3]) / 2.0
        ], axis=-1)
        
        center_pred = K.stack([
            (y_pred_bbox[:, 0] + y_pred_bbox[:, 2]) / 2.0,
            (y_pred_bbox[:, 1] + y_pred_bbox[:, 3]) / 2.0
        ], axis=-1)
        
        center_distance = K.sum(K.square(center_true - center_pred), axis=-1)
        
        diag_length = K.square(y_true_bbox[:, 2] - y_true_bbox[:, 0]) + \
                     K.square(y_true_bbox[:, 3] - y_true_bbox[:, 1])
        
        diag_length = K.maximum(diag_length, 1e-7)
        
        eiou = iou - (center_distance / diag_length)
        eiou_loss_value += K.mean(1.0 - eiou)
    
    return eiou_loss_value / 3.0

def focal_eiou_combined_loss(y_true, y_pred, focal_weight=1.0, eiou_weight=1.0, gamma=2.0, alpha=None):
    fl = focal_loss(y_true, y_pred, gamma=gamma, alpha=alpha)
    eiou = enhanced_iou_loss(y_true, y_pred)
    return focal_weight * fl + eiou_weight * eiou

# ============================================================================
# MODEL ARCHITECTURE WITH MSCA + LBFR + PPM
# ============================================================================

def enhanced_decoder_block_with_triple_attention(x, skip_connection, filters, block_name, dropout_rate=0.3):
    """
    Enhanced decoder block with MSCA + LBFR + PPM triple attention
    
    Sequential enhancement strategy:
    1. Standard decoder convolutions with residual connection
    2. MSCA for multi-scale local context aggregation (dilated convs)
    3. LBFR for bidirectional feature recalibration (channel + spatial attention)
    4. PPM for multi-scale global context aggregation (pyramid pooling)
    
    This combines three complementary attention mechanisms:
    - MSCA: Local multi-scale receptive fields
    - LBFR: Channel and spatial recalibration
    - PPM: Global multi-scale context
    """
    x = layers.UpSampling2D((2, 2), name=f'{block_name}_upsample')(x)
    
    if skip_connection.shape[-1] != x.shape[-1]:
        skip_connection = layers.Conv2D(x.shape[-1], (1, 1), padding='same', 
                                       name=f'{block_name}_skip_adjust')(skip_connection)
    
    x = layers.Concatenate(name=f'{block_name}_concat')([x, skip_connection])
    
    x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same', name=f'{block_name}_conv1')(x)
    x = layers.BatchNormalization(name=f'{block_name}_bn1')(x)
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
    
    # Apply MSCA for local multi-scale context aggregation
    x = msca_block(x, filters, dilation_rates=[1, 2, 3, 5], block_name=f'{block_name}_msca')
    
    # Apply LBFR for bidirectional feature recalibration
    x = lbfr_block(x, filters, reduction=16, block_name=f'{block_name}_lbfr')
    
    # Apply PPM for global multi-scale context aggregation
    x = ppm_block(x, filters, pool_scales=[1, 2, 3, 6], block_name=f'{block_name}_ppm')
    
    return x

def build_unet_with_triple_attention(input_shape=(256, 256, 3), num_classes=3, dropout_rate=0.3):
    """Build MobileNetV2-UNet with MSCA + LBFR + PPM triple attention"""
    inputs = layers.Input(input_shape, dtype='float32')
    backbone = MobileNetV2(input_tensor=inputs, weights='imagenet', include_top=False)
    
    skip_1 = backbone.get_layer('block_1_expand_relu').output  # 128×128
    skip_2 = backbone.get_layer('block_3_expand_relu').output  # 64×64
    skip_3 = backbone.get_layer('block_6_expand_relu').output  # 32×32
    skip_4 = backbone.get_layer('block_13_expand_relu').output # 16×16
    bridge = backbone.output  # 8×8
    
    # Deeper bridge
    bridge = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(bridge)
    bridge = layers.BatchNormalization()(bridge)
    bridge = layers.Dropout(dropout_rate, name='bridge_dropout')(bridge)
    
    # Decoder with triple attention (MSCA + LBFR + PPM)
    x = enhanced_decoder_block_with_triple_attention(bridge, skip_4, 512, 'decoder_1', dropout_rate)
    x = enhanced_decoder_block_with_triple_attention(x, skip_3, 256, 'decoder_2', dropout_rate)
    x = enhanced_decoder_block_with_triple_attention(x, skip_2, 128, 'decoder_3', dropout_rate * 0.7)
    x = enhanced_decoder_block_with_triple_attention(x, skip_1, 64, 'decoder_4', dropout_rate * 0.5)
    
    # Final upsampling
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    outputs = layers.Conv2D(num_classes, (1, 1), activation='softmax', dtype='float32')(x)
    
    model = models.Model(inputs, outputs)
    
    # Print configuration
    dropout_layers = [layer for layer in model.layers if isinstance(layer, layers.Dropout)]
    print(f"\nModel Configuration:")
    print(f"  Total Parameters: {model.count_params():,}")
    print(f"  Dropout Layers: {len(dropout_layers)}")
    print(f"  Triple Attention Modules: MSCA + LBFR + PPM per decoder block")
    print(f"  MSCA Modules: 4 (local multi-scale with dilation rates [1,2,3,5])")
    print(f"  LBFR Modules: 4 (channel + spatial attention + recalibration)")
    print(f"  PPM Modules: 4 (global pyramid pooling [1×1, 2×2, 3×3, 6×6])")
    print(f"  Enhancement Strategy: Sequential MSCA → LBFR → PPM per block")
    print(f"  Total Attention Blocks: 12 (4 MSCA + 4 LBFR + 4 PPM)")
    print(f"  Context Coverage: Local Multi-scale + Attention + Global Multi-scale")
    
    return model

# ============================================================================
# METRICS
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
    """MC Dropout prediction for uncertainty estimation"""
    if verbose:
        print(f"\nRunning MC Dropout with {num_samples} stochastic forward passes...")
    
    predictions = []
    num_batches = int(np.ceil(len(X) / batch_size))
    
    for i in range(num_samples):
        if verbose and (i + 1) % 5 == 0:
            print(f"  Completed: {i+1}/{num_samples} forward passes")
        
        batch_preds = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(X))
            X_batch = X[start_idx:end_idx]
            
            pred_batch = model(X_batch, training=True)
            batch_preds.append(pred_batch.numpy())
        
        pred = np.concatenate(batch_preds, axis=0)
        predictions.append(pred)
        
        if (i + 1) % 5 == 0:
            gc.collect()
    
    all_preds = np.array(predictions)
    mean_pred = np.mean(all_preds, axis=0)
    std_pred = np.std(all_preds, axis=0)
    
    del all_preds
    gc.collect()
    
    if verbose:
        print(f"✓ MC Dropout completed")
    
    return mean_pred, std_pred

# ============================================================================
# GLAUCOMA PROGRESSION SCORING
# ============================================================================

def calculate_cdr(mask, disc_class=1, cup_class=2, method='vertical'):
    """
    Calculate Cup-to-Disc Ratio (CDR) from segmentation mask
    
    Methods:
    - 'vertical': Vertical diameter ratio (clinical standard, most accurate)
    - 'area_sqrt': Square root of area ratio (robust alternative)
    
    Args:
        mask: Segmentation mask (H, W) with class labels
        disc_class: Class label for optic disc (default=1)
        cup_class: Class label for optic cup (default=2)
        method: Calculation method (default='vertical')
    
    Returns:
        dict with vCDR, hCDR, avgCDR, area_CDR
    """
    disc_mask = (mask == disc_class).astype(np.uint8)
    cup_mask = (mask == cup_class).astype(np.uint8)
    
    # Check if disc and cup are present
    disc_area = disc_mask.sum()
    cup_area = cup_mask.sum()
    
    if disc_area == 0 or cup_area == 0:
        return {'vCDR': 0.0, 'hCDR': 0.0, 'avgCDR': 0.0, 'area_CDR': 0.0, 'valid': False, 
                'method': method, 'disc_area': 0, 'cup_area': 0}
    
    # Find coordinates
    disc_coords = np.where(disc_mask > 0)
    cup_coords = np.where(cup_mask > 0)
    
    # Calculate diameters using bounding box
    disc_v_diameter = disc_coords[0].max() - disc_coords[0].min() + 1
    cup_v_diameter = cup_coords[0].max() - cup_coords[0].min() + 1
    disc_h_diameter = disc_coords[1].max() - disc_coords[1].min() + 1
    cup_h_diameter = cup_coords[1].max() - cup_coords[1].min() + 1
    
    # Calculate CDRs
    vCDR = cup_v_diameter / disc_v_diameter if disc_v_diameter > 0 else 0.0
    hCDR = cup_h_diameter / disc_h_diameter if disc_h_diameter > 0 else 0.0
    
    # Calculate area-based CDR
    area_CDR = np.sqrt(cup_area / disc_area)
    
    # Choose primary CDR based on method
    if method == 'vertical':
        avgCDR = vCDR  # Use vertical CDR (clinical standard)
    elif method == 'area_sqrt':
        avgCDR = area_CDR  # Use area-based
    else:
        avgCDR = (vCDR + hCDR) / 2.0  # Average of both
    
    return {
        'vCDR': float(vCDR),
        'hCDR': float(hCDR),
        'avgCDR': float(avgCDR),
        'area_CDR': float(area_CDR),
        'valid': True,
        'method': method,
        'disc_area': int(disc_area),
        'cup_area': int(cup_area),
        'disc_v_diameter': int(disc_v_diameter),
        'cup_v_diameter': int(cup_v_diameter)
    }

def classify_glaucoma_severity(cdr):
    """
    Classify glaucoma severity based on CDR
    
    Classification:
    - Normal: CDR < 0.3
    - Suspect: 0.3 ≤ CDR < 0.6
    - Moderate: 0.6 ≤ CDR < 0.8
    - Critical: CDR ≥ 0.8
    
    Args:
        cdr: Cup-to-Disc Ratio (float)
    
    Returns:
        tuple: (severity_class, severity_label, color)
    """
    if cdr < 0.3:
        return 0, 'Normal', '#4CAF50'  # Green
    elif cdr < 0.6:
        return 1, 'Suspect', '#FFC107'  # Amber
    elif cdr < 0.8:
        return 2, 'Moderate', '#FF9800'  # Orange
    else:
        return 3, 'Critical', '#F44336'  # Red

def analyze_glaucoma_progression(masks, predictions, cdr_method='area_sqrt'):
    """
    Analyze glaucoma progression for a batch of samples
    
    Args:
        masks: Ground truth masks (N, H, W)
        predictions: Predicted masks (N, H, W)
        cdr_method: CDR calculation method (default='area_sqrt' for robustness)
    
    Returns:
        DataFrame with CDR metrics and classifications
    """
    results = []
    
    for i, (gt_mask, pred_mask) in enumerate(zip(masks, predictions)):
        # Calculate CDR for ground truth
        gt_cdr = calculate_cdr(gt_mask, method=cdr_method)
        
        # Calculate CDR for prediction
        pred_cdr = calculate_cdr(pred_mask, method=cdr_method)
        
        # Classify severity
        if gt_cdr['valid']:
            gt_class, gt_label, gt_color = classify_glaucoma_severity(gt_cdr['avgCDR'])
        else:
            gt_class, gt_label, gt_color = -1, 'Invalid', '#9E9E9E'
        
        if pred_cdr['valid']:
            pred_class, pred_label, pred_color = classify_glaucoma_severity(pred_cdr['avgCDR'])
        else:
            pred_class, pred_label, pred_color = -1, 'Invalid', '#9E9E9E'
        
        # Calculate CDR error
        cdr_error = abs(gt_cdr['avgCDR'] - pred_cdr['avgCDR']) if gt_cdr['valid'] and pred_cdr['valid'] else np.nan
        
        # Check if classification matches
        severity_match = (gt_label == pred_label)
        
        # Manual review needed: Critical cases or mismatches
        manual_review = (gt_label == 'Critical') or (not severity_match)
        
        results.append({
            'Sample': i + 1,
            'GT_vCDR': gt_cdr['vCDR'],
            'GT_hCDR': gt_cdr['hCDR'],
            'GT_avgCDR': gt_cdr['avgCDR'],
            'GT_area_CDR': gt_cdr['area_CDR'],
            'GT_Severity': gt_label,
            'Pred_vCDR': pred_cdr['vCDR'],
            'Pred_hCDR': pred_cdr['hCDR'],
            'Pred_avgCDR': pred_cdr['avgCDR'],
            'Pred_area_CDR': pred_cdr['area_CDR'],
            'Pred_Severity': pred_label,
            'CDR_Error': cdr_error,
            'Severity_Match': severity_match,
            'Manual_Review_Needed': manual_review
        })
    
    return pd.DataFrame(results)

def calculate_progression_accuracy(df_progression):
    """
    Calculate simple accuracy metrics: GT progression vs Predicted progression
    
    Returns accuracy % like: "GT says Normal for 250 images, model says 230 are Normal"
    """
    results = {}
    
    # Overall accuracy
    total = len(df_progression)
    correct = df_progression['Severity_Match'].sum()
    overall_accuracy = (correct / total) * 100
    
    results['overall'] = {
        'total': total,
        'correct': correct,
        'accuracy': overall_accuracy
    }
    
    # Per-class accuracy
    severity_classes = ['Normal', 'Suspect', 'Moderate', 'Critical']
    results['per_class'] = {}
    
    for severity in severity_classes:
        gt_count = len(df_progression[df_progression['GT_Severity'] == severity])
        pred_count = len(df_progression[df_progression['Pred_Severity'] == severity])
        
        # Correctly predicted: GT is this class AND model predicted it as this class
        correct_count = len(df_progression[
            (df_progression['GT_Severity'] == severity) & 
            (df_progression['Pred_Severity'] == severity)
        ])
        
        class_accuracy = (correct_count / gt_count * 100) if gt_count > 0 else 0.0
        
        results['per_class'][severity] = {
            'gt_count': gt_count,
            'pred_count': pred_count,
            'correct': correct_count,
            'accuracy': class_accuracy
        }
    
    # Manual review statistics
    manual_review_needed = df_progression['Manual_Review_Needed'].sum()
    manual_review_pct = (manual_review_needed / total) * 100
    
    results['manual_review'] = {
        'count': manual_review_needed,
        'percentage': manual_review_pct
    }
    
    return results

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_sample_predictions(model, X_test, y_test, model_name, num_samples=6, save_dir='results'):
    """Visualize sample predictions with original image, GT, segmented mask, and overlay"""
    os.makedirs(save_dir, exist_ok=True)
    
    y_pred = model.predict(X_test[:num_samples], verbose=0, batch_size=4)
    y_pred_labels = np.argmax(y_pred, axis=-1)
    
    colors = {0: [0, 0, 0], 1: [0, 255, 0], 2: [255, 0, 0]}
    
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
    
    for i in range(num_samples):
        # Original image
        axes[i, 0].imshow(X_test[i])
        axes[i, 0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[i, 0].axis('off')
        
        # Ground truth
        gt_colored = np.zeros((256, 256, 3), dtype=np.uint8)
        for class_id, color in colors.items():
            gt_colored[y_test[i] == class_id] = color
        axes[i, 1].imshow(gt_colored)
        axes[i, 1].set_title('Ground Truth (GT)', fontsize=12, fontweight='bold')
        axes[i, 1].axis('off')
        
        # Segmented mask
        pred_colored = np.zeros((256, 256, 3), dtype=np.uint8)
        for class_id, color in colors.items():
            pred_colored[y_pred_labels[i] == class_id] = color
        axes[i, 2].imshow(pred_colored)
        axes[i, 2].set_title('Segmented Mask', fontsize=12, fontweight='bold')
        axes[i, 2].axis('off')
        
        # Overlay
        overlay = X_test[i].copy()
        for class_id, color in colors.items():
            if class_id > 0:
                mask = (y_pred_labels[i] == class_id).astype(np.float32)
                mask_rgb = np.stack([mask * color[0]/255, mask * color[1]/255, mask * color[2]/255], axis=-1)
                overlay = overlay * 0.6 + mask_rgb * 0.4
        axes[i, 3].imshow(overlay)
        axes[i, 3].set_title('Overlay', fontsize=12, fontweight='bold')
        axes[i, 3].axis('off')
        
        # Calculate dice
        y_test_cat = to_categorical(y_test[i], num_classes=3)
        y_pred_cat = to_categorical(y_pred_labels[i], num_classes=3)
        
        disc_dice = 2 * np.sum(y_test_cat[:,:,1] * y_pred_cat[:,:,1]) / (np.sum(y_test_cat[:,:,1]) + np.sum(y_pred_cat[:,:,1]) + 1e-7)
        cup_dice = 2 * np.sum(y_test_cat[:,:,2] * y_pred_cat[:,:,2]) / (np.sum(y_test_cat[:,:,2]) + np.sum(y_pred_cat[:,:,2]) + 1e-7)
        
        axes[i, 0].text(0.5, -0.1, f'Sample {i+1}\nDisc: {disc_dice:.3f} | Cup: {cup_dice:.3f}', 
                       transform=axes[i, 0].transAxes, ha='center', fontsize=10, 
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(f'{model_name} - Predictions: Original | GT | Segmented | Overlay', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/{model_name}_predictions.png', dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {save_dir}/{model_name}_predictions.png")
    plt.show()
    plt.close()

def plot_sample_predictions_with_cdr(model, X_test, y_test, model_name, num_samples=6, save_dir='results'):
    """Visualize predictions with CDR and glaucoma severity"""
    os.makedirs(save_dir, exist_ok=True)
    
    y_pred = model.predict(X_test[:num_samples], verbose=0, batch_size=4)
    y_pred_labels = np.argmax(y_pred, axis=-1)
    
    colors = {0: [0, 0, 0], 1: [0, 255, 0], 2: [255, 0, 0]}
    
    fig, axes = plt.subplots(num_samples, 4, figsize=(18, 4*num_samples))
    
    for i in range(num_samples):
        # Original image
        axes[i, 0].imshow(X_test[i])
        axes[i, 0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[i, 0].axis('off')
        
        # Ground truth
        gt_colored = np.zeros((256, 256, 3), dtype=np.uint8)
        for class_id, color in colors.items():
            gt_colored[y_test[i] == class_id] = color
        axes[i, 1].imshow(gt_colored)
        
        # Calculate GT CDR
        gt_cdr = calculate_cdr(y_test[i])
        gt_class, gt_label, gt_color = classify_glaucoma_severity(gt_cdr['avgCDR']) if gt_cdr['valid'] else (-1, 'Invalid', '#9E9E9E')
        
        axes[i, 1].set_title(f'Ground Truth\nCDR: {gt_cdr["avgCDR"]:.3f} | {gt_label}', 
                            fontsize=12, fontweight='bold', color=gt_color)
        axes[i, 1].axis('off')
        
        # Prediction
        pred_colored = np.zeros((256, 256, 3), dtype=np.uint8)
        for class_id, color in colors.items():
            pred_colored[y_pred_labels[i] == class_id] = color
        axes[i, 2].imshow(pred_colored)
        
        # Calculate Pred CDR
        pred_cdr = calculate_cdr(y_pred_labels[i])
        pred_class, pred_label, pred_color = classify_glaucoma_severity(pred_cdr['avgCDR']) if pred_cdr['valid'] else (-1, 'Invalid', '#9E9E9E')
        
        axes[i, 2].set_title(f'Prediction\nCDR: {pred_cdr["avgCDR"]:.3f} | {pred_label}', 
                            fontsize=12, fontweight='bold', color=pred_color)
        axes[i, 2].axis('off')
        
        # Overlay
        overlay = X_test[i].copy()
        for class_id, color in colors.items():
            if class_id > 0:
                mask = (y_pred_labels[i] == class_id).astype(np.float32)
                mask_rgb = np.stack([mask * color[0]/255, mask * color[1]/255, mask * color[2]/255], axis=-1)
                overlay = overlay * 0.6 + mask_rgb * 0.4
        axes[i, 3].imshow(overlay)
        axes[i, 3].set_title('Overlay', fontsize=12, fontweight='bold')
        axes[i, 3].axis('off')
        
        # Calculate dice
        y_test_cat = to_categorical(y_test[i], num_classes=3)
        y_pred_cat = to_categorical(y_pred_labels[i], num_classes=3)
        
        disc_dice = 2 * np.sum(y_test_cat[:,:,1] * y_pred_cat[:,:,1]) / (np.sum(y_test_cat[:,:,1]) + np.sum(y_pred_cat[:,:,1]) + 1e-7)
        cup_dice = 2 * np.sum(y_test_cat[:,:,2] * y_pred_cat[:,:,2]) / (np.sum(y_test_cat[:,:,2]) + np.sum(y_pred_cat[:,:,2]) + 1e-7)
        
        cdr_error = abs(gt_cdr['avgCDR'] - pred_cdr['avgCDR']) if gt_cdr['valid'] and pred_cdr['valid'] else np.nan
        
        info_text = f'Sample {i+1}\n'
        info_text += f'Disc Dice: {disc_dice:.3f} | Cup Dice: {cup_dice:.3f}\n'
        info_text += f'CDR Error: {cdr_error:.3f}' if not np.isnan(cdr_error) else 'CDR Error: N/A'
        
        axes[i, 0].text(0.5, -0.15, info_text, transform=axes[i, 0].transAxes, 
                       ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(f'{model_name} - Predictions with Glaucoma Severity Assessment', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/{model_name}_predictions_cdr.png', dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {save_dir}/{model_name}_predictions_cdr.png")
    plt.show()
    plt.close()

def plot_cdr_distribution(df_progression, save_dir='results'):
    """Plot CDR distribution and severity classification"""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # CDR scatter plot (GT vs Pred)
    axes[0, 0].scatter(df_progression['GT_avgCDR'], df_progression['Pred_avgCDR'], 
                      alpha=0.6, s=100, c='blue', edgecolors='black')
    axes[0, 0].plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Prediction')
    axes[0, 0].set_xlabel('Ground Truth CDR', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Predicted CDR', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('CDR: Ground Truth vs Prediction', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add severity zones
    axes[0, 0].axhspan(0, 0.3, alpha=0.1, color='green', label='Normal')
    axes[0, 0].axhspan(0.3, 0.6, alpha=0.1, color='yellow', label='Suspect')
    axes[0, 0].axhspan(0.6, 0.8, alpha=0.1, color='orange', label='Moderate')
    axes[0, 0].axhspan(0.8, 1.0, alpha=0.1, color='red', label='Critical')
    
    # CDR error histogram
    valid_errors = df_progression['CDR_Error'].dropna()
    axes[0, 1].hist(valid_errors, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(valid_errors.mean(), color='red', linestyle='--', linewidth=2, 
                      label=f'Mean Error: {valid_errors.mean():.4f}')
    axes[0, 1].set_xlabel('CDR Error (|GT - Pred|)', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('CDR Error Distribution', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Severity classification confusion matrix
    severity_order = ['Normal', 'Suspect', 'Moderate', 'Critical']
    gt_severity = pd.Categorical(df_progression['GT_Severity'], categories=severity_order, ordered=True)
    pred_severity = pd.Categorical(df_progression['Pred_Severity'], categories=severity_order, ordered=True)
    
    cm = confusion_matrix(gt_severity, pred_severity, labels=severity_order)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd', 
                xticklabels=severity_order, yticklabels=severity_order,
                ax=axes[1, 0], cbar_kws={'label': 'Count'})
    axes[1, 0].set_title('Glaucoma Severity Classification', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylabel('Ground Truth', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Predicted', fontsize=12, fontweight='bold')
    
    # Severity distribution
    severity_counts_gt = df_progression['GT_Severity'].value_counts()
    severity_counts_pred = df_progression['Pred_Severity'].value_counts()
    
    x = np.arange(len(severity_order))
    width = 0.35
    
    counts_gt = [severity_counts_gt.get(s, 0) for s in severity_order]
    counts_pred = [severity_counts_pred.get(s, 0) for s in severity_order]
    
    bars1 = axes[1, 1].bar(x - width/2, counts_gt, width, label='Ground Truth', 
                          color=['#4CAF50', '#FFC107', '#FF9800', '#F44336'], alpha=0.7)
    bars2 = axes[1, 1].bar(x + width/2, counts_pred, width, label='Predicted', 
                          color=['#4CAF50', '#FFC107', '#FF9800', '#F44336'], alpha=0.4)
    
    axes[1, 1].set_xlabel('Severity Level', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Count', fontsize=12, fontweight='bold')
    axes[1, 1].set_title('Severity Distribution', fontsize=14, fontweight='bold')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(severity_order, rotation=15)
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/cdr_analysis.png', dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {save_dir}/cdr_analysis.png")
    plt.show()
    plt.close()

def plot_training_history(history, model_name, save_dir='results'):
    """Plot training history - Loss and Accuracy only"""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curves
    axes[0].plot(history.history['loss'], label='Training Loss', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0].set_title(f'{model_name} - Loss Over Epochs', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy curves
    axes[1].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[1].set_title(f'{model_name} - Accuracy Over Epochs', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/{model_name}_history.png', dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {save_dir}/{model_name}_history.png")
    plt.show()
    plt.close()

def plot_training_curves(history, model_name, save_dir='results'):
    """Plot training loss and accuracy curves"""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss
    axes[0].plot(history.history['loss'], label='Training Loss', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0].set_title(f'{model_name} - Loss Over Epochs', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
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
    """Plot confusion matrices"""
    os.makedirs(save_dir, exist_ok=True)
    
    y_true_labels = np.argmax(y_true, axis=-1).flatten()
    y_pred_labels = np.argmax(y_pred, axis=-1).flatten()
    
    cm = confusion_matrix(y_true_labels, y_pred_labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    class_names = ['Background', 'Disc', 'Cup']
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[0], cbar_kws={'label': 'Count'})
    axes[0].set_title(f'{model_name}\nConfusion Matrix', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('True Label', fontsize=12)
    axes[0].set_xlabel('Predicted Label', fontsize=12)
    
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
    
    print(f"\nConfusion Matrix:\n{cm}")
    print(f"\nNormalized Confusion Matrix:\n{np.round(cm_normalized, 3)}")
    
    return cm, cm_normalized

# ============================================================================
# TRAINING AND EVALUATION
# ============================================================================

def train_model(X_train, y_train, X_val, y_val, X_test, y_test, model_name, 
               augment=None, save_dir='results'):
    """Train model with MSCA + LBFR + PPM triple enhancement"""
    print(f"\n{'='*80}")
    print(f"TRAINING: {model_name}")
    print(f"{'='*80}")
    print(f"  Architecture: Deeper Decoder + MSCA + LBFR + PPM + Focal-EIoU + MC Dropout")
    print(f"  Loss: Focal + Enhanced IoU (Combined)")
    
    K.clear_session()
    
    # Build model
    model = build_unet_with_triple_attention(dropout_rate=0.3)
    param_count = model.count_params()
    
    # Compile
    model.compile(
        optimizer=optimizers.Adam(learning_rate=5e-4),
        loss=focal_eiou_combined_loss,
        metrics=[dice_coef_multiclass, dice_class_0, dice_class_1, dice_class_2,
                "accuracy", iou_coef_multiclass, iou_class_0, iou_class_1, iou_class_2]
    )
    
    # Callbacks
    checkpoint = ModelCheckpoint(f"{save_dir}/{model_name}.keras", save_best_only=True, 
                                verbose=1, monitor='val_loss', mode='min')
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3, 
                                  verbose=1, mode='min', min_lr=1e-7)
    early_stop = EarlyStopping(monitor="val_loss", patience=10, verbose=1, 
                              restore_best_weights=True, mode='min')
    
    # Training
    batch_size = 4
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
    plot_training_history(history, model_name, save_dir)
    plot_sample_predictions_with_cdr(model, X_test, y_test, model_name, num_samples=8, save_dir=save_dir)
    
    return model, history, param_count

def evaluate_model(model, X_test, y_test, model_name, run_mc_dropout=True):
    """Evaluate model with glaucoma progression analysis"""
    print(f"\n{'='*80}")
    print(f"EVALUATION: {model_name}")
    print(f"{'='*80}")
    
    y_test_cat = np.array([to_categorical(mask, num_classes=3) for mask in y_test])
    
    # Predict
    y_pred = model.predict(X_test, verbose=0, batch_size=8)
    y_pred_labels = np.argmax(y_pred, axis=-1)
    
    # Confusion matrices
    plot_confusion_matrices(y_test_cat, y_pred, model_name, save_dir='results')
    
    # Classification metrics
    y_true_labels = np.argmax(y_test_cat, axis=-1).flatten()
    y_pred_labels_flat = y_pred_labels.flatten()
    
    precision = precision_score(y_true_labels, y_pred_labels_flat, average=None, zero_division=0)
    recall = recall_score(y_true_labels, y_pred_labels_flat, average=None, zero_division=0)
    f1 = f1_score(y_true_labels, y_pred_labels_flat, average=None, zero_division=0)
    
    print(f"\nPrecision, Recall, F1 Score per class:")
    class_names = ["Background", "Disc", "Cup"]
    for idx, cname in enumerate(class_names):
        print(f"  {cname}: Precision={precision[idx]:.4f}, Recall={recall[idx]:.4f}, F1={f1[idx]:.4f}")
    
    print(f"\nMacro-averaged metrics:")
    print(f"  Precision: {np.mean(precision):.4f}")
    print(f"  Recall: {np.mean(recall):.4f}")
    print(f"  F1 Score: {np.mean(f1):.4f}")
    
    # Per-class Dice and IoU
    results = {'standard': {}}
    for cidx, cname in enumerate(class_names):
        y_true_c = (np.argmax(y_test_cat, axis=-1) == cidx).astype(np.float32)
        y_pred_c = (y_pred_labels == cidx).astype(np.float32)
        
        intersection = np.sum(y_true_c * y_pred_c)
        dice = (2. * intersection + 1e-7) / (np.sum(y_true_c) + np.sum(y_pred_c) + 1e-7)
        union = np.sum(y_true_c) + np.sum(y_pred_c) - intersection
        iou = (intersection + 1e-7) / (union + 1e-7)
        
        results['standard'][cname] = {'dice': dice, 'iou': iou}
        print(f"  {cname}: Dice={dice:.4f}, IoU={iou:.4f}")
    
    # Glaucoma Progression Analysis
    print(f"\n{'='*80}")
    print(f"GLAUCOMA PROGRESSION ANALYSIS (Using Area-based CDR)")
    print(f"{'='*80}")
    
    df_progression = analyze_glaucoma_progression(y_test, y_pred_labels, cdr_method='area_sqrt')
    
    # CDR Statistics
    print(f"\n{'-'*80}")
    print(f"CDR Statistics:")
    print(f"{'-'*80}")
    print(f"  Mean GT CDR:   {df_progression['GT_avgCDR'].mean():.4f} ± {df_progression['GT_avgCDR'].std():.4f}")
    print(f"  Mean Pred CDR: {df_progression['Pred_avgCDR'].mean():.4f} ± {df_progression['Pred_avgCDR'].std():.4f}")
    print(f"  Mean CDR Error: {df_progression['CDR_Error'].mean():.4f} ± {df_progression['CDR_Error'].std():.4f}")
    
    # Calculate progression accuracy
    accuracy_results = calculate_progression_accuracy(df_progression)
    
    # Overall Accuracy
    print(f"\n{'='*80}")
    print(f"PROGRESSION CLASSIFICATION ACCURACY")
    print(f"{'='*80}")
    print(f"\nOverall Accuracy: {accuracy_results['overall']['accuracy']:.2f}%")
    print(f"  - Correctly Classified: {accuracy_results['overall']['correct']}/{accuracy_results['overall']['total']} samples")
    print(f"  - Incorrectly Classified: {accuracy_results['overall']['total'] - accuracy_results['overall']['correct']}/{accuracy_results['overall']['total']} samples")
    
    # Per-Class Results
    print(f"\n{'-'*80}")
    print(f"Per-Class Results:")
    print(f"{'-'*80}")
    print(f"{'Severity':<12} {'GT Count':<12} {'Pred Count':<12} {'Correct':<12} {'Accuracy':<12}")
    print(f"{'-'*80}")
    
    for severity in ['Normal', 'Suspect', 'Moderate', 'Critical']:
        stats = accuracy_results['per_class'][severity]
        print(f"{severity:<12} {stats['gt_count']:<12} {stats['pred_count']:<12} "
              f"{stats['correct']:<12} {stats['accuracy']:<12.2f}%")
    
    print(f"\n{'='*80}")
    print(f"INTERPRETATION EXAMPLE:")
    print(f"{'='*80}")
    normal_stats = accuracy_results['per_class']['Normal']
    print(f"  - GT says 'Normal' for {normal_stats['gt_count']} images")
    print(f"  - Model says 'Normal' for {normal_stats['pred_count']} images")
    print(f"  - Model correctly predicts {normal_stats['correct']} as 'Normal'")
    print(f"  - Accuracy for Normal class: {normal_stats['correct']}/{normal_stats['gt_count']} = {normal_stats['accuracy']:.1f}%")
    
    # Manual Review Recommendations
    print(f"\n{'='*80}")
    print(f"MANUAL REVIEW RECOMMENDATIONS")
    print(f"{'='*80}")
    print(f"\nSamples Requiring Manual Review: {accuracy_results['manual_review']['count']}/{accuracy_results['overall']['total']} "
          f"({accuracy_results['manual_review']['percentage']:.1f}%)")
    print(f"\nCriteria for Manual Review:")
    print(f"  1. All Critical cases (high-risk, requires expert verification)")
    print(f"  2. Classification mismatches (GT ≠ Predicted severity)")
    print(f"\nReason: Critical cases need expert confirmation, and mismatches may")
    print(f"indicate borderline cases or segmentation issues requiring review.")
    
    # Breakdown of manual review cases
    review_df = df_progression[df_progression['Manual_Review_Needed'] == True]
    critical_cases = len(review_df[review_df['GT_Severity'] == 'Critical'])
    mismatch_cases = len(review_df[review_df['Severity_Match'] == False])
    overlap = len(review_df[(review_df['GT_Severity'] == 'Critical') & (review_df['Severity_Match'] == False)])
    
    print(f"\nBreakdown:")
    print(f"  - Critical cases: {critical_cases} samples")
    print(f"  - Classification mismatches: {mismatch_cases} samples")
    print(f"  - Overlap (Critical + Mismatch): {overlap} samples")
    
    # Save progression data
    df_progression.to_csv('results/glaucoma_progression_analysis.csv', index=False)
    print(f"\n✓ Saved: results/glaucoma_progression_analysis.csv")
    
    # Plot CDR analysis
    plot_cdr_distribution(df_progression, save_dir='results')
    
    results['glaucoma'] = {
        'mean_cdr_error': df_progression['CDR_Error'].mean(),
        'median_cdr_error': df_progression['CDR_Error'].median(),
        'classification_accuracy': accuracy_results['overall']['accuracy'] / 100,
        'df_progression': df_progression,
        'accuracy_breakdown': accuracy_results
    }
    
    # MC Dropout
    if run_mc_dropout:
        print(f"\n{'='*80}")
        print(f"MC DROPOUT UNCERTAINTY ESTIMATION")
        print(f"{'='*80}")
        mean_pred, std_pred = mc_dropout_predict(model, X_test[:100], num_samples=15, verbose=True)
        results['mc_dropout'] = {}
        for cidx, cname in enumerate(class_names):
            mean_std = np.mean(std_pred[:, :, :, cidx])
            results['mc_dropout'][cname] = {'mean_uncertainty': mean_std}
            print(f"  {cname} uncertainty: {mean_std:.4f}")
    
    return results

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Setup
    gpu_available = configure_gpu()
    root_dir = "/kaggle/input/refuge/REFUGE/"
    train_aug = setup_augmentations()
    save_dir = "results"
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n{'='*100}")
    print(f"{'MOBILENET-UNET WITH MSCA+LBFR+PPM + GLAUCOMA PROGRESSION SCORING':^100s}")
    print(f"{'='*100}")
    print(f"\nModel Architecture:")
    print(f"  - Base: MobileNetV2 Encoder + Deeper Decoder")
    print(f"  - Triple Attention: MSCA + LBFR + PPM (12 modules total)")
    print(f"  - Loss: Focal + Enhanced IoU (Combined)")
    print(f"  - Uncertainty: MC Dropout (15 forward passes)")
    print(f"\nGlaucoma Severity Classification:")
    print(f"  - Normal:   CDR < 0.3  (Green)")
    print(f"  - Suspect:  0.3 ≤ CDR < 0.6  (Amber)")
    print(f"  - Moderate: 0.6 ≤ CDR < 0.8  (Orange)")
    print(f"  - Critical: CDR ≥ 0.8  (Red)")
    print(f"  CDR Method: Area-based (sqrt of area ratio) - Robust & Clinically Relevant")
    print(f"\nManual Review: Recommended for Critical cases and classification mismatches")
    
    # Load data
    all_images, all_masks = load_all_data(root_dir)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = create_new_split(all_images, all_masks)
    
    # Train model
    model, history, params = train_model(
        X_train, y_train, X_val, y_val, X_test, y_test,
        "MSCA_LBFR_PPM_GlaucomaScoring",
        augment=train_aug, save_dir=save_dir
    )
    
    # Evaluate with glaucoma analysis
    results = evaluate_model(model, X_test, y_test, "MSCA+LBFR+PPM+GlaucomaScoring", run_mc_dropout=True)
    
    print(f"\n{'='*100}")
    print(f"{'✓ TRAINING AND EVALUATION COMPLETED!':^100s}")
    print(f"{'='*100}")
    print(f"\nModel: MSCA + LBFR + PPM + Focal + EIoU + MC Dropout + Glaucoma Scoring")
    print(f"Parameters: {params:,}")
    print(f"Results saved in: {save_dir}/")
    print(f"\nGenerated Files:")
    print(f"  - Training history: {save_dir}/MSCA_LBFR_PPM_GlaucomaScoring_history.png")
    print(f"  - Predictions with CDR: {save_dir}/MSCA_LBFR_PPM_GlaucomaScoring_predictions_cdr.png")
    print(f"  - CDR Analysis: {save_dir}/cdr_analysis.png")
    print(f"  - Progression CSV: {save_dir}/glaucoma_progression_analysis.csv")
    print(f"  - Confusion Matrices: {save_dir}/MSCA+LBFR+PPM+GlaucomaScoring_confusion_matrices.png")
    print(f"\n{'='*100}\n")