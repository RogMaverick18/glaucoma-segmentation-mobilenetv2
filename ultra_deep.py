"""
ENHANCED COMBINED MOBILENET-UNET EXPERIMENT V2
================================================
Aggressive improvements:
1. MUCH Deeper Decoder (5-6 conv layers per block)
2. Strategic Dropout Placement (varying rates)
3. Optimized hyperparameters for combined approach
4. Longer training with proper patience
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, backend as K
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import albumentations as A
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# GPU CONFIGURATION
# ============================================================================

def configure_gpu():
    """Configure GPU settings for optimal performance"""
    print("Configuring GPU settings...")
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            for gpu in physical_devices:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✓ GPU found: {physical_devices}")
            return True
        except RuntimeError as e:
            print(f"✗ GPU error: {e}")
            return False
    else:
        print("✗ No GPU found")
        return False

# ============================================================================
# DATA LOADING (Same as before)
# ============================================================================

def load_images_masks(images_folder, masks_folder, img_size=(256,256)):
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
    
    return np.concatenate(all_images), np.concatenate(all_masks)

def create_new_split(all_images, all_masks):
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
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.8),
        A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=20, val_shift_limit=15, p=0.6),
        A.GaussianBlur(blur_limit=(3, 5), p=0.4),
        A.RandomGamma(gamma_limit=(80, 120), p=0.4),
        A.ElasticTransform(alpha=1, sigma=50, p=0.3),  # NEW: Elastic deformation
        A.GridDistortion(p=0.2),  # NEW: Grid distortion
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
# ENHANCED FOCAL-EIoU LOSS
# ============================================================================

def focal_loss(y_true, y_pred, gamma=2.0, alpha=None, epsilon=1e-7):
    if alpha is None:
        alpha = [0.20, 1.2, 1.2]  # ADJUSTED: More penalty on disc/cup
    
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
        
        # Standard IoU
        intersection = K.sum(y_true_c * y_pred_c, axis=[1, 2])
        union = K.sum(y_true_c, axis=[1, 2]) + K.sum(y_pred_c, axis=[1, 2]) - intersection
        iou = (intersection + smooth) / (union + smooth)
        
        # Center distance
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
        
        # Aspect ratio
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

def focal_eiou_combined_loss(focal_weight=0.6, eiou_weight=0.4, gamma=2.0, alpha=None):
    """ADJUSTED: 0.6 Focal + 0.4 EIoU"""
    def loss_fn(y_true, y_pred):
        fl = focal_loss(y_true, y_pred, gamma=gamma, alpha=alpha)
        eiou = enhanced_iou_loss(y_true, y_pred)
        return focal_weight * fl + eiou_weight * eiou
    return loss_fn

# ============================================================================
# ULTRA-DEEP DECODER BLOCK (6 Conv Layers)
# ============================================================================

def ultra_deep_decoder_block(x, skip_connection, filters, block_name, dropout_rate=0.3):
    """
    Ultra-deep decoder block with:
    - 6 convolutional layers
    - 2 residual connections
    - Strategic dropout placement
    - Attention mechanism
    """
    x = layers.UpSampling2D((2, 2), name=f'{block_name}_upsample')(x)
    
    # Align skip connection
    if skip_connection.shape[-1] != x.shape[-1]:
        skip_connection = layers.Conv2D(x.shape[-1], (1, 1), activation='relu', 
                                      padding='same', name=f'{block_name}_skip_align')(skip_connection)
        skip_connection = layers.BatchNormalization(name=f'{block_name}_skip_bn')(skip_connection)
    
    # Concatenate
    x = layers.Concatenate(name=f'{block_name}_concat')([x, skip_connection])
    
    # First residual block (3 layers)
    residual_1 = x
    
    x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same', name=f'{block_name}_conv1')(x)
    x = layers.BatchNormalization(name=f'{block_name}_bn1')(x)
    x = layers.Dropout(dropout_rate, name=f'{block_name}_dropout1')(x)
    
    x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same', name=f'{block_name}_conv2')(x)
    x = layers.BatchNormalization(name=f'{block_name}_bn2')(x)
    
    x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same', name=f'{block_name}_conv3')(x)
    x = layers.BatchNormalization(name=f'{block_name}_bn3')(x)
    
    # Match residual dimensions
    residual_1 = layers.Conv2D(filters, (1, 1), padding='same', name=f'{block_name}_res1_match')(residual_1)
    x = layers.Add(name=f'{block_name}_residual1')([x, residual_1])
    
    # Second residual block (3 layers)
    residual_2 = x
    
    x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same', name=f'{block_name}_conv4')(x)
    x = layers.BatchNormalization(name=f'{block_name}_bn4')(x)
    x = layers.Dropout(dropout_rate * 0.7, name=f'{block_name}_dropout2')(x)
    
    x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same', name=f'{block_name}_conv5')(x)
    x = layers.BatchNormalization(name=f'{block_name}_bn5')(x)
    
    x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same', name=f'{block_name}_conv6')(x)
    x = layers.BatchNormalization(name=f'{block_name}_bn6')(x)
    x = layers.Dropout(dropout_rate * 0.5, name=f'{block_name}_dropout3')(x)
    
    x = layers.Add(name=f'{block_name}_residual2')([x, residual_2])
    
    # Attention mechanism (channel attention)
    gap = layers.GlobalAveragePooling2D(name=f'{block_name}_gap')(x)
    attention = layers.Dense(filters // 8, activation='relu', name=f'{block_name}_att1')(gap)
    attention = layers.Dense(filters, activation='sigmoid', name=f'{block_name}_att2')(attention)
    attention = layers.Reshape((1, 1, filters), name=f'{block_name}_att_reshape')(attention)
    x = layers.Multiply(name=f'{block_name}_att_mul')([x, attention])
    
    # Final 1x1 conv
    x = layers.Conv2D(filters, (1, 1), activation='relu', padding='same', name=f'{block_name}_conv_final')(x)
    x = layers.BatchNormalization(name=f'{block_name}_bn_final')(x)
    
    return x

# ============================================================================
# ULTRA-DEEP COMBINED MODEL
# ============================================================================

def build_ultra_deep_combined_unet(input_shape=(256, 256, 3), num_classes=3):
    """
    Ultra-deep decoder with strategic dropout:
    - Bridge: 0.30
    - Level 1 (16x16): 0.30
    - Level 2 (32x32): 0.21
    - Level 3 (64x64): 0.15
    - Level 4 (128x128): 0.10
    """
    inputs = layers.Input(input_shape, dtype='float32')
    
    # Encoder (MobileNetV2)
    backbone = MobileNetV2(input_tensor=inputs, weights='imagenet', include_top=False)
    
    skip_1 = backbone.get_layer('block_1_expand_relu').output    # 128x128
    skip_2 = backbone.get_layer('block_3_expand_relu').output    # 64x64
    skip_3 = backbone.get_layer('block_6_expand_relu').output    # 32x32
    skip_4 = backbone.get_layer('block_13_expand_relu').output   # 16x16
    
    # Enhanced Bridge (8x8)
    bridge = backbone.output
    bridge = layers.Conv2D(1280, (3, 3), activation='relu', padding='same', name='bridge_conv1')(bridge)
    bridge = layers.BatchNormalization(name='bridge_bn1')(bridge)
    bridge = layers.Dropout(0.30, name='bridge_dropout1')(bridge)  # Strategic dropout
    bridge = layers.Conv2D(1024, (3, 3), activation='relu', padding='same', name='bridge_conv2')(bridge)
    bridge = layers.BatchNormalization(name='bridge_bn2')(bridge)
    bridge = layers.Dropout(0.30, name='bridge_dropout2')(bridge)
    
    # Ultra-Deep Decoder with Strategic Dropout
    x = ultra_deep_decoder_block(bridge, skip_4, 512, 'ultra_decoder_1', dropout_rate=0.30)  # 16x16
    x = ultra_deep_decoder_block(x, skip_3, 256, 'ultra_decoder_2', dropout_rate=0.21)       # 32x32
    x = ultra_deep_decoder_block(x, skip_2, 128, 'ultra_decoder_3', dropout_rate=0.15)       # 64x64
    x = ultra_deep_decoder_block(x, skip_1, 64, 'ultra_decoder_4', dropout_rate=0.10)        # 128x128
    
    # Final upsampling to 256x256
    x = layers.UpSampling2D((2, 2), name='final_upsample')(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='final_conv1')(x)
    x = layers.BatchNormalization(name='final_bn1')(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='final_conv2')(x)
    x = layers.BatchNormalization(name='final_bn2')(x)
    
    # Output
    outputs = layers.Conv2D(num_classes, (1, 1), activation='softmax', dtype='float32', name='output')(x)
    
    model = models.Model(inputs, outputs, name='UltraDeep_MobileNetUNet')
    return model

# ============================================================================
# METRICS (Same as before)
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

# ============================================================================
# MC DROPOUT FUNCTIONS
# ============================================================================

def mc_dropout_predict(model, X, num_samples=20, batch_size=2, verbose=True):
    """MC Dropout with MORE samples (20 instead of 10)"""
    if verbose:
        print(f"\nRunning MC Dropout with {num_samples} passes...")
    
    predictions = []
    for i in range(num_samples):
        if verbose and (i + 1) % 5 == 0:
            print(f"  Pass {i+1}/{num_samples}")
        
        batch_preds = []
        num_batches = int(np.ceil(len(X) / batch_size))
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(X))
            X_batch = X[start_idx:end_idx]
            pred_batch = model(X_batch, training=True).numpy()
            batch_preds.append(pred_batch)
        
        pred = np.concatenate(batch_preds, axis=0)
        predictions.append(pred)
    
    all_preds = np.array(predictions)
    mean_pred = np.mean(all_preds, axis=0)
    std_pred = np.std(all_preds, axis=0)
    
    if verbose:
        print(f"✓ MC Dropout completed")
    
    return mean_pred, std_pred

# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_ultra_deep_model(X_train, y_train, X_val, y_val, augment=None):
    """Train ultra-deep model with optimized hyperparameters"""
    print(f"\n{'='*80}")
    print(f"TRAINING: ULTRA-DEEP MOBILENET-UNET")
    print(f"{'='*80}")
    print(f"  Architecture: 6-layer decoder blocks + attention")
    print(f"  Loss: 0.6×Focal(γ=2.0) + 0.4×EIoU")
    print(f"  Dropout: [0.30, 0.30, 0.21, 0.15, 0.10] (strategic)")
    print(f"  Optimizer: Adam(lr=3e-4)")
    print(f"  Batch Size: 2")
    print(f"  Epochs: 60 (patience=12)")
    
    K.clear_session()
    
    # Build model
    model = build_ultra_deep_combined_unet()
    print(f"\n  Total parameters: {model.count_params():,}")
    
    # Compile with optimized hyperparameters
    model.compile(
        optimizer=optimizers.Adam(learning_rate=3e-4),  # Lower LR for stability
        loss=focal_eiou_combined_loss(focal_weight=0.6, eiou_weight=0.4, gamma=2.0, alpha=[0.20, 1.2, 1.2]),
        metrics=[dice_coef_multiclass, dice_class_0, dice_class_1, dice_class_2,
                "accuracy", iou_coef_multiclass, iou_class_0, iou_class_1, iou_class_2]
    )
    
    # Callbacks with extended patience
    checkpoint = ModelCheckpoint("ultra_deep_model.h5", save_best_only=True, verbose=1, 
                                monitor='val_dice_class_2', mode='max')  # Monitor CUP dice
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.4, patience=5, 
                                  verbose=1, mode='min', min_lr=1e-7)
    early_stop = EarlyStopping(monitor="val_loss", patience=12, verbose=1,  # Extended patience
                              restore_best_weights=True, mode='min')
    
    # Training parameters
    batch_size = 2  # Memory-constrained for ultra-deep
    steps_train = X_train.shape[0] // batch_size
    steps_val = X_val.shape[0] // batch_size
    
    print(f"  Batch size: {batch_size}")
    print(f"  Steps/epoch: {steps_train}")
    
    # Training
    try:
        history = model.fit(
            data_generator(X_train, y_train, batch_size, augment),
            validation_data=data_generator(X_val, y_val, batch_size, None),
            steps_per_epoch=steps_train,
            validation_steps=steps_val,
            epochs=60,  # More epochs
            callbacks=[checkpoint, reduce_lr, early_stop],
            verbose=1
        )
    except Exception as e:
        print(f"✗ Training error: {e}")
        raise
    
    print(f"✓ Training completed")
    return model, history

# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_ultra_deep_model(model, X_test, y_test):
    """Comprehensive evaluation with MC Dropout"""
    print(f"\n{'='*80}")
    print(f"EVALUATION: ULTRA-DEEP MODEL")
    print(f"{'='*80}")
    
    y_test_cat = np.array([to_categorical(mask, num_classes=3) for mask in y_test])
    
    # Standard evaluation
    print("\nStandard Evaluation:")
    test_metrics = model.evaluate(X_test, y_test_cat, verbose=0, batch_size=4)
    metric_names = model.metrics_names
    
    for name, val in zip(metric_names, test_metrics):
        print(f"  {name}: {val:.4f}")
    
    y_pred = model.predict(X_test, verbose=0, batch_size=4)
    
    # Per-class metrics
    print("\nPer-Class Metrics:")
    class_names = ["Background", "Disc", "Cup"]
    results = {}
    
    for cidx, cname in enumerate(class_names):
        y_true_c = (np.argmax(y_test_cat, axis=-1) == cidx).astype(np.float32)
        y_pred_c = (np.argmax(y_pred, axis=-1) == cidx).astype(np.float32)
        
        intersection = np.sum(y_true_c * y_pred_c)
        dice = (2. * intersection + 1e-7) / (np.sum(y_true_c) + np.sum(y_pred_c) + 1e-7)
        union = np.sum(y_true_c) + np.sum(y_pred_c) - intersection
        iou = (intersection + 1e-7) / (union + 1e-7)
        
        results[cname] = {'dice': dice, 'iou': iou}
        print(f"  {cname}: Dice={dice:.4f}, IoU={iou:.4f}")
    
    # MC Dropout evaluation (200 samples, 20 passes)
    print("\nMC Dropout Evaluation (200 samples, 20 passes):")
    mean_pred, std_pred = mc_dropout_predict(model, X_test[:200], num_samples=20, batch_size=2, verbose=True)
    
    for cidx, cname in enumerate(class_names):
        mean_std = np.mean(std_pred[:, :, :, cidx])
        max_std = np.max(std_pred[:, :, :, cidx])
        print(f"  {cname}: Mean Uncertainty={mean_std:.4f}, Max Uncertainty={max_std:.4f}")
    
    return results

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Configure GPU
    gpu_available = configure_gpu()
    
    # Dataset path
    root_dir = "/kaggle/input/refuge/REFUGE/"
    
    print(f"\n{'='*100}")
    print(f"{'ULTRA-DEEP MOBILENET-UNET WITH AGGRESSIVE IMPROVEMENTS':^100s}")
    print(f"{'='*100}")
    
    # Load data
    all_images, all_masks = load_all_data(root_dir)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = create_new_split(all_images, all_masks)
    
    # Setup enhanced augmentation
    train_aug = setup_augmentations()
    
    # Train ultra-deep model
    model, history = train_ultra_deep_model(X_train, y_train, X_val, y_val, augment=train_aug)
    
    # Evaluate
    results = evaluate_ultra_deep_model(model, X_test, y_test)
    
    print(f"\n{'='*100}")
    print(f"{'✓ ULTRA-DEEP MODEL TRAINING & EVALUATION COMPLETED!':^100s}")
    print(f"{'='*100}")
    print(f"\nFINAL RESULTS:")
    print(f"  Disc Dice: {results['Disc']['dice']:.4f}")
    print(f"  Cup Dice: {results['Cup']['dice']:.4f}")
    print(f"  Overall: {(results['Disc']['dice'] + results['Cup']['dice'])/2:.4f}")
    print(f"{'='*100}\n")