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
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import albumentations as A

# ---- Improved GPU Configuration ----
def configure_gpu():
    """Configure GPU settings for optimal performance"""
    print("Configuring GPU settings...")
    
    # Force GPU allocation
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            # Enable memory growth to avoid CUDA errors
            for gpu in physical_devices:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Set device placement
            tf.config.set_visible_devices(physical_devices[0], 'GPU')
            
            print(f"GPU found: {physical_devices}")
            print("GPU device name:", tf.test.gpu_device_name())
            print("Is built with CUDA:", tf.test.is_built_with_cuda())
            
            # Test GPU allocation
            with tf.device('/GPU:0'):
                test_tensor = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
                result = tf.matmul(test_tensor, test_tensor, transpose_b=True)
                print("GPU test successful:", result.numpy().shape)
            
            return True
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
            return False
    else:
        print("No GPU found, using CPU")
        return False

# ---- Data Loading Functions ----
def load_images_masks(images_folder, masks_folder, img_size=(256,256)):
    """Load images and masks from folders"""
    images, masks = [], []
    image_files = sorted(os.listdir(images_folder))
    mask_files = sorted(os.listdir(masks_folder))
    mask_dict = {os.path.splitext(f)[0]: f for f in mask_files}
    
    for img_file in image_files:
        base = os.path.splitext(img_file)[0]
        if base in mask_dict:
            # Load and preprocess image
            img = cv2.imread(os.path.join(images_folder, img_file))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, img_size) / 255.0
            
            # Load mask - keep as single channel for augmentation
            mask = cv2.imread(os.path.join(masks_folder, mask_dict[base]), cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, img_size)
            
            # Verify mask values
            unique_vals = np.unique(mask)
            if len(unique_vals) > 3 or np.max(unique_vals) > 2:
                print(f"Warning: Mask {mask_dict[base]} has unexpected values: {unique_vals}")
                # Normalize mask values to 0, 1, 2
                mask = np.clip(mask // 85, 0, 2).astype(np.uint8)
            
            images.append(img)
            masks.append(mask)
    
    return np.array(images, dtype=np.float32), np.array(masks, dtype=np.uint8)

def load_all_data(root_dir):
    """Load all data from existing splits and merge"""
    root_dir = "/kaggle/input/refuge/REFUGE/"
    all_images = []
    all_masks = []
    
    for split in ["train", "val", "test"]:
        print(f"Loading {split} data...")
        imgs, msks = load_images_masks(
            os.path.join(root_dir, split, "Images"),
            os.path.join(root_dir, split, "Masks")
        )
        all_images.append(imgs)
        all_masks.append(msks)
        print(f"{split}: {imgs.shape[0]} samples")
        print(f"Mask value range: {np.min(msks)} - {np.max(msks)}")
    
    all_images = np.concatenate(all_images, axis=0)
    all_masks = np.concatenate(all_masks, axis=0)
    print(f"Total samples: {all_images.shape[0]}")
    print(f"Final mask value distribution: {np.bincount(all_masks.flatten())}")
    
    return all_images, all_masks

def create_new_split(all_images, all_masks):
    """Create new split: 600 train, 200 val, 400 test"""
    print("Creating new split (600-200-400)...")
    
    # Split: 600 train, 200 val, 400 test (randomized)
    X_temp, X_test, y_temp, y_test = train_test_split(
        all_images, all_masks, test_size=400, random_state=42, stratify=None
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=200, random_state=42, stratify=None
    )
    
    print(f"New split - Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

# ---- Augmentation Setup ----
def setup_augmentations():
    """Setup augmentations for training"""
    train_aug = A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.7),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.5),
        A.GaussianBlur(blur_limit=3, p=0.3),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
    ])
    
    return train_aug

# ---- Data Generator ----
def data_generator(images, masks, batch_size, augment=None):
    """Data generator with optional augmentation"""
    idxs = np.arange(len(images))
    while True:
        np.random.shuffle(idxs)
        for i in range(0, len(images), batch_size):
            batch_idxs = idxs[i:i+batch_size]
            batch_x, batch_y = [], []
            
            for idx in batch_idxs:
                img, mask = images[idx], masks[idx]
                
                if augment:
                    # Apply augmentation to single-channel mask
                    aug = augment(image=img, mask=mask)
                    img_aug = aug['image']
                    mask_aug = aug['mask']
                else:
                    img_aug, mask_aug = img, mask
                
                # Convert mask to categorical after augmentation
                mask_cat = to_categorical(mask_aug, num_classes=3)
                
                batch_x.append(img_aug)
                batch_y.append(mask_cat)
            
            # Ensure consistent float32 dtype
            batch_x = np.stack(batch_x).astype(np.float32)
            batch_y = np.stack(batch_y).astype(np.float32)
            
            yield batch_x, batch_y

# ---- Enhanced Decoder Block ----
def enhanced_decoder_block(x, skip_connection, filters, block_name, dropout_rate=0.3):
    """Enhanced decoder block with more layers and better feature processing"""
    
    print(f"Building enhanced decoder block: {block_name} with {filters} filters")
    
    # Upsampling
    x = layers.UpSampling2D((2, 2), name=f'{block_name}_upsample')(x)
    
    # Skip connection fusion with 1x1 conv for channel alignment if needed
    if skip_connection.shape[-1] != x.shape[-1]:
        skip_connection = layers.Conv2D(x.shape[-1], (1, 1), activation='relu', 
                                      padding='same', name=f'{block_name}_skip_align')(skip_connection)
        skip_connection = layers.BatchNormalization(name=f'{block_name}_skip_bn')(skip_connection)
    
    # Concatenate features
    x = layers.Concatenate(name=f'{block_name}_concat')([x, skip_connection])
    
    # First conv block - reduce channels and process concatenated features
    x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same', name=f'{block_name}_conv1')(x)
    x = layers.BatchNormalization(name=f'{block_name}_bn1')(x)
    x = layers.Dropout(dropout_rate, name=f'{block_name}_dropout1')(x)
    
    # Second conv block - refine features
    x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same', name=f'{block_name}_conv2')(x)
    x = layers.BatchNormalization(name=f'{block_name}_bn2')(x)
    
    # Third conv block - additional feature refinement (NEW)
    x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same', name=f'{block_name}_conv3')(x)
    x = layers.BatchNormalization(name=f'{block_name}_bn3')(x)
    
    # Fourth conv block with residual connection - deep feature learning (NEW)
    residual = x
    x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same', name=f'{block_name}_conv4')(x)
    x = layers.BatchNormalization(name=f'{block_name}_bn4')(x)
    x = layers.Dropout(dropout_rate, name=f'{block_name}_dropout2')(x)
    
    # Residual connection
    x = layers.Add(name=f'{block_name}_residual')([x, residual])
    
    # Final refinement conv - enhance boundaries (NEW)
    x = layers.Conv2D(filters, (1, 1), activation='relu', padding='same', name=f'{block_name}_conv_final')(x)
    x = layers.BatchNormalization(name=f'{block_name}_bn_final')(x)
    
    return x

# ---- Deep Decoder Block (for critical layers) ----
def deep_decoder_block(x, skip_connection, filters, block_name, dropout_rate=0.3):
    """Extra deep decoder block for critical layers (Class 1 & 2 focused)"""
    
    print(f"Building DEEP decoder block: {block_name} with {filters} filters")
    
    # Upsampling
    x = layers.UpSampling2D((2, 2), name=f'{block_name}_upsample')(x)
    
    # Skip connection processing - enhance skip features
    skip_processed = layers.Conv2D(filters//2, (1, 1), activation='relu', 
                                 padding='same', name=f'{block_name}_skip_process')(skip_connection)
    skip_processed = layers.BatchNormalization(name=f'{block_name}_skip_bn')(skip_processed)
    
    # Multi-scale feature extraction from upsampled features
    x_1x1 = layers.Conv2D(filters//4, (1, 1), activation='relu', padding='same', name=f'{block_name}_1x1')(x)
    x_3x3 = layers.Conv2D(filters//4, (3, 3), activation='relu', padding='same', name=f'{block_name}_3x3')(x)
    x_5x5 = layers.Conv2D(filters//4, (5, 5), activation='relu', padding='same', name=f'{block_name}_5x5')(x)
    x_pool = layers.MaxPooling2D((3, 3), strides=1, padding='same', name=f'{block_name}_pool')(x)
    x_pool = layers.Conv2D(filters//4, (1, 1), activation='relu', padding='same', name=f'{block_name}_pool_conv')(x_pool)
    
    # Concatenate multi-scale features
    x_multi = layers.Concatenate(name=f'{block_name}_multi_concat')([x_1x1, x_3x3, x_5x5, x_pool])
    x_multi = layers.BatchNormalization(name=f'{block_name}_multi_bn')(x_multi)
    
    # Concatenate with processed skip connection
    x = layers.Concatenate(name=f'{block_name}_concat')([x_multi, skip_processed])
    
    # Deep processing layers
    for i in range(6):  # 6 layers for very deep processing
        x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same', 
                         name=f'{block_name}_deep_conv_{i+1}')(x)
        x = layers.BatchNormalization(name=f'{block_name}_deep_bn_{i+1}')(x)
        
        # Add dropout every 2 layers
        if i % 2 == 1:
            x = layers.Dropout(dropout_rate, name=f'{block_name}_deep_dropout_{i+1}')(x)
        
        # Add residual connections every 3 layers
        if i == 2:
            residual_1 = x
        elif i == 5:
            x = layers.Add(name=f'{block_name}_deep_residual')([x, residual_1])
    
    # Final boundary enhancement
    x = layers.Conv2D(filters, (1, 1), activation='relu', padding='same', name=f'{block_name}_boundary_conv')(x)
    x = layers.BatchNormalization(name=f'{block_name}_boundary_bn')(x)
    
    return x

# ---- Enhanced MobileNet-UNet with Deeper Decoder ----
def build_mobilenet_unet_deeper(input_shape=(256, 256, 3), num_classes=3):
    """Build MobileNet-UNet with much deeper decoder blocks"""
    
    print("Building MobileNet-UNet with Enhanced Deeper Decoder...")
    
    # Input
    inputs = layers.Input(input_shape, dtype='float32')
    
    # Encoder (MobileNetV2 backbone)
    backbone = MobileNetV2(input_tensor=inputs, weights='imagenet', include_top=False)
    
    # Extract skip connections from encoder layers
    skip_1 = backbone.get_layer('block_1_expand_relu').output    # 128x128
    skip_2 = backbone.get_layer('block_3_expand_relu').output    # 64x64
    skip_3 = backbone.get_layer('block_6_expand_relu').output    # 32x32
    skip_4 = backbone.get_layer('block_13_expand_relu').output   # 16x16
    
    # Bridge (bottom of U) with enhanced processing
    bridge = backbone.output  # 8x8
    print(f"Bridge shape: {bridge.shape}")
    
    # Enhanced bridge processing
    bridge = layers.Conv2D(1024, (3, 3), activation='relu', padding='same', name='bridge_conv1')(bridge)
    bridge = layers.BatchNormalization(name='bridge_bn1')(bridge)
    bridge = layers.Conv2D(1024, (3, 3), activation='relu', padding='same', name='bridge_conv2')(bridge)
    bridge = layers.BatchNormalization(name='bridge_bn2')(bridge)
    bridge = layers.Dropout(0.4, name='bridge_dropout')(bridge)
    
    print("Building decoder with enhanced blocks...")
    
    # Decoder Layer 1: 8x8 -> 16x16 (Enhanced block)
    x = enhanced_decoder_block(bridge, skip_4, 512, 'decoder_1', dropout_rate=0.3)
    
    # Decoder Layer 2: 16x16 -> 32x32 (Enhanced block)
    x = enhanced_decoder_block(x, skip_3, 256, 'decoder_2', dropout_rate=0.3)
    
    # Decoder Layer 3: 32x32 -> 64x64 (DEEP block - CRITICAL for Class 1 & 2)
    x = deep_decoder_block(x, skip_2, 128, 'decoder_3_deep', dropout_rate=0.2)
    
    # Decoder Layer 4: 64x64 -> 128x128 (DEEP block - CRITICAL for fine details)
    x = deep_decoder_block(x, skip_1, 64, 'decoder_4_deep', dropout_rate=0.2)
    
    # Final upsampling: 128x128 -> 256x256 with enhanced processing
    x = layers.UpSampling2D((2, 2), name='final_upsample')(x)
    
    # Final processing layers - multiple scales for boundary refinement
    x_fine = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='final_conv_fine')(x)
    x_fine = layers.BatchNormalization(name='final_bn_fine')(x_fine)
    
    x_coarse = layers.Conv2D(32, (5, 5), activation='relu', padding='same', name='final_conv_coarse')(x)
    x_coarse = layers.BatchNormalization(name='final_bn_coarse')(x_coarse)
    
    x_detail = layers.Conv2D(32, (1, 1), activation='relu', padding='same', name='final_conv_detail')(x)
    x_detail = layers.BatchNormalization(name='final_bn_detail')(x_detail)
    
    # Combine multi-scale final features
    x = layers.Add(name='final_combine')([x_fine, x_coarse, x_detail])
    x = layers.BatchNormalization(name='final_combine_bn')(x)
    
    # Final boundary enhancement for Class 1 & 2
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='boundary_enhance_1')(x)
    x = layers.BatchNormalization(name='boundary_enhance_bn1')(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='boundary_enhance_2')(x)
    x = layers.BatchNormalization(name='boundary_enhance_bn2')(x)
    
    # Output layer - ensure float32
    outputs = layers.Conv2D(num_classes, (1, 1), activation='softmax', name='output', dtype='float32')(x)
    
    model = models.Model(inputs, outputs, name="MobileNet-UNet-Deeper")
    
    print(f"Enhanced deeper model built successfully!")
    print(f"Total parameters: {model.count_params():,}")
    
    return model

# ---- Original MobileNet-UNet Model (for comparison) ----
def build_mobilenet_unet_original(input_shape=(256, 256, 3), num_classes=3):
    """Build original MobileNet-UNet without enhancements"""
    
    # Input
    inputs = layers.Input(input_shape, dtype='float32')
    
    # Encoder (MobileNetV2 backbone)
    backbone = MobileNetV2(input_tensor=inputs, weights='imagenet', include_top=False)
    
    # Extract skip connections from encoder layers
    skip_connections = [
        backbone.get_layer('block_1_expand_relu').output,    # Layer 1: 128x128
        backbone.get_layer('block_3_expand_relu').output,    # Layer 2: 64x64
        backbone.get_layer('block_6_expand_relu').output,    # Layer 3: 32x32
        backbone.get_layer('block_13_expand_relu').output,   # Layer 4: 16x16
    ]
    
    # Bridge (bottom of U)
    bridge = backbone.output  # 8x8
    
    # Decoder with 4 layers
    # Decoder Layer 1: 8x8 -> 16x16
    x = layers.UpSampling2D((2, 2))(bridge)
    x = layers.Concatenate()([x, skip_connections[3]])
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    # Decoder Layer 2: 16x16 -> 32x32
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Concatenate()([x, skip_connections[2]])
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    # Decoder Layer 3: 32x32 -> 64x64
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Concatenate()([x, skip_connections[1]])
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    # Decoder Layer 4: 64x64 -> 128x128
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Concatenate()([x, skip_connections[0]])
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    # Final upsampling to original size: 128x128 -> 256x256
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    # Output layer - ensure float32
    outputs = layers.Conv2D(num_classes, (1, 1), activation='softmax', name='output', dtype='float32')(x)
    
    model = models.Model(inputs, outputs, name="MobileNet-UNet-Original")
    return model

# ---- Metrics ----
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

# ---- Numpy-based metric calculations ----
def dice_coef_multiclass_np(y_true, y_pred, smooth=1e-7):
    """Calculate overall multiclass Dice coefficient using numpy"""
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def iou_coef_multiclass_np(y_true, y_pred, smooth=1e-7):
    """Calculate overall multiclass IoU coefficient using numpy"""
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    union = np.sum(y_true_f) + np.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

def dice_coef_class_explicit_np(y_true, y_pred, class_idx, smooth=1e-7):
    """Calculate per-class Dice coefficient using numpy"""
    y_true_c = (np.argmax(y_true, axis=-1) == class_idx).astype(np.float32)
    y_pred_c = (np.argmax(y_pred, axis=-1) == class_idx).astype(np.float32)
    intersection = np.sum(y_true_c * y_pred_c)
    dice = (2. * intersection + smooth) / (np.sum(y_true_c) + np.sum(y_pred_c) + smooth)
    return dice

def iou_coef_class_explicit_np(y_true, y_pred, class_idx, smooth=1e-7):
    """Calculate per-class IoU coefficient using numpy"""
    y_true_c = (np.argmax(y_true, axis=-1) == class_idx).astype(np.float32)
    y_pred_c = (np.argmax(y_pred, axis=-1) == class_idx).astype(np.float32)
    intersection = np.sum(y_true_c * y_pred_c)
    union = np.sum(y_true_c) + np.sum(y_pred_c) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou

# ---- Training Function ----
def train_model(X_train, y_train, X_val, y_val, model_name, use_deeper=False, augment=None, gpu_available=True):
    """Train MobileNet-UNet model with or without deeper decoder"""
    
    print(f"\nTraining {model_name}...")
    print(f"Using deeper decoder: {'Yes' if use_deeper else 'No'}")
    print(f"Using augmentation: {'Yes' if augment else 'No'}")
    
    # Clear any previous models
    K.clear_session()
    
    # Build model
    if use_deeper:
        model = build_mobilenet_unet_deeper()
    else:
        model = build_mobilenet_unet_original()
    
    # Optimizer with different learning rate for deeper model
    initial_lr = 5e-4 if use_deeper else 1e-3  # Lower LR for deeper model
    optimizer = optimizers.Adam(learning_rate=initial_lr)
    
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=[
            dice_coef_multiclass, dice_class_0, dice_class_1, dice_class_2, "accuracy",
            iou_coef_multiclass, iou_class_0, iou_class_1, iou_class_2
        ]
    )
    
    # Callbacks with adjusted patience for deeper model
    checkpoint = ModelCheckpoint(
        f"{model_name}.h5", 
        save_best_only=True, 
        verbose=1, 
        monitor='val_loss',
        mode='min'
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=4 if use_deeper else 3,  # More patience for deeper model
        verbose=1,
        mode='min',
        min_lr=1e-8
    )
    
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=12 if use_deeper else 8,  # More patience for deeper model
        verbose=1, 
        restore_best_weights=True,
        mode='min'
    )
    
    # Set batch size (smaller for deeper model due to memory usage)
    batch_size = 4 if use_deeper else 8
    steps_train = X_train.shape[0] // batch_size
    steps_val = X_val.shape[0] // batch_size
    
    print(f"Using batch size: {batch_size}")
    print(f"Steps per epoch - Train: {steps_train}, Val: {steps_val}")
    
    # Training
    try:
        history = model.fit(
            data_generator(X_train, y_train, batch_size, augment=augment),
            validation_data=data_generator(X_val, y_val, batch_size, augment=None),
            steps_per_epoch=steps_train,
            validation_steps=steps_val,
            epochs=60 if use_deeper else 50,  # More epochs for deeper model
            callbacks=[checkpoint, reduce_lr, early_stop],
            verbose=1
        )
    except Exception as e:
        print(f"Training error: {e}")
        print("Reducing batch size...")
        batch_size = 2 if use_deeper else 4
        steps_train = X_train.shape[0] // batch_size
        steps_val = X_val.shape[0] // batch_size
        
        history = model.fit(
            data_generator(X_train, y_train, batch_size, augment=augment),
            validation_data=data_generator(X_val, y_val, batch_size, augment=None),
            steps_per_epoch=steps_train,
            validation_steps=steps_val,
            epochs=60 if use_deeper else 50,
            callbacks=[checkpoint, reduce_lr, early_stop],
            verbose=1
        )
    
    return model, history

# ---- Evaluation Functions ----
def calculate_flops_manual(model, input_shape=(1, 256, 256, 3)):
    """Manual FLOPS calculation for the model"""
    try:
        total_flops = 0
        current_h, current_w = input_shape[1], input_shape[2]
        
        print(f"Calculating FLOPS for model with {len(model.layers)} layers...")
        
        for i, layer in enumerate(model.layers):
            layer_flops = 0
            
            if isinstance(layer, layers.Conv2D):
                config = layer.get_config()
                kernel_size = config['kernel_size']
                filters = config['filters']
                strides = config['strides']
                
                if hasattr(layer, 'input_spec') and layer.input_spec is not None:
                    if hasattr(layer.input_spec, 'shape') and layer.input_spec.shape is not None:
                        input_channels = layer.input_spec.shape[-1]
                    else:
                        input_channels = 3 if i == 0 else 32
                else:
                    input_channels = 3 if i == 0 else 32
                
                output_h = current_h // strides[0] if strides[0] > 1 else current_h
                output_w = current_w // strides[1] if strides[1] > 1 else current_w
                
                kernel_flops = kernel_size[0] * kernel_size[1]
                layer_flops = output_h * output_w * kernel_flops * input_channels * filters
                
                current_h, current_w = output_h, output_w
                
            elif isinstance(layer, layers.UpSampling2D):
                config = layer.get_config()
                size = config['size']
                current_h *= size[0]
                current_w *= size[1]
                layer_flops = 0
                
            elif isinstance(layer, layers.MaxPooling2D):
                config = layer.get_config()
                pool_size = config['pool_size']
                strides = config.get('strides', pool_size)
                current_h = current_h // strides[0]
                current_w = current_w // strides[1]
                layer_flops = 0
                
            elif isinstance(layer, layers.BatchNormalization):
                layer_flops = current_h * current_w * 2
                
            elif isinstance(layer, layers.Activation) or 'relu' in layer.name.lower():
                layer_flops = current_h * current_w
            
            total_flops += layer_flops
            
            if layer_flops > 0 and i < 10:
                print(f"Layer {i} ({layer.__class__.__name__}): {layer_flops/1e6:.2f} MFLOPs, Output: {current_h}x{current_w}")
        
        gflops = total_flops / 1e9
        print(f"Total estimated FLOPS: {total_flops:.0f} ({gflops:.2f} GFLOPs)")
        return gflops
        
    except Exception as e:
        print(f"FLOPS calculation failed: {e}")
        try:
            params = model.count_params()
            estimated_flops = params * 2
            gflops = estimated_flops / 1e9
            print(f"Simplified FLOPS estimate: {estimated_flops:.0f} ({gflops:.2f} GFLOPs)")
            return gflops
        except Exception as e2:
            print(f"Simplified estimation also failed: {e2}")
            return None

def evaluate_model(model, X_test, y_test, model_name, gpu_available=True):
    """Comprehensive model evaluation with all required metrics"""
    
    print(f"\nEvaluating {model_name}...")
    
    # Convert test masks to categorical for evaluation
    y_test_cat = np.array([to_categorical(mask, num_classes=3) for mask in y_test])
    
    # Test metrics using model.evaluate()
    test_metrics = model.evaluate(X_test, y_test_cat, verbose=1, batch_size=4)
    metric_names = model.metrics_names
    
    print("\nTest Metrics:")
    for name, val in zip(metric_names, test_metrics):
        print(f"{name}: {val:.4f}")
    
    # Get predictions for additional calculations
    y_pred_prob = model.predict(X_test, verbose=1, batch_size=4)
    
    # Calculate overall Dice and IoU using numpy
    overall_dice = dice_coef_multiclass_np(y_test_cat, y_pred_prob)
    overall_iou = iou_coef_multiclass_np(y_test_cat, y_pred_prob)
    
    print(f"\nOverall multiclass Dice: {overall_dice:.4f}")
    print(f"Overall multiclass IoU: {overall_iou:.4f}")
    
    # Per-class Dice and IoU on TEST set
    print(f"\nPer-class Dice on TEST set for {model_name}:")
    class_names = ["Background", "Disc", "Cup"]
    class_results = {}
    for cidx, cname in enumerate(class_names):
        dice = dice_coef_class_explicit_np(y_test_cat, y_pred_prob, cidx)
        iou = iou_coef_class_explicit_np(y_test_cat, y_pred_prob, cidx)
        class_results[cname] = {'dice': dice, 'iou': iou}
        print(f"{cname} - Dice: {dice:.4f}, IoU: {iou:.4f}")
    
    # Additional sklearn metrics
    y_true_flat = np.reshape(y_test, (-1,))
    y_pred_flat = np.reshape(np.argmax(y_pred_prob, axis=-1), (-1,))
    
    precision = precision_score(y_true_flat, y_pred_flat, average='weighted', zero_division=0)
    recall = recall_score(y_true_flat, y_pred_flat, average='weighted', zero_division=0)
    fscore = f1_score(y_true_flat, y_pred_flat, average='weighted', zero_division=0)
    cm = confusion_matrix(y_true_flat, y_pred_flat)
    
    print(f"\nPrecision (weighted): {precision:.4f}")
    print(f"Recall (weighted): {recall:.4f}")
    print(f"F-score (weighted): {fscore:.4f}")
    print("Confusion Matrix:")
    print(cm)
    
    # Model complexity
    print(f"\nTotal Parameters: {model.count_params():,}")
    
    # Manual FLOPS calculation
    manual_flops = calculate_flops_manual(model)
    if manual_flops:
        print(f"GFLOPS (manual estimate): {manual_flops:.2f}")
    
    return {
        'test_metrics': test_metrics,
        'metric_names': metric_names,
        'overall_dice': overall_dice,
        'overall_iou': overall_iou,
        'class_results': class_results,
        'precision': precision,
        'recall': recall,
        'fscore': fscore
    }

def plot_training_history(history, title_suffix=""):
    """Plot training curves"""
    
    plt.figure(figsize=(20, 5))
    
    # Loss plot
    plt.subplot(1, 4, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'Loss over Epochs {title_suffix}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Accuracy plot
    plt.subplot(1, 4, 2)
    if 'accuracy' in history.history:
        plt.plot(history.history['accuracy'], label='Train Acc')
        plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title(f'Accuracy over Epochs {title_suffix}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Dice coefficient plot
    plt.subplot(1, 4, 3)
    if 'dice_coef_multiclass' in history.history:
        plt.plot(history.history['dice_coef_multiclass'], label='Train Dice')
        plt.plot(history.history['val_dice_coef_multiclass'], label='Val Dice')
    plt.title(f'Dice Coefficient over Epochs {title_suffix}')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Coefficient')
    plt.legend()
    plt.grid(True)
    
    # Learning rate plot
    plt.subplot(1, 4, 4)
    if 'lr' in history.history:
        plt.plot(history.history['lr'], label='Learning Rate')
    plt.title(f'Learning Rate over Epochs {title_suffix}')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def visualize_predictions(model, X_test, y_test, model_name, num_samples=3):
    """Visualize sample predictions"""
    
    print(f"\nSample predictions for {model_name}:")
    
    # Generate predictions
    preds = model.predict(X_test[:num_samples], batch_size=num_samples)
    
    for i in range(num_samples):
        plt.figure(figsize=(16, 4))
        
        plt.subplot(1, 4, 1)
        plt.imshow(X_test[i])
        plt.title("Original Image")
        plt.axis('off')
        
        plt.subplot(1, 4, 2)
        plt.imshow(y_test[i], cmap='jet', vmin=0, vmax=2)
        plt.title("Ground Truth")
        plt.axis('off')
        
        plt.subplot(1, 4, 3)
        plt.imshow(np.argmax(preds[i], axis=-1), cmap='jet', vmin=0, vmax=2)
        plt.title("Prediction")
        plt.axis('off')
        
        plt.subplot(1, 4, 4)
        overlay = X_test[i].copy()
        pred_mask = np.argmax(preds[i], axis=-1)
        overlay[pred_mask == 1] = [1, 0, 0]  # Red for disc
        overlay[pred_mask == 2] = [0, 1, 0]  # Green for cup
        plt.imshow(overlay)
        plt.title("Overlay")
        plt.axis('off')
        
        plt.suptitle(f"{model_name} - Sample {i+1}")
        plt.show()

def compare_models(results_original, results_deeper):
    """Compare original vs deeper decoder results"""
    
    print("\n" + "="*80)
    print("COMPARISON: ORIGINAL vs DEEPER DECODER")
    print("="*80)
    
    print(f"{'Metric':<20} {'Original':<12} {'Deeper':<12} {'Improvement':<15}")
    print("-" * 60)
    
    # Overall metrics
    orig_dice = results_original['overall_dice']
    deeper_dice = results_deeper['overall_dice']
    dice_improvement = ((deeper_dice - orig_dice) / orig_dice) * 100
    
    orig_iou = results_original['overall_iou']
    deeper_iou = results_deeper['overall_iou']
    iou_improvement = ((deeper_iou - orig_iou) / orig_iou) * 100
    
    print(f"{'Overall Dice':<20} {orig_dice:<12.4f} {deeper_dice:<12.4f} {dice_improvement:+.3f}%")
    print(f"{'Overall IoU':<20} {orig_iou:<12.4f} {deeper_iou:<12.4f} {iou_improvement:+.3f}%")
    
    print("\n--- CRITICAL CLASS IMPROVEMENTS (Disc & Cup) ---")
    for class_name in ["Disc", "Cup"]:
        if class_name in results_original['class_results']:
            orig_dice_class = results_original['class_results'][class_name]['dice']
            deeper_dice_class = results_deeper['class_results'][class_name]['dice']
            dice_class_improvement = ((deeper_dice_class - orig_dice_class) / orig_dice_class) * 100
            
            orig_iou_class = results_original['class_results'][class_name]['iou']
            deeper_iou_class = results_deeper['class_results'][class_name]['iou']
            iou_class_improvement = ((deeper_iou_class - orig_iou_class) / orig_iou_class) * 100
            
            print(f"\n{class_name} Class:")
            print(f"{'  Dice':<18} {orig_dice_class:<12.4f} {deeper_dice_class:<12.4f} {dice_class_improvement:+.3f}%")
            print(f"{'  IoU':<18} {orig_iou_class:<12.4f} {deeper_iou_class:<12.4f} {iou_class_improvement:+.3f}%")

# ---- Main Execution ----
if __name__ == "__main__":
    # Configure GPU first
    gpu_available = configure_gpu()
    
    # Dataset path - UPDATE THIS TO YOUR PATH
    root_dir = "/kaggle/input/refuge/REFUGE/"  # Change this to your actual path
    
    # Setup augmentations
    train_aug = setup_augmentations()
    
    print("="*80)
    print("MOBILENET-UNET: DEEPER DECODER EXPERIMENT")
    print(f"GPU Available: {gpu_available}")
    print("="*80)
    
    # Load and prepare data
    all_images, all_masks = load_all_data(root_dir)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = create_new_split(all_images, all_masks)
    
    # Train Original Model
    print("\n" + "="*60)
    print("TRAINING: ORIGINAL MODEL (Baseline)")
    print("="*60)
    
    model_original, history_original = train_model(
        X_train, y_train, X_val, y_val,
        "mobilenet_unet_original", use_deeper=False, augment=train_aug, gpu_available=gpu_available
    )
    
    # Train Deeper Decoder Model
    print("\n" + "="*60)
    print("TRAINING: DEEPER DECODER MODEL")
    print("="*60)
    
    model_deeper, history_deeper = train_model(
        X_train, y_train, X_val, y_val,
        "mobilenet_unet_deeper", use_deeper=True, augment=train_aug, gpu_available=gpu_available
    )
    
    # Evaluate Original Model
    print("\n" + "="*60)
    print("EVALUATION: ORIGINAL MODEL")
    print("="*60)
    
    results_original = evaluate_model(
        model_original, X_test, y_test, "Original MobileNet-UNet", gpu_available
    )
    
    # Evaluate Deeper Model
    print("\n" + "="*60)
    print("EVALUATION: DEEPER DECODER MODEL")
    print("="*60)
    
    results_deeper = evaluate_model(
        model_deeper, X_test, y_test, "Deeper Decoder MobileNet-UNet", gpu_available
    )
    
    # Compare Results
    compare_models(results_original, results_deeper)
    
    # Plot training histories
    print("\n" + "="*60)
    print("TRAINING CURVES")
    print("="*60)
    
    plot_training_history(history_original, "- Original Model")
    plot_training_history(history_deeper, "- Deeper Decoder Model")
    
    # Visualize sample predictions
    print("\n" + "="*60)
    print("SAMPLE PREDICTIONS")
    print("="*60)
    
    visualize_predictions(model_original, X_test, y_test, "Original Model")
    visualize_predictions(model_deeper, X_test, y_test, "Deeper Decoder Model")
    
    # Final Summary
    print("\n" + "="*100)
    print("DEEPER DECODER EXPERIMENT COMPLETED!")
    print("="*100)
    
    print(f"\nKey Results:")
    print(f"Original Model - Disc Dice: {results_original['class_results']['Disc']['dice']:.4f}, Cup Dice: {results_original['class_results']['Cup']['dice']:.4f}")
    print(f"Deeper Model   - Disc Dice: {results_deeper['class_results']['Disc']['dice']:.4f}, Cup Dice: {results_deeper['class_results']['Cup']['dice']:.4f}")
    
    disc_improvement = ((results_deeper['class_results']['Disc']['dice'] - results_original['class_results']['Disc']['dice']) / results_original['class_results']['Disc']['dice']) * 100
    cup_improvement = ((results_deeper['class_results']['Cup']['dice'] - results_original['class_results']['Cup']['dice']) / results_original['class_results']['Cup']['dice']) * 100
    
    print(f"\nDeeper Decoder Impact:")
    print(f"Disc Dice improvement: {disc_improvement:+.3f}%")
    print(f"Cup Dice improvement: {cup_improvement:+.3f}%")
    
    if disc_improvement > 0 or cup_improvement > 0:
        print("\n✅ Deeper Decoder shows improvement for Class 1/2!")
    else:
        print("\n❌ Deeper Decoder did not improve Class 1/2 performance")
    
    print(f"\nModel Complexity Comparison:")
    print(f"Original Parameters: {model_original.count_params():,}")
    print(f"Deeper Parameters: {model_deeper.count_params():,}")
    parameter_increase = ((model_deeper.count_params() - model_original.count_params()) / model_original.count_params()) * 100
    print(f"Parameter increase: {parameter_increase:+.1f}%")
    
    print("\n" + "="*100)