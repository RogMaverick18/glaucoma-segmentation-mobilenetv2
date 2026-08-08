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

# ---- CBAM Attention Modules ----
def channel_attention(input_feature, ratio=8, name=""):
    """Channel Attention Module (CAM)"""
    channel = input_feature.shape[-1]
    
    # Shared MLP layers
    shared_layer_one = layers.Dense(channel//ratio, activation='relu', name=f'channel_attention_dense1_{name}')
    shared_layer_two = layers.Dense(channel, activation='sigmoid', name=f'channel_attention_dense2_{name}')
    
    # Global Average Pooling
    avg_pool = layers.GlobalAveragePooling2D()(input_feature)
    avg_pool = layers.Reshape((1, 1, channel))(avg_pool)
    avg_pool = shared_layer_one(avg_pool)
    avg_pool = shared_layer_two(avg_pool)
    
    # Global Max Pooling
    max_pool = layers.GlobalMaxPooling2D()(input_feature)
    max_pool = layers.Reshape((1, 1, channel))(max_pool)
    max_pool = shared_layer_one(max_pool)
    max_pool = shared_layer_two(max_pool)
    
    # Combine and apply
    cbam_feature = layers.Add()([avg_pool, max_pool])
    cbam_feature = layers.Multiply()([input_feature, cbam_feature])
    
    return cbam_feature

def spatial_attention(input_feature, name=""):
    """Spatial Attention Module (SAM)"""
    kernel_size = 7
    
    # Channel-wise pooling
    avg_pool = layers.Lambda(lambda x: tf.reduce_mean(x, axis=3, keepdims=True))(input_feature)
    max_pool = layers.Lambda(lambda x: tf.reduce_max(x, axis=3, keepdims=True))(input_feature)
    concat = layers.Concatenate(axis=3)([avg_pool, max_pool])
    
    # Spatial attention map
    cbam_feature = layers.Conv2D(filters=1, kernel_size=kernel_size, strides=1, 
                                padding='same', activation='sigmoid', 
                                name=f'spatial_attention_conv_{name}')(concat)
    cbam_feature = layers.Multiply()([input_feature, cbam_feature])
    
    return cbam_feature

def cbam_block(cbam_feature, ratio=8, name=""):
    """CBAM: Convolutional Block Attention Module (Channel + Spatial)"""
    # Apply channel attention first
    cbam_feature = channel_attention(cbam_feature, ratio, name=f"{name}_channel")
    # Then apply spatial attention
    cbam_feature = spatial_attention(cbam_feature, name=f"{name}_spatial")
    return cbam_feature

# ---- Enhanced MobileNet-UNet with CBAM ----
def build_mobilenet_unet_cbam(input_shape=(256, 256, 3), num_classes=3):
    """Build MobileNet-UNet with CBAM attention modules"""
    
    print("Building MobileNet-UNet with CBAM attention...")
    
    # Input
    inputs = layers.Input(input_shape, dtype='float32')
    
    # Encoder (MobileNetV2 backbone)
    backbone = MobileNetV2(input_tensor=inputs, weights='imagenet', include_top=False)
    
    # Extract skip connections from encoder layers
    skip_1 = backbone.get_layer('block_1_expand_relu').output    # 128x128
    skip_2 = backbone.get_layer('block_3_expand_relu').output    # 64x64
    skip_3 = backbone.get_layer('block_6_expand_relu').output    # 32x32
    skip_4 = backbone.get_layer('block_13_expand_relu').output   # 16x16
    
    # Apply CBAM attention to skip connections (KEY ENHANCEMENT)
    print("Applying CBAM to skip connections...")
    skip_1_att = cbam_block(skip_1, ratio=8, name="skip1")
    skip_2_att = cbam_block(skip_2, ratio=8, name="skip2")
    skip_3_att = cbam_block(skip_3, ratio=8, name="skip3")
    skip_4_att = cbam_block(skip_4, ratio=8, name="skip4")
    
    # Bridge (bottom of U) with attention
    bridge = backbone.output  # 8x8
    bridge_att = cbam_block(bridge, ratio=8, name="bridge")
    
    print("Building decoder with CBAM attention...")
    
    # Decoder Layer 1: 8x8 -> 16x16
    x = layers.UpSampling2D((2, 2))(bridge_att)
    x = layers.Concatenate()([x, skip_4_att])
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    # Apply CBAM after conv layers
    x = cbam_block(x, ratio=8, name="dec1")
    
    # Decoder Layer 2: 16x16 -> 32x32
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Concatenate()([x, skip_3_att])
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    # Apply CBAM after conv layers
    x = cbam_block(x, ratio=8, name="dec2")
    
    # Decoder Layer 3: 32x32 -> 64x64 (CRITICAL for disc/cup segmentation)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Concatenate()([x, skip_2_att])
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    # Apply CBAM - very important for Class 1 & 2
    x = cbam_block(x, ratio=8, name="dec3")
    
    # Decoder Layer 4: 64x64 -> 128x128
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Concatenate()([x, skip_1_att])
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    # Apply CBAM
    x = cbam_block(x, ratio=8, name="dec4")
    
    # Final upsampling to original size: 128x128 -> 256x256
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    # Final CBAM for output refinement
    x = cbam_block(x, ratio=4, name="final")
    
    # Output layer - ensure float32
    outputs = layers.Conv2D(num_classes, (1, 1), activation='softmax', name='output', dtype='float32')(x)
    
    model = models.Model(inputs, outputs, name="MobileNet-UNet-CBAM")
    
    print(f"Model built successfully with CBAM attention!")
    print(f"Total parameters: {model.count_params():,}")
    
    return model

# ---- Original MobileNet-UNet Model (for comparison) ----
def build_mobilenet_unet_original(input_shape=(256, 256, 3), num_classes=3):
    """Build original MobileNet-UNet without attention"""
    
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
def train_model(X_train, y_train, X_val, y_val, model_name, use_cbam=False, augment=None, gpu_available=True):
    """Train MobileNet-UNet model with or without CBAM"""
    
    print(f"\nTraining {model_name}...")
    print(f"Using CBAM attention: {'Yes' if use_cbam else 'No'}")
    print(f"Using augmentation: {'Yes' if augment else 'No'}")
    
    # Clear any previous models
    K.clear_session()
    
    # Build model
    if use_cbam:
        model = build_mobilenet_unet_cbam()
    else:
        model = build_mobilenet_unet_original()
    
    # Optimizer
    initial_lr = 1e-3
    optimizer = optimizers.Adam(learning_rate=initial_lr)
    
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=[
            dice_coef_multiclass, dice_class_0, dice_class_1, dice_class_2, "accuracy",
            iou_coef_multiclass, iou_class_0, iou_class_1, iou_class_2
        ]
    )
    
    # Callbacks
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
        patience=3,
        verbose=1,
        mode='min',
        min_lr=1e-7
    )
    
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=8,
        verbose=1, 
        restore_best_weights=True,
        mode='min'
    )
    
    # Set batch size (reduce if using CBAM due to increased memory usage)
    batch_size = 6 if use_cbam else 8
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
            epochs=50,
            callbacks=[checkpoint, reduce_lr, early_stop],
            verbose=1
        )
    except Exception as e:
        print(f"Training error: {e}")
        print("Reducing batch size...")
        batch_size = 4 if use_cbam else 6
        steps_train = X_train.shape[0] // batch_size
        steps_val = X_val.shape[0] // batch_size
        
        history = model.fit(
            data_generator(X_train, y_train, batch_size, augment=augment),
            validation_data=data_generator(X_val, y_val, batch_size, augment=None),
            steps_per_epoch=steps_train,
            validation_steps=steps_val,
            epochs=50,
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
    
    # Get predictions for calculations
    y_pred_prob = model.predict(X_test, verbose=1, batch_size=4)
    y_pred_labels = np.argmax(y_pred_prob, axis=-1)
    
    # Flatten for per-class metrics
    y_true_flat = y_test.flatten()
    y_pred_flat = y_pred_labels.flatten()
    
    # Model complexity
    print(f"\nTotal Parameters: {model.count_params():,}")
    
    # Calculate per-class precision, recall, F1
    class_names = ["Background", "Disc", "Cup"]
    precision_per_class = precision_score(y_true_flat, y_pred_flat, average=None, zero_division=0, labels=[0, 1, 2])
    recall_per_class = recall_score(y_true_flat, y_pred_flat, average=None, zero_division=0, labels=[0, 1, 2])
    f1_per_class = f1_score(y_true_flat, y_pred_flat, average=None, zero_division=0, labels=[0, 1, 2])
    
    # Overall metrics (mean of 3 classes)
    overall_precision = np.mean(precision_per_class)
    overall_recall = np.mean(recall_per_class)
    overall_f1 = np.mean(f1_per_class)
    
    # Print in required format
    print("\nPrecision, Recall, F1 Score per class:")
    for i, cname in enumerate(class_names):
        print(f"  {cname}: Precision={precision_per_class[i]:.4f}, Recall={recall_per_class[i]:.4f}, F1={f1_per_class[i]:.4f}")
    
    print(f"Overall Precision: {overall_precision:.4f}")
    print(f"Overall Recall: {overall_recall:.4f}")
    print(f"Overall F1 Score: {overall_f1:.4f}")
    
    # Calculate per-class Dice and IoU
    dice_per_class = []
    iou_per_class = []
    class_results = {}
    
    print("\nDice and IoU per class:")
    for cidx, cname in enumerate(class_names):
        dice = dice_coef_class_explicit_np(y_test_cat, y_pred_prob, cidx)
        iou = iou_coef_class_explicit_np(y_test_cat, y_pred_prob, cidx)
        dice_per_class.append(dice)
        iou_per_class.append(iou)
        class_results[cname] = {'dice': dice, 'iou': iou}
        print(f"  {cname}: Dice={dice:.4f}, IoU={iou:.4f}")
    
    # Overall Dice and IoU (mean of 3 classes)
    overall_dice = np.mean(dice_per_class)
    overall_iou = np.mean(iou_per_class)
    
    print(f"Overall Dice: {overall_dice:.4f}")
    print(f"Overall IoU: {overall_iou:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_true_flat, y_pred_flat)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Manual FLOPS calculation
    manual_flops = calculate_flops_manual(model)
    if manual_flops:
        print(f"\nGFLOPS (manual estimate): {manual_flops:.2f}")
    
    return {
        'overall_dice': overall_dice,
        'overall_iou': overall_iou,
        'class_results': class_results,
        'precision': overall_precision,
        'recall': overall_recall,
        'fscore': overall_f1
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

def compare_models(results_original, results_cbam):
    """Compare original vs CBAM results"""
    
    print("\n" + "="*80)
    print("COMPARISON: ORIGINAL vs CBAM ATTENTION")
    print("="*80)
    
    print(f"{'Metric':<20} {'Original':<12} {'CBAM':<12} {'Improvement':<15}")
    print("-" * 60)
    
    # Overall metrics
    orig_dice = results_original['overall_dice']
    cbam_dice = results_cbam['overall_dice']
    dice_improvement = ((cbam_dice - orig_dice) / orig_dice) * 100
    
    orig_iou = results_original['overall_iou']
    cbam_iou = results_cbam['overall_iou']
    iou_improvement = ((cbam_iou - orig_iou) / orig_iou) * 100
    
    print(f"{'Overall Dice':<20} {orig_dice:<12.4f} {cbam_dice:<12.4f} {dice_improvement:+.3f}%")
    print(f"{'Overall IoU':<20} {orig_iou:<12.4f} {cbam_iou:<12.4f} {iou_improvement:+.3f}%")
    
    print("\n--- PER-CLASS IMPROVEMENTS (KEY METRICS) ---")
    for class_name in ["Disc", "Cup"]:
        if class_name in results_original['class_results']:
            orig_dice_class = results_original['class_results'][class_name]['dice']
            cbam_dice_class = results_cbam['class_results'][class_name]['dice']
            dice_class_improvement = ((cbam_dice_class - orig_dice_class) / orig_dice_class) * 100
            
            orig_iou_class = results_original['class_results'][class_name]['iou']
            cbam_iou_class = results_cbam['class_results'][class_name]['iou']
            iou_class_improvement = ((cbam_iou_class - orig_iou_class) / orig_iou_class) * 100
            
            print(f"\n{class_name} Class:")
            print(f"{'  Dice':<18} {orig_dice_class:<12.4f} {cbam_dice_class:<12.4f} {dice_class_improvement:+.3f}%")
            print(f"{'  IoU':<18} {orig_iou_class:<12.4f} {cbam_iou_class:<12.4f} {iou_class_improvement:+.3f}%")

# ---- Main Execution ----
if __name__ == "__main__":
    # Configure GPU first
    gpu_available = configure_gpu()
    
    # Dataset path - UPDATE THIS TO YOUR PATH
    root_dir = "/kaggle/input/refuge/REFUGE/"  # Change this to your actual path
    
    # Setup augmentations
    train_aug = setup_augmentations()
    
    print("="*80)
    print("MOBILENET-UNET: CBAM ATTENTION EXPERIMENT")
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
        "mobilenet_unet_original", use_cbam=False, augment=train_aug, gpu_available=gpu_available
    )
    
    # Train CBAM Model
    print("\n" + "="*60)
    print("TRAINING: CBAM ATTENTION MODEL")
    print("="*60)
    
    model_cbam, history_cbam = train_model(
        X_train, y_train, X_val, y_val,
        "mobilenet_unet_cbam", use_cbam=True, augment=train_aug, gpu_available=gpu_available
    )
    
    # Evaluate Original Model
    print("\n" + "="*60)
    print("EVALUATION: ORIGINAL MODEL")
    print("="*60)
    
    results_original = evaluate_model(
        model_original, X_test, y_test, "Original MobileNet-UNet", gpu_available
    )
    
    # Evaluate CBAM Model
    print("\n" + "="*60)
    print("EVALUATION: CBAM MODEL")
    print("="*60)
    
    results_cbam = evaluate_model(
        model_cbam, X_test, y_test, "CBAM MobileNet-UNet", gpu_available
    )
    
    # Compare Results
    compare_models(results_original, results_cbam)
    
    # Plot training histories
    print("\n" + "="*60)
    print("TRAINING CURVES")
    print("="*60)
    
    plot_training_history(history_original, "- Original Model")
    plot_training_history(history_cbam, "- CBAM Model")
    
    # Visualize sample predictions
    print("\n" + "="*60)
    print("SAMPLE PREDICTIONS")
    print("="*60)
    
    visualize_predictions(model_original, X_test, y_test, "Original Model")
    visualize_predictions(model_cbam, X_test, y_test, "CBAM Model")
    
    # Final Summary
    print("\n" + "="*100)
    print("CBAM ATTENTION EXPERIMENT COMPLETED!")
    print("="*100)
    
    print(f"\nKey Results:")
    print(f"Original Model - Disc Dice: {results_original['class_results']['Disc']['dice']:.4f}, Cup Dice: {results_original['class_results']['Cup']['dice']:.4f}")
    print(f"CBAM Model    - Disc Dice: {results_cbam['class_results']['Disc']['dice']:.4f}, Cup Dice: {results_cbam['class_results']['Cup']['dice']:.4f}")
    
    disc_improvement = ((results_cbam['class_results']['Disc']['dice'] - results_original['class_results']['Disc']['dice']) / results_original['class_results']['Disc']['dice']) * 100
    cup_improvement = ((results_cbam['class_results']['Cup']['dice'] - results_original['class_results']['Cup']['dice']) / results_original['class_results']['Cup']['dice']) * 100
    
    print(f"\nCBAM Impact:")
    print(f"Disc Dice improvement: {disc_improvement:+.3f}%")
    print(f"Cup Dice improvement: {cup_improvement:+.3f}%")
    
    if disc_improvement > 0 or cup_improvement > 0:
        print("\n✅ CBAM Attention shows improvement for Class 1/2!")
    else:
        print("\n❌ CBAM Attention did not improve Class 1/2 performance")
    
    print("\n" + "="*100)