import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

# ---- MobileNet-UNet Model ----
def build_mobilenet_unet(input_shape=(256, 256, 3), num_classes=3):
    """Build MobileNet-UNet with minimum 4 encoder/decoder layers"""
    
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
    
    model = models.Model(inputs, outputs, name="MobileNet-UNet")
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
def train_model(X_train, y_train, X_val, y_val, model_name, augment=None, gpu_available=True):
    """Train MobileNet-UNet model"""
    
    print(f"\nTraining {model_name}...")
    print(f"Using augmentation: {'Yes' if augment else 'No'}")
    
    # Clear any previous models
    K.clear_session()
    
    # Build model
    model = build_mobilenet_unet()
    
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
    
    # Set batch size
    batch_size = 8
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
        print("Reducing batch size to 4...")
        batch_size = 4
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
    y_pred_prob = model.predict(X_test, verbose=1, batch_size=8)
    y_pred_labels = np.argmax(y_pred_prob, axis=-1)
    
    # Flatten for per-class metrics
    y_true_flat = y_test.flatten()
    y_pred_flat = y_pred_labels.flatten()
    
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
    
    print("\nDice and IoU per class:")
    for cidx, cname in enumerate(class_names):
        dice = dice_coef_class_explicit_np(y_test_cat, y_pred_prob, cidx)
        iou = iou_coef_class_explicit_np(y_test_cat, y_pred_prob, cidx)
        dice_per_class.append(dice)
        iou_per_class.append(iou)
        print(f"  {cname}: Dice={dice:.4f}, IoU={iou:.4f}")
    
    # Overall Dice and IoU (mean of 3 classes)
    overall_dice = np.mean(dice_per_class)
    overall_iou = np.mean(iou_per_class)
    
    print(f"Overall Dice: {overall_dice:.4f}")
    print(f"Overall IoU: {overall_iou:.4f}")
    
    # Model complexity
    print(f"\nTotal Parameters: {model.count_params():,}")
    
    # Manual FLOPS calculation
    manual_flops = calculate_flops_manual(model)
    if manual_flops:
        print(f"GFLOPS (manual estimate): {manual_flops:.2f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_true_flat, y_pred_flat)
    print("\nConfusion Matrix:")
    print(cm)
    
    return overall_precision, overall_recall, overall_f1, overall_dice, overall_iou

def plot_confusion_matrices(y_test, y_pred, model_name, save_dir='results'):
    """Plot confusion matrices"""
    os.makedirs(save_dir, exist_ok=True)
    
    y_test_cat = np.array([to_categorical(mask, num_classes=3) for mask in y_test])
    y_true_labels = np.argmax(y_test_cat, axis=-1).flatten()
    y_pred_labels = np.argmax(y_pred, axis=-1).flatten()
    
    cm = confusion_matrix(y_true_labels, y_pred_labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    class_names = ['Background', 'Disc', 'Cup']
    
    # Regular confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[0], cbar_kws={'label': 'Count'})
    axes[0].set_title(f'{model_name}\nConfusion Matrix', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('True Label', fontsize=12)
    axes[0].set_xlabel('Predicted Label', fontsize=12)
    
    # Normalized confusion matrix
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

def visualize_predictions(model, X_test, y_test, model_name, num_samples=6, save_dir='results'):
    """Visualize sample predictions with disc and cup dice scores"""
    
    print(f"\nGenerating sample predictions for {model_name}...")
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate predictions
    preds = model.predict(X_test[:num_samples], batch_size=4, verbose=0)
    pred_labels = np.argmax(preds, axis=-1)
    
    # Color map for classes
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
            pred_colored[pred_labels[i] == class_id] = color
        axes[i, 2].imshow(pred_colored)
        axes[i, 2].set_title('Segmented Mask', fontsize=12, fontweight='bold')
        axes[i, 2].axis('off')
        
        # Overlay
        overlay = X_test[i].copy()
        for class_id, color in colors.items():
            if class_id > 0:
                mask = (pred_labels[i] == class_id).astype(np.float32)
                mask_rgb = np.stack([mask * color[0]/255, mask * color[1]/255, mask * color[2]/255], axis=-1)
                overlay = overlay * 0.6 + mask_rgb * 0.4
        axes[i, 3].imshow(overlay)
        axes[i, 3].set_title('Overlay', fontsize=12, fontweight='bold')
        axes[i, 3].axis('off')
        
        # Calculate dice scores for this sample
        y_test_cat = to_categorical(y_test[i], num_classes=3)
        y_pred_cat = to_categorical(pred_labels[i], num_classes=3)
        
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

# ---- Main Execution ----
if __name__ == "__main__":
    # Configure GPU first
    gpu_available = configure_gpu()
    
    # Dataset path - UPDATE THIS TO YOUR PATH
    root_dir = "/kaggle/input/refuge/REFUGE/"  # Change this to your actual path
    
    # Setup augmentations
    train_aug = setup_augmentations()
    
    print("="*80)
    print("MOBILENET-UNET: NEW SPLIT WITH AUGMENTATION")
    print(f"GPU Available: {gpu_available}")
    print("="*80)
    
    # Load and prepare data
    all_images, all_masks = load_all_data(root_dir)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = create_new_split(all_images, all_masks)
    
    # Train the model
    print("\n" + "="*60)
    print("TRAINING: NEW SPLIT WITH AUGMENTATION")
    print("="*60)
    
    model, history = train_model(
        X_train, y_train, X_val, y_val,
        "mobilenet_unet_new_with_aug", augment=train_aug, gpu_available=gpu_available
    )
    
    # Evaluate the model
    print("\n" + "="*60)
    print("EVALUATION")
    print("="*60)
    
    evaluate_model(model, X_test, y_test, "MobileNetV2-UNet", gpu_available)
    
    # Get predictions for confusion matrix
    y_pred = model.predict(X_test, verbose=0, batch_size=8)
    
    # Plot training history
    print("\n" + "="*60)
    print("TRAINING VISUALIZATIONS")
    print("="*60)
    
    plot_training_curves(history, "MobileNetV2-UNet", save_dir='results')
    
    # Confusion matrices
    print("\n" + "="*60)
    print("CONFUSION MATRICES")
    print("="*60)
    
    plot_confusion_matrices(y_test, y_pred, "MobileNetV2-UNet", save_dir='results')
    
    # Visualize sample predictions
    print("\n" + "="*60)
    print("SAMPLE PREDICTIONS")
    print("="*60)
    
    visualize_predictions(model, X_test, y_test, "MobileNetV2-UNet", num_samples=6, save_dir='results')
    
    # Final Results Summary
    print("\n" + "="*100)
    print("FINAL RESULTS SUMMARY")
    print("="*100)
    
    print(f"\nModel: MobileNetV2-UNet with Augmentation")
    print(f"Parameters: {model.count_params():,}")
    print(f"\nAll visualizations and metrics saved in: results/")
    print(f"  - Training curves: MobileNetV2-UNet_training_curves.png")
    print(f"  - Training history: MobileNetV2-UNet_history.png")
    print(f"  - Confusion matrices: MobileNetV2-UNet_confusion_matrices.png")
    print(f"  - Sample predictions: MobileNetV2-UNet_predictions.png")
    
    print("\n" + "="*100)
    print("EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("="*100)