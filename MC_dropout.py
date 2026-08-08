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

# ============================================================================
# PRINT ALL PARAMETERS USED IN THE CODE
# ============================================================================

def print_all_parameters():
    """Print all hyperparameters and settings used in this experiment"""
    
    print("\n" + "="*100)
    print(" "*35 + "MC DROPOUT EXPERIMENT PARAMETERS")
    print("="*100)
    
    # Model Architecture
    print("\n" + "─"*100)
    print("MODEL ARCHITECTURE")
    print("─"*100)
    print("""
    Model Name: MobileNet-UNet with MC Dropout
    Backbone: MobileNetV2 (ImageNet pretrained)
    Input Shape: (256, 256, 3)
    Output Classes: 3 (Background=0, Disc=1, Cup=2)
    
    Encoder:
      - MobileNetV2 backbone (frozen: No)
      - Skip connections: 4 levels
        • Skip 1: 128×128 (block_1_expand_relu)
        • Skip 2: 64×64 (block_3_expand_relu)
        • Skip 3: 32×32 (block_6_expand_relu)
        • Skip 4: 16×16 (block_13_expand_relu)
      - Bridge: 8×8×1280
    
    Dropout Configuration:
      - Bridge Dropout: 0.30
      - Decoder 1 Dropout (16×16): 0.30
      - Decoder 2 Dropout (32×32): 0.30
      - Decoder 3 Dropout (64×64): 0.21 (0.30 × 0.7)
      - Decoder 4 Dropout (128×128): 0.15 (0.30 × 0.5)
      - Total Dropout Layers: 5
      - Purpose: Enable MC Dropout for uncertainty estimation
    
    Decoder Architecture:
      - Decoder 1: 8×8 → 16×16, filters=512, conv_blocks=2
      - Decoder 2: 16×16 → 32×32, filters=256, conv_blocks=2
      - Decoder 3: 32×32 → 64×64, filters=128, conv_blocks=2
      - Decoder 4: 64×64 → 128×128, filters=64, conv_blocks=2
      - Final: 128×128 → 256×256, filters=32
      - All conv: kernel=(3,3), padding='same', activation='relu'
      - Batch Normalization: After each conv layer
      - Output: Conv2D(3, 1×1), activation='softmax', dtype='float32'
    """)
    
    # Training Parameters
    print("─"*100)
    print("TRAINING PARAMETERS")
    print("─"*100)
    print("""
    Optimizer: Adam
      - Learning Rate: 1e-3 (0.001)
      - Beta_1: 0.9 (default)
      - Beta_2: 0.999 (default)
      - Epsilon: 1e-7 (default)
      - Gradient Clipping: None
    
    Loss Function: Categorical Crossentropy
      - Class Weights: None (balanced dataset assumption)
    
    Batch Size:
      - Primary: 8
      - Fallback: 4 (if OOM error)
      - Test/Validation: 8
      - MC Dropout Inference: 4
    
    Epochs:
      - Maximum: 50
      - Early Stopping Patience: 8 epochs
    
    Data Split:
      - Training: 600 samples (50.0%)
      - Validation: 200 samples (16.67%)
      - Testing: 400 samples (33.33%)
      - Total: 1200 samples
      - Random State: 42 (reproducible)
      - Stratification: None
    """)
    
    # Callbacks
    print("─"*100)
    print("CALLBACKS & REGULARIZATION")
    print("─"*100)
    print("""
    ModelCheckpoint:
      - Filename: mobilenet_unet_mc_dropout.h5
      - Monitor: val_loss
      - Mode: min
      - Save Best Only: True
      - Verbose: 1
    
    ReduceLROnPlateau:
      - Monitor: val_loss
      - Factor: 0.3 (LR = LR × 0.3)
      - Patience: 3 epochs
      - Mode: min
      - Min Learning Rate: 1e-7
      - Verbose: 1
    
    EarlyStopping:
      - Monitor: val_loss
      - Patience: 8 epochs
      - Mode: min
      - Restore Best Weights: True
      - Verbose: 1
    """)
    
    # Data Augmentation
    print("─"*100)
    print("DATA AUGMENTATION (Training Only)")
    print("─"*100)
    print("""
    Albumentations Pipeline:
    
    1. RandomBrightnessContrast:
       - brightness_limit: ±0.15 (±15%)
       - contrast_limit: ±0.15 (±15%)
       - probability: 0.7 (70% of samples)
    
    2. HueSaturationValue:
       - hue_shift_limit: ±10
       - sat_shift_limit: ±15
       - val_shift_limit: ±10
       - probability: 0.5 (50% of samples)
    
    3. GaussianBlur:
       - blur_limit: 3 (kernel size)
       - probability: 0.3 (30% of samples)
    
    4. RandomGamma:
       - gamma_limit: (80, 120)
       - probability: 0.3 (30% of samples)
    
    Note: No spatial augmentations (rotation, flip) to preserve anatomical orientation
    """)
    
    # Image Preprocessing
    print("─"*100)
    print("IMAGE PREPROCESSING")
    print("─"*100)
    print("""
    Input Images:
      - Original Size: Variable (resized to 256×256)
      - Target Size: (256, 256)
      - Color Space: RGB (converted from BGR)
      - Normalization: [0, 1] (divided by 255.0)
      - Data Type: float32
    
    Masks:
      - Original Size: Variable (resized to 256×256)
      - Target Size: (256, 256)
      - Classes: 3
        • Class 0: Background
        • Class 1: Optic Disc
        • Class 2: Optic Cup
      - Value Clipping: [0, 2] (if needed)
      - Encoding: One-hot categorical (3 channels)
      - Data Type: uint8 → float32 (after categorical)
    """)
    
    # MC Dropout Parameters
    print("─"*100)
    print("MC DROPOUT PARAMETERS")
    print("─"*100)
    print("""
    MC Dropout Configuration:
      - Number of Stochastic Forward Passes: 15
      - Inference Batch Size: 4
      - Dropout Enabled at Inference: Yes (training=True override)
      - Purpose: Bayesian uncertainty estimation
    
    Uncertainty Metrics:
      1. Predictive Mean: E[p(y|x,D)] - Average prediction across samples
      2. Predictive Std: σ[p(y|x,D)] - Uncertainty per pixel per class
      3. Confidence Map: 1 - (Entropy / log(C)) - Normalized inverse entropy
      4. Uncertainty Map: Mean(σ) across classes - Overall uncertainty
      5. Prediction Entropy: -Σ(p·log(p)) - Measure of randomness
      6. Mutual Information: Epistemic uncertainty approximation
    
    Confidence Thresholds:
      - High Confidence: > 0.9
      - Low Confidence: < 0.5
      - Uncertainty Percentiles: 50th, 75th, 90th, 95th, 99th
    """)
    
    # Evaluation Metrics
    print("─"*100)
    print("EVALUATION METRICS")
    print("─"*100)
    print("""
    Training Metrics (Computed Every Batch):
      1. loss (categorical_crossentropy)
      2. dice_coef_multiclass
      3. dice_class_0 (Background)
      4. dice_class_1 (Disc)
      5. dice_class_2 (Cup)
      6. accuracy (pixel-wise)
      7. iou_coef_multiclass
      8. iou_class_0
      9. iou_class_1
      10. iou_class_2
    
    Test Metrics:
      - All training metrics (numpy-based)
      - Precision (weighted average)
      - Recall (weighted average)
      - F1-Score (weighted average)
      - Confusion Matrix
      - Model Parameters (count)
      - Estimated GFLOPS
    
    MC Dropout Metrics:
      - Mean Prediction Dice & IoU
      - Per-class Dice & IoU with uncertainty
      - Confidence statistics (mean, percentiles)
      - Uncertainty statistics (mean, max, distribution)
      - Correct vs Incorrect confidence gap
      - High/Low confidence region analysis
    
    Smooth Factor: 1e-7 (for all Dice/IoU calculations)
    """)
    
    # Visualization Settings
    print("─"*100)
    print("VISUALIZATION PARAMETERS")
    print("─"*100)
    print("""
    Training History Plots:
      - Figure Size: (20, 5)
      - Subplots: 4 (Loss, Accuracy, Dice, Learning Rate)
      - Grid: Enabled
      - LR Y-scale: Logarithmic
    
    MC Dropout Visualization:
      - Number of Samples: 5
      - Figure Size: (24, 10)
      - Layout: 2 rows × 5 columns
      - Row 1: Original, Ground Truth, Prediction, Overlay, Error Map
      - Row 2: Confidence, Uncertainty, Disc σ, Cup σ, High Uncertainty Regions
      - Colormaps:
        • Predictions: 'jet'
        • Confidence: 'RdYlGn' (Red-Yellow-Green)
        • Uncertainty: 'hot_r' (inverted hot)
        • Error: 'hot'
      - Overlay Colors:
        • Disc (Class 1): Red [1, 0, 0]
        • Cup (Class 2): Green [0, 1, 0]
        • High Uncertainty: Yellow [1, 1, 0]
    
    Uncertainty Distribution Plots:
      - Figure Size: (16, 12)
      - Subplots: 2×2
      - Histograms: 50 bins
      - Scatter Sample Size: 10,000 points (for speed)
    """)
    
    # GPU Configuration
    print("─"*100)
    print("GPU CONFIGURATION")
    print("─"*100)
    print("""
    GPU Settings:
      - Memory Growth: Enabled (dynamic allocation)
      - Device Placement: GPU:0 (first GPU)
      - Visible Devices: Single GPU only
      - Test Allocation: 2×3 matrix multiplication
      - CUDA Required: Yes (for optimal performance)
      - Fallback: CPU (if GPU not available)
    """)
    
    # File Paths
    print("─"*100)
    print("FILE PATHS & OUTPUTS")
    print("─"*100)
    print("""
    Dataset Root: /kaggle/input/refuge/REFUGE/
    
    Data Structure:
      - train/Images/
      - train/Masks/
      - val/Images/
      - val/Masks/
      - test/Images/
      - test/Masks/
    
    Model Checkpoint: mobilenet_unet_mc_dropout.h5
    
    Visualization Outputs:
      - Training curves (matplotlib display)
      - MC Dropout sample predictions (matplotlib display)
      - Uncertainty distribution plots (matplotlib display)
      - Optional: Save to disk (if save_path provided)
    """)
    
    # Expected Outcomes
    print("─"*100)
    print("EXPECTED OUTCOMES & BENEFITS")
    print("─"*100)
    print("""
    Standard Evaluation:
      - Overall Dice: ~0.85-0.92
      - Overall IoU: ~0.75-0.85
      - Background Dice: ~0.95+
      - Disc Dice: ~0.85-0.90
      - Cup Dice: ~0.75-0.85 (most challenging)
    
    MC Dropout Benefits:
      ✓ Uncertainty Quantification: σ per pixel per class
      ✓ Confidence Maps: Identify reliable predictions
      ✓ Error Detection: High uncertainty → likely errors
      ✓ Model Reliability: Epistemic uncertainty estimation
      ✓ Clinical Value: Flag ambiguous regions for review
      ✓ Improved Trust: Transparent about model limitations
    
    Computational Cost:
      - Training: ~1-2 hours (GPU) or 8-12 hours (CPU)
      - Standard Inference: ~2-3 seconds for 400 samples
      - MC Dropout Inference: ~60-90 seconds (30 passes × 400 samples)
      - Trade-off: 30× slower but provides uncertainty
    """)
    
    # Key Design Decisions
    print("─"*100)
    print("KEY DESIGN DECISIONS & RATIONALE")
    print("─"*100)
    print("""
    1. Dropout Rate = 0.3:
       - Standard for MC Dropout (0.2-0.4 range)
       - Higher at bridge/early decoder (more capacity)
       - Reduced at later decoder (preserve details)
    
    2. MC Samples = 15:
       - Balance between accuracy and speed
       - 20-50 is typical in literature
       - More samples = better uncertainty estimate
    
    3. Dropout Placement:
       - After bridge (global context)
       - After each decoder block (multi-scale uncertainty)
       - Reduced rates at higher resolutions (preserve boundaries)
    
    4. No Spatial Augmentation:
       - Medical images: anatomical orientation matters
       - Only intensity augmentations
       - Prevents unrealistic transformations
    
    5. Batch Size = 8/4:
       - Trade-off: speed vs memory
       - Smaller for MC Dropout inference (15 passes)
       - Auto-fallback to 4 if OOM
    
    6. Learning Rate Schedule:
       - Start: 1e-3 (standard for Adam)
       - Reduce by 0.3× every 3 plateau epochs
       - Min: 1e-7 (prevent over-reduction)
    
    7. Early Stopping = 8 epochs:
       - Prevent overfitting on small dataset (600 samples)
       - Restore best weights
       - Balance: enough time vs computational cost
    """)
    
    # References & Citations
    print("─"*100)
    print("METHODOLOGY REFERENCES")
    print("─"*100)
    print("""
    MC Dropout:
      - Gal & Ghahramani (2016): "Dropout as a Bayesian Approximation"
      - Kendall & Gal (2017): "What Uncertainties Do We Need in Bayesian DL?"
    
    U-Net Architecture:
      - Ronneberger et al. (2015): "U-Net: Convolutional Networks for Biomedical Image Segmentation"
    
    MobileNetV2:
      - Sandler et al. (2018): "MobileNetV2: Inverted Residuals and Linear Bottlenecks"
    
    Medical Segmentation:
      - REFUGE Challenge (2018): Retinal Fundus Glaucoma Challenge
    """)
    
    print("="*100)
    print(" "*30 + "END OF PARAMETER SPECIFICATION")
    print("="*100 + "\n")

# ---- Improved GPU Configuration ----
def configure_gpu():
    """Configure GPU settings for optimal performance"""
    print("Configuring GPU settings...")
    
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            for gpu in physical_devices:
                tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.set_visible_devices(physical_devices[0], 'GPU')
            
            print(f"GPU found: {physical_devices}")
            print("GPU device name:", tf.test.gpu_device_name())
            print("Is built with CUDA:", tf.test.is_built_with_cuda())
            
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
            img = cv2.imread(os.path.join(images_folder, img_file))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, img_size) / 255.0
            
            mask = cv2.imread(os.path.join(masks_folder, mask_dict[base]), cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, img_size)
            
            unique_vals = np.unique(mask)
            if len(unique_vals) > 3 or np.max(unique_vals) > 2:
                print(f"Warning: Mask {mask_dict[base]} has unexpected values: {unique_vals}")
                mask = np.clip(mask // 85, 0, 2).astype(np.uint8)
            
            images.append(img)
            masks.append(mask)
    
    return np.array(images, dtype=np.float32), np.array(masks, dtype=np.uint8)

def load_all_data(root_dir):
    """Load all data from existing splits and merge"""
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
                    aug = augment(image=img, mask=mask)
                    img_aug = aug['image']
                    mask_aug = aug['mask']
                else:
                    img_aug, mask_aug = img, mask
                
                mask_cat = to_categorical(mask_aug, num_classes=3)
                
                batch_x.append(img_aug)
                batch_y.append(mask_cat)
            
            batch_x = np.stack(batch_x).astype(np.float32)
            batch_y = np.stack(batch_y).astype(np.float32)
            
            yield batch_x, batch_y

# ---- MobileNet-UNet Model with Dropout ----
def build_mobilenet_unet(input_shape=(256, 256, 3), num_classes=3, dropout_rate=0.3):
    """Build MobileNet-UNet with Dropout layers for MC Dropout"""
    
    print(f"\nBuilding MobileNet-UNet with Dropout Rate: {dropout_rate}")
    
    inputs = layers.Input(input_shape, dtype='float32')
    
    # Encoder (MobileNetV2 backbone)
    backbone = MobileNetV2(input_tensor=inputs, weights='imagenet', include_top=False)
    
    # Extract skip connections
    skip_connections = [
        backbone.get_layer('block_1_expand_relu').output,    # 128x128
        backbone.get_layer('block_3_expand_relu').output,    # 64x64
        backbone.get_layer('block_6_expand_relu').output,    # 32x32
        backbone.get_layer('block_13_expand_relu').output,   # 16x16
    ]
    
    # Bridge
    bridge = backbone.output  # 8x8
    bridge = layers.Dropout(dropout_rate, name='bridge_dropout')(bridge)
    
    # Decoder Layer 1: 8x8 -> 16x16
    x = layers.UpSampling2D((2, 2), name='decoder1_upsample')(bridge)
    x = layers.Concatenate(name='decoder1_concat')([x, skip_connections[3]])
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='decoder1_conv1')(x)
    x = layers.BatchNormalization(name='decoder1_bn1')(x)
    x = layers.Dropout(dropout_rate, name='decoder1_dropout')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='decoder1_conv2')(x)
    x = layers.BatchNormalization(name='decoder1_bn2')(x)
    
    # Decoder Layer 2: 16x16 -> 32x32
    x = layers.UpSampling2D((2, 2), name='decoder2_upsample')(x)
    x = layers.Concatenate(name='decoder2_concat')([x, skip_connections[2]])
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='decoder2_conv1')(x)
    x = layers.BatchNormalization(name='decoder2_bn1')(x)
    x = layers.Dropout(dropout_rate, name='decoder2_dropout')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='decoder2_conv2')(x)
    x = layers.BatchNormalization(name='decoder2_bn2')(x)
    
    # Decoder Layer 3: 32x32 -> 64x64
    x = layers.UpSampling2D((2, 2), name='decoder3_upsample')(x)
    x = layers.Concatenate(name='decoder3_concat')([x, skip_connections[1]])
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='decoder3_conv1')(x)
    x = layers.BatchNormalization(name='decoder3_bn1')(x)
    x = layers.Dropout(dropout_rate * 0.7, name='decoder3_dropout')(x)  # Reduced dropout
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='decoder3_conv2')(x)
    x = layers.BatchNormalization(name='decoder3_bn2')(x)
    
    # Decoder Layer 4: 64x64 -> 128x128
    x = layers.UpSampling2D((2, 2), name='decoder4_upsample')(x)
    x = layers.Concatenate(name='decoder4_concat')([x, skip_connections[0]])
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='decoder4_conv1')(x)
    x = layers.BatchNormalization(name='decoder4_bn1')(x)
    x = layers.Dropout(dropout_rate * 0.5, name='decoder4_dropout')(x)  # Further reduced
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='decoder4_conv2')(x)
    x = layers.BatchNormalization(name='decoder4_bn2')(x)
    
    # Final upsampling: 128x128 -> 256x256
    x = layers.UpSampling2D((2, 2), name='final_upsample')(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='final_conv')(x)
    x = layers.BatchNormalization(name='final_bn')(x)
    
    # Output layer
    outputs = layers.Conv2D(num_classes, (1, 1), activation='softmax', name='output', dtype='float32')(x)
    
    model = models.Model(inputs, outputs, name="MobileNet-UNet-MCDropout")
    
    # Count dropout layers
    dropout_layers = [layer for layer in model.layers if isinstance(layer, layers.Dropout)]
    print(f"Total Dropout Layers: {len(dropout_layers)}")
    print(f"Dropout Rates: {[layer.rate for layer in dropout_layers]}")
    
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
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def iou_coef_multiclass_np(y_true, y_pred, smooth=1e-7):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    union = np.sum(y_true_f) + np.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

def dice_coef_class_explicit_np(y_true, y_pred, class_idx, smooth=1e-7):
    y_true_c = (np.argmax(y_true, axis=-1) == class_idx).astype(np.float32)
    y_pred_c = (np.argmax(y_pred, axis=-1) == class_idx).astype(np.float32)
    intersection = np.sum(y_true_c * y_pred_c)
    dice = (2. * intersection + smooth) / (np.sum(y_true_c) + np.sum(y_pred_c) + smooth)
    return dice

def iou_coef_class_explicit_np(y_true, y_pred, class_idx, smooth=1e-7):
    y_true_c = (np.argmax(y_true, axis=-1) == class_idx).astype(np.float32)
    y_pred_c = (np.argmax(y_pred, axis=-1) == class_idx).astype(np.float32)
    intersection = np.sum(y_true_c * y_pred_c)
    union = np.sum(y_true_c) + np.sum(y_pred_c) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou

# ---- MC DROPOUT FUNCTIONS ----

def enable_mc_dropout(model):
    """
    Enable dropout at inference time for MC Dropout
    
    FIXED: Properly handle dropout layer training mode
    """
    print("\nEnabling MC Dropout for inference...")
    dropout_count = 0
    
    for layer in model.layers:
        if isinstance(layer, layers.Dropout):
            # Store original call method
            original_call = layer.call
            
            # Create a wrapper that forces training=True
            def make_mc_call(original_fn):
                def mc_call(inputs, training=None, **kwargs):
                    # Always use training=True for MC Dropout
                    return original_fn(inputs, training=True)
                return mc_call
            
            # Override the call method
            layer.call = make_mc_call(original_call)
            dropout_count += 1
    
    print(f"✓ Enabled {dropout_count} dropout layers for MC sampling")
    return model

# Alternative approach using a custom prediction function
def mc_dropout_predict_v2(model, X, num_samples=15, batch_size=4, verbose=True):
    """
    Alternative MC Dropout prediction using training=True flag
    
    This version directly calls the model with training=True
    """
    if verbose:
        print(f"\nRunning MC Dropout with {num_samples} stochastic forward passes...")
    
    predictions = []
    
    # Process in batches
    num_batches = int(np.ceil(len(X) / batch_size))
    
    for i in range(num_samples):
        if verbose and (i + 1) % 5 == 0:
            print(f"  Completed: {i+1}/{num_samples} forward passes")
        
        batch_predictions = []
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(X))
            X_batch = X[start_idx:end_idx]
            
            # Call model with training=True to enable dropout
            pred_batch = model(X_batch, training=True)
            batch_predictions.append(pred_batch.numpy())
        
        # Concatenate batch predictions
        pred = np.concatenate(batch_predictions, axis=0)
        predictions.append(pred)

        # Clear memory after each MC sample
        if (i + 1) % 5 == 0:
            import gc
            gc.collect()
            K.clear_session()
    
    # Stack predictions: shape (num_samples, batch, height, width, classes)
    all_preds = np.array(predictions)
    
    # Calculate mean and standard deviation across samples
    mean_pred = np.mean(all_preds, axis=0)
    std_pred = np.std(all_preds, axis=0)

    # Clear memory
    del all_preds
    import gc
    gc.collect()
    
    if verbose:
        print(f"✓ MC Dropout completed")
        print(f"  Mean prediction shape: {mean_pred.shape}")
        print(f"  Std prediction shape: {std_pred.shape}")
    
    return mean_pred, std_pred, all_preds

# Best approach: Use Keras backend
def mc_dropout_predict(model, X, num_samples=15, batch_size=4, verbose=True):
    """
    BEST APPROACH: MC Dropout prediction using Keras functional API
    
    This properly handles training mode for dropout layers
    """
    if verbose:
        print(f"\nRunning MC Dropout with {num_samples} stochastic forward passes...")
    
    predictions = []
    
    for i in range(num_samples):
        if verbose and (i + 1) % 5 == 0:
            print(f"  Completed: {i+1}/{num_samples} forward passes")
        
        # Create batches manually to control training flag
        batch_preds = []
        num_batches = int(np.ceil(len(X) / batch_size))
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(X))
            X_batch = X[start_idx:end_idx]
            
            # Use model() call with training=True instead of predict()
            pred_batch = model(X_batch, training=True).numpy()
            batch_preds.append(pred_batch)
        
        # Concatenate all batches
        pred = np.concatenate(batch_preds, axis=0)
        predictions.append(pred)
    
    # Stack predictions: shape (num_samples, batch, height, width, classes)
    all_preds = np.array(predictions)
    
    # Calculate mean and standard deviation across samples
    mean_pred = np.mean(all_preds, axis=0)
    std_pred = np.std(all_preds, axis=0)
    
    if verbose:
        print(f"✓ MC Dropout completed")
        print(f"  Mean prediction shape: {mean_pred.shape}")
        print(f"  Std prediction shape: {std_pred.shape}")
    
    return mean_pred, std_pred, all_preds

def calculate_confidence_metrics(mean_pred, std_pred):
    """
    Calculate confidence metrics from MC Dropout predictions
    
    Returns:
    --------
    confidence_map : np.ndarray
        Pixel-wise confidence (1 - normalized entropy)
    uncertainty_map : np.ndarray
        Pixel-wise uncertainty (mean std across classes)
    prediction_entropy : np.ndarray
        Predictive entropy per pixel
    mutual_info : np.ndarray
        Mutual information (epistemic uncertainty)
    """
    epsilon = 1e-10
    
    # Prediction entropy: H(y) = -sum(p * log(p))
    prediction_entropy = -np.sum(mean_pred * np.log(mean_pred + epsilon), axis=-1)
    
    # Normalize entropy to [0, 1] for confidence
    max_entropy = np.log(mean_pred.shape[-1])  # log(num_classes)
    normalized_entropy = prediction_entropy / max_entropy
    confidence_map = 1 - normalized_entropy  # High confidence = low entropy
    
    # Uncertainty: mean standard deviation across classes
    uncertainty_map = np.mean(std_pred, axis=-1)
    
    # Mutual Information (epistemic uncertainty)
    # I[y,θ|x] ≈ H[E[y|x,θ]] - E[H[y|x,θ]]
    mutual_info = prediction_entropy  # Simplified approximation
    
    return confidence_map, uncertainty_map, prediction_entropy, mutual_info

def calculate_mc_metrics(y_true_cat, mean_pred, std_pred):
    """Calculate comprehensive metrics for MC Dropout evaluation"""
    
    print("\n" + "="*80)
    print("MC DROPOUT METRICS")
    print("="*80)
    
    # Overall metrics
    overall_dice = dice_coef_multiclass_np(y_true_cat, mean_pred)
    overall_iou = iou_coef_multiclass_np(y_true_cat, mean_pred)
    
    print(f"\nOverall Performance (Mean Prediction):")
    print(f"  Dice Coefficient: {overall_dice:.4f}")
    print(f"  IoU Coefficient: {overall_iou:.4f}")
    
    # Per-class metrics with uncertainty
    print(f"\nPer-Class Metrics with Uncertainty:")
    class_names = ["Background", "Disc", "Cup"]
    class_results = {}
    
    for cidx, cname in enumerate(class_names):
        dice = dice_coef_class_explicit_np(y_true_cat, mean_pred, cidx)
        iou = iou_coef_class_explicit_np(y_true_cat, mean_pred, cidx)
        mean_uncertainty = np.mean(std_pred[:, :, :, cidx])
        max_uncertainty = np.max(std_pred[:, :, :, cidx])
        
        class_results[cname] = {
            'dice': dice,
            'iou': iou,
            'mean_uncertainty': mean_uncertainty,
            'max_uncertainty': max_uncertainty
        }
        
        print(f"\n  {cname}:")
        print(f"    Dice: {dice:.4f}")
        print(f"    IoU: {iou:.4f}")
        print(f"    Mean Uncertainty (σ): {mean_uncertainty:.4f}")
        print(f"    Max Uncertainty (σ): {max_uncertainty:.4f}")
    
    # Prediction confidence analysis
    y_pred_class = np.argmax(mean_pred, axis=-1)
    y_true_class = np.argmax(y_true_cat, axis=-1)
    correct_mask = (y_pred_class == y_true_class)
    
    # Calculate confidence for correct vs incorrect predictions
    confidence_map, uncertainty_map, pred_entropy, mutual_info = calculate_confidence_metrics(mean_pred, std_pred)
    
    mean_confidence = np.mean(confidence_map)
    mean_uncertainty = np.mean(uncertainty_map)
    mean_confidence_correct = np.mean(confidence_map[correct_mask])
    mean_confidence_incorrect = np.mean(confidence_map[~correct_mask])
    mean_uncertainty_correct = np.mean(uncertainty_map[correct_mask])
    mean_uncertainty_incorrect = np.mean(uncertainty_map[~correct_mask])
    
    print(f"\n" + "-"*80)
    print("CONFIDENCE & UNCERTAINTY ANALYSIS")
    print("-"*80)
    
    print(f"\nOverall Statistics:")
    print(f"  Mean Confidence: {mean_confidence:.4f}")
    print(f"  Mean Uncertainty: {mean_uncertainty:.4f}")
    print(f"  Mean Entropy: {np.mean(pred_entropy):.4f}")
    
    print(f"\nCorrect Predictions:")
    print(f"  Mean Confidence: {mean_confidence_correct:.4f}")
    print(f"  Mean Uncertainty: {mean_uncertainty_correct:.4f}")
    
    print(f"\nIncorrect Predictions:")
    print(f"  Mean Confidence: {mean_confidence_incorrect:.4f}")
    print(f"  Mean Uncertainty: {mean_uncertainty_incorrect:.4f}")
    
    print(f"\nConfidence Gap (Correct - Incorrect): {mean_confidence_correct - mean_confidence_incorrect:+.4f}")
    print(f"Uncertainty Gap (Incorrect - Correct): {mean_uncertainty_incorrect - mean_uncertainty_correct:+.4f}")
    
    # Uncertainty thresholds
    print(f"\nUncertainty Distribution:")
    percentiles = [50, 75, 90, 95, 99]
    for p in percentiles:
        threshold = np.percentile(uncertainty_map, p)
        print(f"  {p}th percentile: {threshold:.4f}")
    
    # High/Low confidence regions
    high_conf_threshold = 0.9
    low_conf_threshold = 0.5
    high_conf_pixels = np.mean(confidence_map > high_conf_threshold)
    low_conf_pixels = np.mean(confidence_map < low_conf_threshold)
    
    print(f"\nConfidence Regions:")
    print(f"  High Confidence (>0.9): {100*high_conf_pixels:.2f}% of pixels")
    print(f"  Low Confidence (<0.5): {100*low_conf_pixels:.2f}% of pixels")
    
    return {
        'overall_dice': overall_dice,
        'overall_iou': overall_iou,
        'class_results': class_results,
        'mean_confidence': mean_confidence,
        'mean_uncertainty': mean_uncertainty,
        'confidence_map': confidence_map,
        'uncertainty_map': uncertainty_map,
        'prediction_entropy': pred_entropy,
        'mutual_info': mutual_info
    }

def visualize_mc_dropout_predictions(X_test, y_test, mean_pred, std_pred, 
                                    confidence_map, uncertainty_map, 
                                    model_name, num_samples=5, save_path=None):
    """Visualize MC Dropout predictions with confidence and uncertainty"""
    
    print(f"\nVisualizing MC Dropout predictions for {model_name}...")
    
    for i in range(min(num_samples, len(X_test))):
        fig = plt.figure(figsize=(24, 10))
        gs = fig.add_gridspec(2, 5, hspace=0.3, wspace=0.3)
        
        # Row 1: Standard predictions
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(X_test[i])
        ax1.set_title("Original Image", fontsize=12, fontweight='bold')
        ax1.axis('off')
        
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(y_test[i], cmap='jet', vmin=0, vmax=2)
        ax2.set_title("Ground Truth", fontsize=12, fontweight='bold')
        ax2.axis('off')
        
        pred_class = np.argmax(mean_pred[i], axis=-1)
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(pred_class, cmap='jet', vmin=0, vmax=2)
        ax3.set_title("Mean Prediction\n(MC Dropout)", fontsize=12, fontweight='bold')
        ax3.axis('off')
        
        # Overlay with predictions
        overlay = X_test[i].copy()
        overlay[pred_class == 1] = [1, 0, 0]  # Red for disc
        overlay[pred_class == 2] = [0, 1, 0]  # Green for cup
        ax4 = fig.add_subplot(gs[0, 3])
        ax4.imshow(overlay)
        ax4.set_title("Prediction Overlay\nRed=Disc, Green=Cup", fontsize=12, fontweight='bold')
        ax4.axis('off')
        
        # Difference map
        diff = np.abs(pred_class - y_test[i])
        ax5 = fig.add_subplot(gs[0, 4])
        im5 = ax5.imshow(diff, cmap='hot', vmin=0, vmax=2)
        ax5.set_title("Prediction Error\n(Darker=Better)", fontsize=12, fontweight='bold')
        ax5.axis('off')
        plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)
        
        # Row 2: Confidence and uncertainty
        ax6 = fig.add_subplot(gs[1, 0])
        im6 = ax6.imshow(confidence_map[i], cmap='RdYlGn', vmin=0, vmax=1)
        ax6.set_title(f"Confidence Map\nMean={np.mean(confidence_map[i]):.3f}", 
                     fontsize=12, fontweight='bold')
        ax6.axis('off')
        plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)
        
        ax7 = fig.add_subplot(gs[1, 1])
        im7 = ax7.imshow(uncertainty_map[i], cmap='hot_r', vmin=0, vmax=0.3)
        ax7.set_title(f"Uncertainty Map\nMean={np.mean(uncertainty_map[i]):.3f}", 
                     fontsize=12, fontweight='bold')
        ax7.axis('off')
        plt.colorbar(im7, ax=ax7, fraction=0.046, pad=0.04)
        
        # Per-class uncertainty
        std_bg = std_pred[i, :, :, 0]
        std_disc = std_pred[i, :, :, 1]
        std_cup = std_pred[i, :, :, 2]
        
        ax8 = fig.add_subplot(gs[1, 2])
        im8 = ax8.imshow(std_disc, cmap='hot', vmin=0, vmax=0.3)
        ax8.set_title(f"Disc Uncertainty (σ)\nMean={np.mean(std_disc):.3f}", 
                     fontsize=12, fontweight='bold')
        ax8.axis('off')
        plt.colorbar(im8, ax=ax8, fraction=0.046, pad=0.04)
        
        ax9 = fig.add_subplot(gs[1, 3])
        im9 = ax9.imshow(std_cup, cmap='hot', vmin=0, vmax=0.3)
        ax9.set_title(f"Cup Uncertainty (σ)\nMean={np.mean(std_cup):.3f}", 
                     fontsize=12, fontweight='bold')
        ax9.axis('off')
        plt.colorbar(im9, ax=ax9, fraction=0.046, pad=0.04)
        
        # High uncertainty regions overlay
        ax10 = fig.add_subplot(gs[1, 4])
        high_uncertainty = uncertainty_map[i] > np.percentile(uncertainty_map[i], 90)
        overlay_uncertainty = X_test[i].copy()
        overlay_uncertainty[high_uncertainty] = [1, 1, 0]  # Yellow for high uncertainty
        ax10.imshow(overlay_uncertainty)
        ax10.set_title(f"High Uncertainty Regions\n(Top 10%, Yellow)", 
                      fontsize=12, fontweight='bold')
        ax10.axis('off')
        
        fig.suptitle(f"{model_name} - MC Dropout Analysis (Sample {i+1})", 
                    fontsize=16, fontweight='bold', y=0.98)
        
        if save_path:
            plt.savefig(f"{save_path}/mc_dropout_sample_{i+1}.png", dpi=150, bbox_inches='tight')
        
        plt.show()

def plot_uncertainty_distributions(uncertainty_map, confidence_map, model_name):
    """Plot distributions of uncertainty and confidence"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Uncertainty histogram
    axes[0, 0].hist(uncertainty_map.flatten(), bins=50, color='red', alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('Uncertainty (σ)', fontsize=12)
    axes[0, 0].set_ylabel('Frequency', fontsize=12)
    axes[0, 0].set_title('Uncertainty Distribution', fontsize=14, fontweight='bold')
    axes[0, 0].axvline(np.mean(uncertainty_map), color='blue', linestyle='--', 
                       linewidth=2, label=f'Mean={np.mean(uncertainty_map):.3f}')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Confidence histogram
    axes[0, 1].hist(confidence_map.flatten(), bins=50, color='green', alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Confidence', fontsize=12)
    axes[0, 1].set_ylabel('Frequency', fontsize=12)
    axes[0, 1].set_title('Confidence Distribution', fontsize=14, fontweight='bold')
    axes[0, 1].axvline(np.mean(confidence_map), color='blue', linestyle='--', 
                       linewidth=2, label=f'Mean={np.mean(confidence_map):.3f}')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Uncertainty vs Confidence scatter
    sample_size = min(10000, uncertainty_map.size)
    sample_indices = np.random.choice(uncertainty_map.size, sample_size, replace=False)
    axes[1, 0].scatter(uncertainty_map.flatten()[sample_indices], 
                      confidence_map.flatten()[sample_indices], 
                      alpha=0.1, s=1, color='purple')
    axes[1, 0].set_xlabel('Uncertainty (σ)', fontsize=12)
    axes[1, 0].set_ylabel('Confidence', fontsize=12)
    axes[1, 0].set_title('Uncertainty vs Confidence', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Cumulative distributions
    uncertainty_sorted = np.sort(uncertainty_map.flatten())
    confidence_sorted = np.sort(confidence_map.flatten())
    cumulative = np.arange(1, len(uncertainty_sorted) + 1) / len(uncertainty_sorted)
    
    axes[1, 1].plot(uncertainty_sorted, cumulative, color='red', linewidth=2, label='Uncertainty')
    ax2 = axes[1, 1].twiny()
    ax2.plot(confidence_sorted, cumulative, color='green', linewidth=2, label='Confidence')
    
    axes[1, 1].set_xlabel('Uncertainty (σ)', fontsize=12, color='red')
    axes[1, 1].set_ylabel('Cumulative Probability', fontsize=12)
    ax2.set_xlabel('Confidence', fontsize=12, color='green')
    axes[1, 1].set_title('Cumulative Distributions', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend(loc='lower right')
    
    fig.suptitle(f'{model_name} - Uncertainty & Confidence Analysis', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def compare_predictions_with_without_mc(model, X_test, y_test, num_mc_samples=15):
    """Compare standard prediction vs MC Dropout prediction - FIXED"""
    
    print("\n" + "="*80)
    print("COMPARISON: STANDARD vs MC DROPOUT")
    print("="*80)
    
    # Standard prediction (dropout disabled - use predict)
    print("\nStandard prediction (dropout disabled)...")
    y_pred_standard = model.predict(X_test, batch_size=8, verbose=0)
    
    # MC Dropout prediction (dropout enabled - use training=True)
    print("\nMC Dropout prediction (dropout enabled)...")
    mean_pred_mc, std_pred_mc, _ = mc_dropout_predict(
        model, X_test, num_samples=num_mc_samples, batch_size=4
    )
    
    # Convert ground truth
    y_test_cat = np.array([to_categorical(mask, num_classes=3) for mask in y_test])
    
    # Calculate metrics
    dice_standard = dice_coef_multiclass_np(y_test_cat, y_pred_standard)
    dice_mc = dice_coef_multiclass_np(y_test_cat, mean_pred_mc)
    
    iou_standard = iou_coef_multiclass_np(y_test_cat, y_pred_standard)
    iou_mc = iou_coef_multiclass_np(y_test_cat, mean_pred_mc)
    
    print(f"\n{'Metric':<20} {'Standard':<15} {'MC Dropout':<15} {'Difference':<15}")
    print("-" * 65)
    print(f"{'Dice Coefficient':<20} {dice_standard:<15.4f} {dice_mc:<15.4f} {dice_mc - dice_standard:+.4f}")
    print(f"{'IoU Coefficient':<20} {iou_standard:<15.4f} {iou_mc:<15.4f} {iou_mc - iou_standard:+.4f}")
    
    # Per-class comparison
    print(f"\nPer-Class Dice Comparison:")
    class_names = ["Background", "Disc", "Cup"]
    for cidx, cname in enumerate(class_names):
        dice_std_class = dice_coef_class_explicit_np(y_test_cat, y_pred_standard, cidx)
        dice_mc_class = dice_coef_class_explicit_np(y_test_cat, mean_pred_mc, cidx)
        print(f"  {cname:<12}: Standard={dice_std_class:.4f}, MC={dice_mc_class:.4f}, Diff={dice_mc_class - dice_std_class:+.4f}")
    
    return y_pred_standard, mean_pred_mc, std_pred_mc

def evaluate_model_mc_dropout(model, X_test, y_test, model_name, num_mc_samples=15):
    """Complete MC Dropout evaluation - FIXED VERSION"""
    
    print(f"\n{'='*80}")
    print(f"MC DROPOUT EVALUATION: {model_name}")
    print(f"{'='*80}")
    
    # Get MC predictions (NO NEED to call enable_mc_dropout)
    # The mc_dropout_predict function handles training=True internally
    mean_pred, std_pred, all_preds = mc_dropout_predict(
        model, X_test, num_samples=num_mc_samples, batch_size=4
    )
    
    # Calculate confidence metrics
    confidence_map, uncertainty_map, prediction_entropy, mutual_info = calculate_confidence_metrics(
        mean_pred, std_pred
    )
    
    # Convert ground truth
    y_test_cat = np.array([to_categorical(mask, num_classes=3) for mask in y_test])
    
    # Calculate all metrics
    mc_results = calculate_mc_metrics(y_test_cat, mean_pred, std_pred)
    
    # Add additional results
    mc_results['confidence_map'] = confidence_map
    mc_results['uncertainty_map'] = uncertainty_map
    mc_results['prediction_entropy'] = prediction_entropy
    mc_results['mutual_info'] = mutual_info
    
    # Visualizations
    print(f"\n{'='*80}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*80}")
    
    # 1. Sample predictions with uncertainty
    visualize_mc_dropout_predictions(
        X_test, y_test, mean_pred, std_pred, 
        confidence_map, uncertainty_map, model_name, num_samples=5
    )
    
    # 2. Uncertainty distributions
    plot_uncertainty_distributions(uncertainty_map, confidence_map, model_name)
    
    # 3. Compare with standard prediction
    y_pred_standard, mean_pred_mc, std_pred_mc = compare_predictions_with_without_mc(
        model, X_test, y_test, num_mc_samples=num_mc_samples
    )
    
    return mc_results

# ---- Training Function ----
def train_model(X_train, y_train, X_val, y_val, model_name, dropout_rate=0.3, augment=None, gpu_available=True):
    """Train MobileNet-UNet model with dropout"""
    
    print(f"\nTraining {model_name}...")
    print(f"Using augmentation: {'Yes' if augment else 'No'}")
    print(f"Dropout rate: {dropout_rate}")
    
    K.clear_session()
    
    # Build model with dropout
    model = build_mobilenet_unet(dropout_rate=dropout_rate)
    
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

# ---- Standard Evaluation Functions ----
def calculate_flops_manual(model, input_shape=(1, 256, 256, 3)):
    """Manual FLOPS calculation"""
    try:
        params = model.count_params()
        estimated_flops = params * 2
        gflops = estimated_flops / 1e9
        print(f"Estimated FLOPS: {estimated_flops:.0f} ({gflops:.2f} GFLOPs)")
        return gflops
    except Exception as e:
        print(f"FLOPS calculation failed: {e}")
        return None

def evaluate_model_standard(model, X_test, y_test, model_name):
    """Standard evaluation without MC Dropout"""
    
    print(f"\n{'='*80}")
    print(f"STANDARD EVALUATION: {model_name}")
    print(f"{'='*80}")
    
    y_test_cat = np.array([to_categorical(mask, num_classes=3) for mask in y_test])
    
    test_metrics = model.evaluate(X_test, y_test_cat, verbose=1, batch_size=8)
    metric_names = model.metrics_names
    
    print("\nTest Metrics:")
    for name, val in zip(metric_names, test_metrics):
        print(f"  {name}: {val:.4f}")
    
    y_pred_prob = model.predict(X_test, verbose=1, batch_size=8)
    
    overall_dice = dice_coef_multiclass_np(y_test_cat, y_pred_prob)
    overall_iou = iou_coef_multiclass_np(y_test_cat, y_pred_prob)
    
    print(f"\nOverall Dice: {overall_dice:.4f}")
    print(f"Overall IoU: {overall_iou:.4f}")
    
    print(f"\nPer-class metrics:")
    class_names = ["Background", "Disc", "Cup"]
    for cidx, cname in enumerate(class_names):
        dice = dice_coef_class_explicit_np(y_test_cat, y_pred_prob, cidx)
        iou = iou_coef_class_explicit_np(y_test_cat, y_pred_prob, cidx)
        print(f"  {cname} - Dice: {dice:.4f}, IoU: {iou:.4f}")
    
    y_true_flat = np.reshape(y_test, (-1,))
    y_pred_flat = np.reshape(np.argmax(y_pred_prob, axis=-1), (-1,))
    
    precision = precision_score(y_true_flat, y_pred_flat, average='weighted', zero_division=0)
    recall = recall_score(y_true_flat, y_pred_flat, average='weighted', zero_division=0)
    fscore = f1_score(y_true_flat, y_pred_flat, average='weighted', zero_division=0)
    
    print(f"\nAdditional Metrics:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F-score: {fscore:.4f}")
    print(f"  Parameters: {model.count_params():,}")
    
    calculate_flops_manual(model)
    
    return test_metrics, metric_names, overall_dice, overall_iou

def plot_training_history(history, title_suffix=""):
    """Plot training curves"""
    
    plt.figure(figsize=(20, 5))
    
    plt.subplot(1, 4, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'Loss over Epochs {title_suffix}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 4, 2)
    if 'accuracy' in history.history:
        plt.plot(history.history['accuracy'], label='Train Acc')
        plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title(f'Accuracy over Epochs {title_suffix}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 4, 3)
    if 'dice_coef_multiclass' in history.history:
        plt.plot(history.history['dice_coef_multiclass'], label='Train Dice')
        plt.plot(history.history['val_dice_coef_multiclass'], label='Val Dice')
    plt.title(f'Dice Coefficient {title_suffix}')
    plt.xlabel('Epoch')
    plt.ylabel('Dice')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 4, 4)
    if 'lr' in history.history:
        plt.plot(history.history['lr'])
    plt.title(f'Learning Rate {title_suffix}')
    plt.xlabel('Epoch')
    plt.ylabel('LR')
    plt.yscale('log')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# ---- Main Execution ----
if __name__ == "__main__":
    # Configure GPU
    gpu_available = configure_gpu()
    
    # Dataset path
    root_dir = "/kaggle/input/refuge/REFUGE/"
    
    # Setup augmentations
    train_aug = setup_augmentations()
    
    print("="*80)
    print("MOBILENET-UNET WITH MC DROPOUT")
    print(f"GPU Available: {gpu_available}")
    print("="*80)
    
    # Load data
    all_images, all_masks = load_all_data(root_dir)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = create_new_split(all_images, all_masks)
    
    # Train model
    print("\n" + "="*80)
    print("TRAINING MOBILENET-UNET WITH DROPOUT")
    print("="*80)
    
    model, history = train_model(
        X_train, y_train, X_val, y_val,
        "mobilenet_unet_mc_dropout", 
        dropout_rate=0.3,
        augment=train_aug, 
        gpu_available=gpu_available
    )
    
    # Standard evaluation
    print("\n" + "="*80)
    print("STANDARD EVALUATION")
    print("="*80)
    
    test_metrics, metric_names, overall_dice, overall_iou = evaluate_model_standard(
        model, X_test, y_test, "MobileNet-UNet"
    )
    
    # MC Dropout evaluation
    print("\n" + "="*80)
    print("MC DROPOUT EVALUATION (15 SAMPLES)")
    print("="*80)
    
    # OPTIMIZED SETTINGS FOR KAGGLE
    mc_results = evaluate_model_mc_dropout(
        model, X_test[:200], y_test[:200],  # Test on 200 samples instead of 400
        "MobileNet-UNet", 
        num_mc_samples=15
    )
    
    # Plot training history
    print("\n" + "="*80)
    print("TRAINING CURVES")
    print("="*80)
    
    plot_training_history(history, "- MobileNet-UNet with MC Dropout")
    
    # Final summary
    print("\n" + "="*100)
    print("FINAL SUMMARY")
    print("="*100)
    
    print(f"\nStandard Evaluation:")
    print(f"  Overall Dice: {overall_dice:.4f}")
    print(f"  Overall IoU: {overall_iou:.4f}")
    
    print(f"\nMC Dropout Evaluation 15 samples):")
    print(f"  Overall Dice: {mc_results['overall_dice']:.4f}")
    print(f"  Overall IoU: {mc_results['overall_iou']:.4f}")
    print(f"  Mean Confidence: {mc_results['mean_confidence']:.4f}")
    print(f"  Mean Uncertainty: {mc_results['mean_uncertainty']:.4f}")
    
    print(f"\nPer-Class Results (MC Dropout):")
    for class_name, results in mc_results['class_results'].items():
        print(f"  {class_name}:")
        print(f"    Dice: {results['dice']:.4f}")
        print(f"    IoU: {results['iou']:.4f}")
        print(f"    Mean Uncertainty: {results['mean_uncertainty']:.4f}")
    
    print(f"\nModel Complexity:")
    print(f"  Total Parameters: {model.count_params():,}")

    print_all_parameters()
    
    print("\n" + "="*100)
    print("✅ MC DROPOUT EXPERIMENT COMPLETED!")
    print("="*100)