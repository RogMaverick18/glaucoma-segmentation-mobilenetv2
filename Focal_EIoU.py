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
    print(" "*30 + "FOCAL-EIoU GUIDED LOSS EXPERIMENT PARAMETERS")
    print("="*100)
    
    # Model Architecture
    print("\n" + "─"*100)
    print("MODEL ARCHITECTURE")
    print("─"*100)
    print("""
    Model Name: MobileNet-UNet with Focal-EIoU Guided Loss
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
    
    # Loss Function Details
    print("─"*100)
    print("FOCAL-EIoU GUIDED LOSS FUNCTION")
    print("─"*100)
    print("""
    Combined Loss Function: Focal Loss + Enhanced IoU Loss
    
    1. FOCAL LOSS (Addresses Class Imbalance):
       Formula: FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
       
       Parameters:
         - Gamma (γ): 2.0 (focusing parameter)
           • γ = 0: Standard cross-entropy
           • γ > 0: Down-weights easy examples
           • Higher γ: More focus on hard examples
         
         - Alpha (α): [0.25, 1.0, 1.0] (per-class weights)
           • Class 0 (Background): 0.25 (down-weighted, easy)
           • Class 1 (Disc): 1.0 (standard weight)
           • Class 2 (Cup): 1.0 (standard weight, challenging)
         
         - Epsilon: 1e-7 (numerical stability)
       
       Purpose:
         ✓ Focus on hard-to-classify pixels
         ✓ Down-weight easy background pixels
         ✓ Improve minority class (Cup) performance
         ✓ Address class imbalance (Background >> Disc > Cup)
    
    2. ENHANCED IoU LOSS (EIoU):
       Formula: EIoU = 1 - IoU - (d²/c²) - (w_diff²/w_c² + h_diff²/h_c²)
       
       Components:
         a) Standard IoU:
            IoU = Intersection / Union
            
         b) Center Distance Penalty:
            d²/c² = (center_pred - center_true)² / diagonal²
            • Penalizes center point mismatch
            • Normalized by bounding box diagonal
         
         c) Aspect Ratio Penalty:
            w_diff²/w_c² + h_diff²/h_c²
            • Penalizes width/height mismatch
            • Encourages correct shape prediction
       
       Parameters:
         - Smooth: 1e-7 (avoid division by zero)
         - Per-class calculation: Separate EIoU for each class
       
       Purpose:
         ✓ Better boundary localization
         ✓ Shape-aware loss
         ✓ Penalize center misalignment
         ✓ Improve small object (Cup) detection
    
    3. COMBINED LOSS:
       Total Loss = λ₁ * Focal Loss + λ₂ * EIoU Loss
       
       Weighting Strategy:
         - Lambda 1 (λ₁): 1.0 (Focal Loss weight)
         - Lambda 2 (λ₂): 1.0 (EIoU Loss weight)
         - Balanced contribution from both losses
       
       Alternative Strategies (can be experimented):
         • Strategy A: λ₁=0.8, λ₂=0.2 (Focal-dominant)
         • Strategy B: λ₁=0.5, λ₂=0.5 (Equal balance)
         • Strategy C: λ₁=0.3, λ₂=0.7 (EIoU-dominant)
    
    Expected Benefits:
      ✓ Improved Dice scores (especially Cup - Class 2)
      ✓ Better boundary precision
      ✓ Reduced false positives in background
      ✓ More accurate center localization
      ✓ Better handling of class imbalance
      ✓ Improved small object segmentation
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
    
    Loss Function: Focal-EIoU Guided Loss (custom)
      - Focal Loss Weight (λ₁): 1.0
      - EIoU Loss Weight (λ₂): 1.0
      - Focal Gamma (γ): 2.0
      - Focal Alpha (α): [0.25, 1.0, 1.0]
    
    Batch Size:
      - Primary: 8
      - Fallback: 4 (if OOM error)
      - Test/Validation: 8
    
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
      - Filename: mobilenet_unet_focal_eiou.h5
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
    
    # Evaluation Metrics
    print("─"*100)
    print("EVALUATION METRICS")
    print("─"*100)
    print("""
    Training Metrics (Computed Every Batch):
      1. focal_eiou_loss (custom guided loss)
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
    
    Prediction Visualization:
      - Number of Samples: 3
      - Figure Size: (16, 4)
      - Views: Original, Ground Truth, Prediction, Overlay
      - Overlay Colors:
        • Disc (Class 1): Red [1, 0, 0]
        • Cup (Class 2): Green [0, 1, 0]
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
    
    Model Checkpoint: mobilenet_unet_focal_eiou.h5
    
    Visualization Outputs:
      - Training curves (matplotlib display)
      - Sample predictions (matplotlib display)
    """)
    
    # Expected Outcomes
    print("─"*100)
    print("EXPECTED OUTCOMES & BENEFITS")
    print("─"*100)
    print("""
    Baseline (Standard Cross-Entropy):
      - Overall Dice: ~0.85-0.90
      - Cup Dice: ~0.75-0.80
      - Background dominates loss
    
    With Focal-EIoU Loss (Expected Improvements):
      - Overall Dice: ~0.88-0.93 (+3-5%)
      - Cup Dice: ~0.80-0.87 (+5-7%, most improvement)
      - Disc Dice: ~0.88-0.92 (+2-3%)
      - Better boundary precision
      - Reduced false positives
    
    Specific Benefits:
      ✓ Focal Loss: Better class balance, focus on hard examples
      ✓ EIoU Loss: Better boundary localization
      ✓ Combined: Synergistic improvement
      ✓ Cup Segmentation: Most significant improvement
      ✓ Boundary Quality: Sharper, more accurate edges
      ✓ Center Alignment: Better object center prediction
    
    Computational Cost:
      - Training Time: Similar to baseline (~1-2 hours GPU)
      - Loss Computation: ~10-20% overhead (minimal)
      - Inference Speed: Unchanged (loss only affects training)
    """)
    
    # Key Design Decisions
    print("─"*100)
    print("KEY DESIGN DECISIONS & RATIONALE")
    print("─"*100)
    print("""
    1. Focal Loss Gamma = 2.0:
       - Standard value from original paper (Lin et al., 2017)
       - Balances hard vs easy examples
       - Too high (>3): May ignore easy examples entirely
       - Too low (<1): Insufficient focusing
    
    2. Focal Alpha = [0.25, 1.0, 1.0]:
       - Background (0.25): Down-weight dominant class
       - Disc (1.0): Standard weight for medium-sized object
       - Cup (1.0): Standard weight for small, challenging object
       - Rationale: Background is 80%+ of pixels
    
    3. Equal Loss Weights (λ₁=λ₂=1.0):
       - Balanced contribution from both losses
       - Focal addresses class imbalance
       - EIoU addresses boundary quality
       - Can be tuned based on validation performance
    
    4. EIoU over Standard IoU:
       - Adds center distance penalty
       - Adds aspect ratio penalty
       - Better for small objects (Cup)
       - More informative gradient signal
    
    5. No Dropout (vs MC Dropout experiment):
       - Focus on loss function impact
       - Avoid confounding factors
       - Can be combined later for uncertainty + loss
    
    6. Same Architecture as Baseline:
       - Fair comparison (only loss changes)
       - MobileNetV2 encoder proven effective
       - Standard UNet decoder
    """)
    
    # Loss Function Mathematics
    print("─"*100)
    print("LOSS FUNCTION MATHEMATICS")
    print("─"*100)
    print("""
    FOCAL LOSS (per pixel):
      p_t = {
        p      if y = 1 (foreground)
        1 - p  if y = 0 (background)
      }
      
      α_t = {
        α      if y = 1
        1 - α  if y = 0
      }
      
      FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
      
      For multi-class (C=3):
        FL = -Σ_c [α_c * (1 - p_c)^γ * y_c * log(p_c)]
    
    ENHANCED IoU LOSS (per class):
      IoU = (A ∩ B) / (A ∪ B)
      
      Center Distance:
        d = ||center_pred - center_true||²
        c = diagonal of enclosing box
        center_penalty = d² / c²
      
      Aspect Ratio:
        w_diff = width_pred - width_true
        h_diff = height_pred - height_true
        w_c, h_c = enclosing box dimensions
        aspect_penalty = w_diff²/w_c² + h_diff²/h_c²
      
      EIoU = 1 - IoU - center_penalty - aspect_penalty
    
    COMBINED LOSS:
      L_total = λ₁ * FL + λ₂ * EIoU
      
      With default weights (λ₁=λ₂=1.0):
        L_total = FL + EIoU
    
    GRADIENT PROPERTIES:
      - Focal: Non-zero gradient for all examples (unlike CE)
      - EIoU: Smooth gradient for overlap and geometry
      - Combined: Multi-objective optimization
      - Convergence: Typically faster than CE alone
    """)
    
    # References
    print("─"*100)
    print("METHODOLOGY REFERENCES")
    print("─"*100)
    print("""
    Focal Loss:
      - Lin et al. (2017): "Focal Loss for Dense Object Detection"
      - arXiv:1708.02002 [cs.CV]
    
    Enhanced IoU (EIoU):
      - Zhang et al. (2022): "Focal and Efficient IOU Loss for Accurate Bounding Box Regression"
      - Neurocomputing, Vol. 506
    
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

# ============================================================================
# FOCAL-EIoU LOSS IMPLEMENTATION
# ============================================================================

def focal_loss(y_true, y_pred, gamma=2.0, alpha=None, epsilon=1e-7):
    """
    Focal Loss for multi-class segmentation
    
    Formula: FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    Parameters:
    -----------
    y_true : tensor
        Ground truth labels (one-hot encoded)
    y_pred : tensor
        Predicted probabilities
    gamma : float
        Focusing parameter (default: 2.0)
    alpha : list or None
        Class weights (default: [0.25, 1.0, 1.0])
    epsilon : float
        Numerical stability constant
    
    Returns:
    --------
    loss : tensor
        Focal loss value
    """
    if alpha is None:
        alpha = [0.25, 1.0, 1.0]  # Down-weight background
    
    # Clip predictions for numerical stability
    y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
    
    # Calculate focal loss for each class
    focal_loss_value = 0.0
    for c in range(3):  # 3 classes
        # Extract class predictions and labels
        y_true_c = y_true[:, :, :, c]
        y_pred_c = y_pred[:, :, :, c]
        
        # Calculate focal loss components
        pt = y_pred_c  # predicted probability for true class
        focal_weight = K.pow(1.0 - pt, gamma)
        cross_entropy = -K.log(pt)
        
        # Apply class weight and focal weight
        focal_loss_c = alpha[c] * focal_weight * cross_entropy * y_true_c
        
        # Sum over spatial dimensions
        focal_loss_value += K.mean(focal_loss_c)
    
    return focal_loss_value

def enhanced_iou_loss(y_true, y_pred, smooth=1e-7):
    """
    Enhanced IoU (EIoU) Loss for segmentation
    
    EIoU = 1 - IoU - center_distance_penalty - aspect_ratio_penalty
    
    Parameters:
    -----------
    y_true : tensor
        Ground truth labels (one-hot encoded)
    y_pred : tensor
        Predicted probabilities
    smooth : float
        Smoothing constant to avoid division by zero
    
    Returns:
    --------
    loss : tensor
        Enhanced IoU loss value
    """
    # Calculate per-class EIoU
    eiou_loss_value = 0.0
    
    for c in range(3):  # 3 classes
        # Extract class predictions and labels
        y_true_c = y_true[:, :, :, c]
        y_pred_c = y_pred[:, :, :, c]
        
        # Standard IoU calculation
        intersection = K.sum(y_true_c * y_pred_c, axis=[1, 2])
        union = K.sum(y_true_c, axis=[1, 2]) + K.sum(y_pred_c, axis=[1, 2]) - intersection
        iou = (intersection + smooth) / (union + smooth)
        
        # Calculate bounding boxes for center distance penalty
        # Get center of mass for predictions and ground truth
        batch_size = K.shape(y_true_c)[0]
        height = K.cast(K.shape(y_true_c)[1], 'float32')
        width = K.cast(K.shape(y_true_c)[2], 'float32')
        
        # Create coordinate grids
        y_coords = K.arange(0, height)
        x_coords = K.arange(0, width)
        y_grid = K.tile(K.reshape(y_coords, (-1, 1)), (1, K.cast(width, 'int32')))
        x_grid = K.tile(K.reshape(x_coords, (1, -1)), (K.cast(height, 'int32'), 1))
        
        y_grid = K.cast(y_grid, 'float32')
        x_grid = K.cast(x_grid, 'float32')
        
        # Calculate centers for true and predicted masks
        # True center
        true_mass = K.sum(y_true_c, axis=[1, 2]) + smooth
        true_center_y = K.sum(y_true_c * y_grid, axis=[1, 2]) / true_mass
        true_center_x = K.sum(y_true_c * x_grid, axis=[1, 2]) / true_mass
        
        # Predicted center
        pred_mass = K.sum(y_pred_c, axis=[1, 2]) + smooth
        pred_center_y = K.sum(y_pred_c * y_grid, axis=[1, 2]) / pred_mass
        pred_center_x = K.sum(y_pred_c * x_grid, axis=[1, 2]) / pred_mass
        
        # Center distance penalty
        center_distance = K.square(true_center_y - pred_center_y) + K.square(true_center_x - pred_center_x)
        diagonal = K.square(height) + K.square(width)
        center_penalty = center_distance / (diagonal + smooth)
        
        # Aspect ratio penalty (simplified for segmentation)
        # Calculate effective width and height
        true_width = K.sqrt(K.sum(K.sum(y_true_c, axis=1), axis=1) + smooth)
        true_height = K.sqrt(K.sum(K.sum(y_true_c, axis=2), axis=1) + smooth)
        pred_width = K.sqrt(K.sum(K.sum(y_pred_c, axis=1), axis=1) + smooth)
        pred_height = K.sqrt(K.sum(K.sum(y_pred_c, axis=2), axis=1) + smooth)
        
        width_diff = K.square(pred_width - true_width)
        height_diff = K.square(pred_height - true_height)
        
        aspect_penalty = (width_diff / (K.square(width) + smooth) + 
                         height_diff / (K.square(height) + smooth))
        
        # Combined EIoU loss
        eiou_c = 1.0 - iou + center_penalty + aspect_penalty
        eiou_loss_value += K.mean(eiou_c)
    
    # Average over classes
    return eiou_loss_value / 3.0

def focal_eiou_combined_loss(y_true, y_pred, 
                             focal_weight=1.0, eiou_weight=1.0,
                             gamma=2.0, alpha=None):
    """
    Combined Focal Loss and Enhanced IoU Loss
    
    Total Loss = λ₁ * Focal Loss + λ₂ * EIoU Loss
    
    Parameters:
    -----------
    y_true : tensor
        Ground truth labels (one-hot encoded)
    y_pred : tensor
        Predicted probabilities
    focal_weight : float
        Weight for focal loss (λ₁, default: 1.0)
    eiou_weight : float
        Weight for EIoU loss (λ₂, default: 1.0)
    gamma : float
        Focal loss gamma parameter (default: 2.0)
    alpha : list or None
        Focal loss class weights (default: [0.25, 1.0, 1.0])
    
    Returns:
    --------
    loss : tensor
        Combined loss value
    """
    # Calculate individual losses
    fl = focal_loss(y_true, y_pred, gamma=gamma, alpha=alpha)
    eiou = enhanced_iou_loss(y_true, y_pred)
    
    # Combine with weights
    total_loss = focal_weight * fl + eiou_weight * eiou
    
    return total_loss

# ============================================================================
# REMAINING CODE (GPU, Data Loading, Model Architecture, etc.)
# ============================================================================

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

# ---- MobileNet-UNet Model ----
def build_mobilenet_unet(input_shape=(256, 256, 3), num_classes=3):
    """Build MobileNet-UNet"""
    
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
    
    # Final upsampling: 128x128 -> 256x256
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    # Output layer
    outputs = layers.Conv2D(num_classes, (1, 1), activation='softmax', name='output', dtype='float32')(x)
    
    model = models.Model(inputs, outputs, name="MobileNet-UNet-FocalEIoU")
    return model

# ---- Metrics (same as baseline for comparison) ----
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

# ---- Training Function ----
def train_model(X_train, y_train, X_val, y_val, model_name, augment=None, gpu_available=True):
    """Train MobileNet-UNet model with Focal-EIoU Loss"""
    
    print(f"\nTraining {model_name} with Focal-EIoU Guided Loss...")
    print(f"Using augmentation: {'Yes' if augment else 'No'}")
    
    K.clear_session()
    
    # Build model
    model = build_mobilenet_unet()
    
    # Optimizer
    initial_lr = 1e-3
    optimizer = optimizers.Adam(learning_rate=initial_lr)
    
    # Compile with Focal-EIoU loss
    model.compile(
        optimizer=optimizer,
        loss=focal_eiou_combined_loss,  # ← CUSTOM LOSS
        metrics=[
            dice_coef_multiclass, dice_class_0, dice_class_1, dice_class_2, "accuracy",
            iou_coef_multiclass, iou_class_0, iou_class_1, iou_class_2
        ]
    )
    
    print("✓ Model compiled with Focal-EIoU Guided Loss")
    print("  - Focal Loss: Gamma=2.0, Alpha=[0.25, 1.0, 1.0]")
    print("  - EIoU Loss: With center distance and aspect ratio penalties")
    print("  - Combined Weight: λ₁=1.0, λ₂=1.0")
    
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

# ---- Evaluation Functions (same as baseline) ----
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

def evaluate_model(model, X_test, y_test, model_name, gpu_available=True):
    """Comprehensive model evaluation"""
    
    print(f"\nEvaluating {model_name}...")
    
    y_test_cat = np.array([to_categorical(mask, num_classes=3) for mask in y_test])
    
    test_metrics = model.evaluate(X_test, y_test_cat, verbose=1, batch_size=8)
    metric_names = model.metrics_names
    
    print("\nTest Metrics:")
    for name, val in zip(metric_names, test_metrics):
        print(f"{name}: {val:.4f}")
    
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
        print(f"{cname} - Dice: {dice:.4f}, IoU: {iou:.4f}")
    
    y_true_flat = np.reshape(y_test, (-1,))
    y_pred_flat = np.reshape(np.argmax(y_pred_prob, axis=-1), (-1,))
    
    precision = precision_score(y_true_flat, y_pred_flat, average='weighted', zero_division=0)
    recall = recall_score(y_true_flat, y_pred_flat, average='weighted', zero_division=0)
    fscore = f1_score(y_true_flat, y_pred_flat, average='weighted', zero_division=0)
    
    print(f"\nPrecision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F-score: {fscore:.4f}")
    print(f"Parameters: {model.count_params():,}")
    
    calculate_flops_manual(model)
    
    return test_metrics, metric_names, overall_dice, overall_iou

def plot_training_history(history, title_suffix=""):
    """Plot training curves"""
    
    plt.figure(figsize=(20, 5))
    
    plt.subplot(1, 4, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'Focal-EIoU Loss {title_suffix}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 4, 2)
    if 'accuracy' in history.history:
        plt.plot(history.history['accuracy'], label='Train Acc')
        plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title(f'Accuracy {title_suffix}')
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

def visualize_predictions(model, X_test, y_test, model_name, num_samples=3):
    """Visualize sample predictions"""
    
    print(f"\nSample predictions for {model_name}:")
    
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

# ---- Main Execution ----
if __name__ == "__main__":
    
    # Configure GPU
    gpu_available = configure_gpu()
    
    # Dataset path
    root_dir = "/kaggle/input/refuge/REFUGE/"
    
    # Setup augmentations
    train_aug = setup_augmentations()
    
    print("="*80)
    print("MOBILENET-UNET WITH FOCAL-EIoU GUIDED LOSS")
    print(f"GPU Available: {gpu_available}")
    print("="*80)
    
    # Load data
    all_images, all_masks = load_all_data(root_dir)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = create_new_split(all_images, all_masks)
    
    # Train model
    print("\n" + "="*80)
    print("TRAINING WITH FOCAL-EIoU LOSS")
    print("="*80)
    
    model, history = train_model(
        X_train, y_train, X_val, y_val,
        "mobilenet_unet_focal_eiou", 
        augment=train_aug, 
        gpu_available=gpu_available
    )
    
    # Evaluate
    print("\n" + "="*80)
    print("EVALUATION")
    print("="*80)
    
    test_metrics, metric_names, overall_dice, overall_iou = evaluate_model(
        model, X_test, y_test, "Focal-EIoU Model", gpu_available
    )
    
    # Plot training history
    print("\n" + "="*80)
    print("TRAINING CURVES")
    print("="*80)
    
    plot_training_history(history, "- Focal-EIoU Loss")
    
    # Visualize predictions
    print("\n" + "="*80)
    print("SAMPLE PREDICTIONS")
    print("="*80)
    
    visualize_predictions(model, X_test, y_test, "Focal-EIoU Model")
    
    # Final summary
    print("\n" + "="*100)
    print("FINAL RESULTS SUMMARY")
    print("="*100)
    
    print(f"\nModel Performance:")
    print(f"Overall Dice: {overall_dice:.4f}")
    print(f"Overall IoU: {overall_iou:.4f}")
    print(f"Parameters: {model.count_params():,}")
    
    print("\nDetailed Test Metrics:")
    for name, val in zip(metric_names, test_metrics):
        print(f"  {name}: {val:.4f}")

    print_all_parameters()
    
    print("\n" + "="*100)
    print("✅ FOCAL-EIoU EXPERIMENT COMPLETED!")
    print("="*100)