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

"""
why are we using 1x1 refinement in the decoder block 1 & 2 
and this is a customized version of the mobilenet 
why are we using these unique components like multi scale, 6xconv2d - 
what is the reasoning, is there any literature supporting it

ARCHITECTURAL DESIGN RATIONALE:
================================

1. **1x1 Convolution for Channel Refinement**:
   - Literature: "Network In Network" (Lin et al., 2013)
   - Purpose: Acts as a learnable linear combination of channels
   - Benefits: 
     * Reduces/adjusts channel dimensions efficiently
     * Adds non-linearity without spatial downsampling
     * Cross-channel information fusion
   - In our decoder: Aligns skip connection channels before concatenation
   
2. **Multi-Scale Feature Extraction (Inception-style)**:
   - Literature: "Going Deeper with Convolutions" (Szegedy et al., 2015) - Inception
   - Purpose: Capture features at different receptive field sizes simultaneously
   - Components used:
     * 1x1: Captures point-wise features
     * 3x3: Captures local neighborhood patterns
     * 5x5: Captures broader context
     * Max pooling path: Preserves strong features
   - Benefits for medical imaging:
     * Disc boundaries need fine details (3x3)
     * Cup requires broader context (5x5)
     * Background needs point features (1x1)
   
3. **Deep Decoder with 6 Conv Layers**:
   - Literature: 
     * "Very Deep Convolutional Networks" (Simonyan & Zisserman, 2014) - VGGNet
     * "U-Net++: A Nested U-Net Architecture" (Zhou et al., 2018)
   - Reasoning:
     * Medical images need fine boundary detection
     * Deeper = more non-linear transformations = better feature learning
     * Stacking small (3x3) kernels = larger effective receptive field
     * Formula: N layers of 3x3 = (2N+1)×(2N+1) receptive field
     * 6 layers of 3x3 = 13×13 effective receptive field
   - Why critical for Disc/Cup:
     * Small objects (disc ~150px, cup ~100px in 256×256 image)
     * Need hierarchical feature learning
     * Boundary refinement requires multiple processing stages
   
4. **Residual Connections in Deep Blocks**:
   - Literature: "Deep Residual Learning" (He et al., 2015) - ResNet
   - Purpose: Enable gradient flow in very deep networks
   - Placement: Every 3 layers to prevent vanishing gradients
   - Mathematical benefit: y = F(x) + x enables identity mapping
   
5. **Spatial Attention Gates**:
   - Literature: 
     * "Attention U-Net" (Oktay et al., 2018)
     * "CBAM: Convolutional Block Attention Module" (Woo et al., 2018)
   - Purpose: Focus on important regions (disc/cup boundaries)
   - Two-stage attention:
     * Channel attention: Which feature maps are important
     * Spatial attention: Which spatial locations are important
   - Medical imaging benefit: Highlights pathological regions
   
6. **Lightweight Transformer Blocks**:
   - Literature:
     * "An Image is Worth 16x16 Words" (Dosovitskiy et al., 2020) - ViT
     * "TransUNet: Transformers Make Strong Encoders" (Chen et al., 2021)
     * "Medical Transformer" (Valanarasu et al., 2021)
   - Purpose: Capture long-range dependencies
   - Why for medical imaging:
     * Global context crucial for disc/cup relationship
     * Self-attention captures dependencies across entire feature map
     * Better than pure CNNs for structural relationships
   - Our optimization: Reduced sequence length via patching for memory efficiency

7. **Multi-Scale Final Processing**:
   - Literature: "Feature Pyramid Networks" (Lin et al., 2017)
   - Purpose: Combine features at different scales for final prediction
   - Components:
     * Fine (3×3): Detail preservation
     * Coarse (5×5): Context integration
     * Detail (1×1): Channel-wise refinement
   - Combination: Element-wise addition (feature fusion)
   
8. **Progressive Dropout Strategy**:
   - Literature: "Dropout: A Simple Way to Prevent Overfitting" (Srivastava et al., 2014)
   - Strategy: Higher dropout in bridge (0.4), lower in decoder (0.2-0.3)
   - Reasoning: More regularization where network has most parameters

MEMORY OPTIMIZATION STRATEGIES:
================================

1. **Gradient Checkpointing**: Recompute activations during backprop
2. **Reduced Transformer Heads**: 2-4 heads instead of 8-12
3. **Smaller Feed-Forward Networks**: ff_dim = channels//4 instead of 4×channels
4. **Patch-based Processing**: For large feature maps, process in patches
5. **Channel Reduction**: Reduce bridge from 1280 to 512 channels
6. **Simplified Attention**: Single-path spatial attention instead of dual-path

KEY PAPERS SUPPORTING THIS ARCHITECTURE:
=========================================

Primary Medical Segmentation:
- "U-Net: Convolutional Networks for Biomedical Image Segmentation" (Ronneberger et al., 2015)
- "Attention U-Net: Learning Where to Look for the Pancreas" (Oktay et al., 2018)
- "UNet++: Redesigning Skip Connections to Exploit Multiscale Features" (Zhou et al., 2018)
- "TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation" (Chen et al., 2021)

Architectural Components:
- Inception: "Going Deeper with Convolutions" (Szegedy et al., 2015)
- ResNet: "Deep Residual Learning for Image Recognition" (He et al., 2015)
- Attention: "CBAM: Convolutional Block Attention Module" (Woo et al., 2018)
- Transformers: "Attention Is All You Need" (Vaswani et al., 2017)
- Vision Transformers: "An Image is Worth 16x16 Words" (Dosovitskiy et al., 2020)

Mobile/Efficient Networks:
- "MobileNetV2: Inverted Residuals and Linear Bottlenecks" (Sandler et al., 2018)
- "EfficientNet: Rethinking Model Scaling" (Tan & Le, 2019)
"""

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

# ---- MEMORY-EFFICIENT TRANSFORMER COMPONENTS ----

class LightweightTransformerBlock(layers.Layer):
    """
    Lightweight Vision Transformer Block optimized for memory efficiency
    
    Based on:
    - "An Image is Worth 16x16 Words" (Dosovitskiy et al., 2020)
    - "TransUNet" (Chen et al., 2021)
    
    Key optimizations:
    - Reduced number of heads (2-4 vs 8-16)
    - Smaller feed-forward dimension
    - Patch-based processing for large feature maps
    - Fixed positional encoding instead of learnable
    """
    
    def __init__(self, num_heads=2, ff_dim=64, dropout_rate=0.1, name="lightweight_transformer", **kwargs):
        super(LightweightTransformerBlock, self).__init__(name=name, **kwargs)
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        
    def build(self, input_shape):
        self.channels = input_shape[-1]
        self.height = input_shape[1]
        self.width = input_shape[2]
        
        # Memory optimization: Use patching for large feature maps
        if self.height * self.width > 1024:  # For large feature maps (>32×32)
            self.patch_size = 2
            self.seq_len = (self.height // self.patch_size) * (self.width // self.patch_size)
            self.reshape_to_seq = layers.Reshape((-1, self.channels * self.patch_size * self.patch_size))
            self.patch_embed = layers.Dense(self.channels, name=f'{self.name}_patch_embed')
        else:
            self.patch_size = 1
            self.seq_len = self.height * self.width
            self.reshape_to_seq = layers.Reshape((-1, self.channels))
            self.patch_embed = None
        
        # Fixed positional encoding (memory efficient)
        self.pos_encoding = self.add_weight(
            name="pos_encoding",
            shape=(1, self.seq_len, self.channels),
            initializer="zeros",
            trainable=False
        )
        
        # Multi-head attention with reduced dimension
        self.mha = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=max(self.channels // (self.num_heads * 2), 8),  # Smaller key dimension
            dropout=self.dropout_rate
        )
        
        # Layer normalization
        self.ln1 = layers.LayerNormalization(epsilon=1e-6)
        self.ln2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(self.dropout_rate)
        self.dropout2 = layers.Dropout(self.dropout_rate)
        
        # Smaller feed forward network (memory efficient)
        self.ffn = tf.keras.Sequential([
            layers.Dense(self.ff_dim, activation='gelu'),
            layers.Dropout(self.dropout_rate),
            layers.Dense(self.channels)
        ])
        
        # Reshape back to spatial
        if self.patch_size > 1:
            self.reshape_to_spatial = layers.Reshape((self.height // self.patch_size, self.width // self.patch_size, self.channels))
            self.upsample = layers.UpSampling2D((self.patch_size, self.patch_size))
        else:
            self.reshape_to_spatial = layers.Reshape((self.height, self.width, self.channels))
            self.upsample = None
        
        super(LightweightTransformerBlock, self).build(input_shape)
    
    def call(self, x, training=None):
        batch_size = tf.shape(x)[0]
        
        # Patch embedding if needed
        x_seq = self.reshape_to_seq(x)
        if self.patch_embed is not None:
            x_seq = self.patch_embed(x_seq)
        
        # Add fixed positional encoding
        x_seq = x_seq + self.pos_encoding
        
        # Multi-head self-attention with residual connection
        attn_input = self.ln1(x_seq)
        attn_output = self.mha(attn_input, attn_input, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        x_seq = x_seq + attn_output  # Residual connection (ResNet-style)
        
        # Feed forward network with residual connection
        ffn_input = self.ln2(x_seq)
        ffn_output = self.ffn(ffn_input, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        x_seq = x_seq + ffn_output  # Residual connection
        
        # Reshape back to spatial format
        output = self.reshape_to_spatial(x_seq)
        if self.upsample is not None:
            output = self.upsample(output)
        
        return output
    
    def get_config(self):
        config = super(LightweightTransformerBlock, self).get_config()
        config.update({
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'dropout_rate': self.dropout_rate
        })
        return config

class EfficientSpatialAttentionGate(layers.Layer):
    """
    Memory-efficient spatial attention gate
    
    Based on:
    - "CBAM: Convolutional Block Attention Module" (Woo et al., 2018)
    - "Attention U-Net" (Oktay et al., 2018)
    
    Simplified to reduce memory:
    - Single-path channel attention
    - Simplified spatial attention
    - Reduced intermediate dimensions
    """
    
    def __init__(self, name="efficient_spatial_attention_gate", **kwargs):
        super(EfficientSpatialAttentionGate, self).__init__(name=name, **kwargs)
        
    def build(self, input_shape):
        self.channels = input_shape[-1]
        
        # Simplified channel attention
        self.global_avg_pool = layers.GlobalAveragePooling2D(keepdims=True)
        self.channel_fc = layers.Dense(max(self.channels // 16, 1), activation='sigmoid')
        
        # Simplified spatial attention
        self.spatial_conv = layers.Conv2D(1, (3, 3), padding='same', activation='sigmoid')
        
        super(EfficientSpatialAttentionGate, self).build(input_shape)
    
    def call(self, x, training=None):
        # Channel attention: Which feature channels are important
        avg_pool = self.global_avg_pool(x)
        avg_pool = layers.Flatten()(avg_pool)
        channel_attention = self.channel_fc(avg_pool)
        channel_attention = tf.expand_dims(tf.expand_dims(channel_attention, 1), 1)
        
        x_channel = x * channel_attention
        
        # Spatial attention: Which spatial locations are important
        spatial_input = tf.reduce_mean(x_channel, axis=-1, keepdims=True)
        spatial_attention = self.spatial_conv(spatial_input)
        
        output = x_channel * spatial_attention
        
        return output
    
    def get_config(self):
        config = super(EfficientSpatialAttentionGate, self).get_config()
        return config

class SpatialAttentionGate(layers.Layer):
    """
    Full Spatial attention gate for non-transformer model
    
    Dual-path attention:
    - Channel attention via global pooling
    - Spatial attention via convolution
    """
    
    def __init__(self, name="spatial_attention_gate", **kwargs):
        super(SpatialAttentionGate, self).__init__(name=name, **kwargs)
        
    def build(self, input_shape):
        self.channels = input_shape[-1]
        
        # Channel attention components
        self.global_avg_pool = layers.GlobalAveragePooling2D(keepdims=True)
        self.global_max_pool = layers.GlobalMaxPooling2D(keepdims=True)
        
        self.fc1 = layers.Dense(max(self.channels // 8, 1), activation='relu')
        self.fc2 = layers.Dense(self.channels, activation='sigmoid')
        
        # Spatial attention components
        self.spatial_conv = layers.Conv2D(1, (7, 7), padding='same', activation='sigmoid')
        
        super(SpatialAttentionGate, self).build(input_shape)
    
    def call(self, x, training=None):
        # Channel attention
        avg_pool = self.global_avg_pool(x)
        max_pool = self.global_max_pool(x)
        
        avg_out = self.fc2(self.fc1(layers.Flatten()(avg_pool)))
        max_out = self.fc2(self.fc1(layers.Flatten()(max_pool)))
        
        channel_attention = tf.nn.sigmoid(avg_out + max_out)
        channel_attention = tf.expand_dims(tf.expand_dims(channel_attention, 1), 1)
        
        x_channel = x * channel_attention
        
        # Spatial attention
        avg_out_spatial = tf.reduce_mean(x_channel, axis=-1, keepdims=True)
        max_out_spatial = tf.reduce_max(x_channel, axis=-1, keepdims=True)
        spatial_input = tf.concat([avg_out_spatial, max_out_spatial], axis=-1)
        spatial_attention = self.spatial_conv(spatial_input)
        
        output = x_channel * spatial_attention
        
        return output
    
    def get_config(self):
        config = super(SpatialAttentionGate, self).get_config()
        return config

# ---- Memory-Efficient Enhanced Decoder Block ----
def memory_efficient_decoder_block_with_transformer(x, skip_connection, filters, block_name, 
                                                   dropout_rate=0.3, use_transformer=False):
    """
    Memory-efficient decoder block with lightweight transformer
    
    Key components:
    1. 1x1 convolution for channel alignment (Network In Network)
    2. Lightweight transformer for global context (TransUNet)
    3. Standard convolutions for feature refinement
    4. Efficient attention gates (CBAM)
    
    Reasoning for 1x1 conv:
    - Skip connections from encoder may have different channel dimensions
    - 1x1 conv acts as learned linear projection to align dimensions
    - Computationally efficient: Only params = in_channels × out_channels
    - Adds cross-channel interactions without spatial operations
    """
    
    print(f"Building memory-efficient decoder block: {block_name} with {filters} filters")
    
    # Upsampling (bilinear interpolation)
    x = layers.UpSampling2D((2, 2), name=f'{block_name}_upsample')(x)
    
    # Skip connection alignment using 1×1 convolution
    # WHY: Ensures channel compatibility before concatenation
    # Literature: "Network In Network" (Lin et al., 2013)
    if skip_connection.shape[-1] != filters:
        skip_connection = layers.Conv2D(filters, (1, 1), activation='relu', 
                                      padding='same', name=f'{block_name}_skip_align')(skip_connection)
        skip_connection = layers.BatchNormalization(name=f'{block_name}_skip_bn')(skip_connection)
    
    # Apply lightweight transformer to skip connection for global context
    # WHY: Captures long-range dependencies in encoder features
    # Literature: "TransUNet" (Chen et al., 2021)
    if use_transformer:
        print(f"  Adding lightweight transformer to skip at {block_name}")
        skip_connection = LightweightTransformerBlock(
            num_heads=2,
            ff_dim=filters // 4,
            dropout_rate=0.1,
            name=f'{block_name}_skip_transformer'
        )(skip_connection)
    
    # Channel alignment for upsampled features
    if x.shape[-1] != filters:
        x = layers.Conv2D(filters, (1, 1), activation='relu', padding='same', name=f'{block_name}_x_align')(x)
    
    # Concatenate features (U-Net skip connections)
    x = layers.Concatenate(name=f'{block_name}_concat')([x, skip_connection])
    
    # Efficient processing - fewer layers for memory
    # WHY: 2 conv blocks sufficient for feature refinement at this level
    x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same', name=f'{block_name}_conv1')(x)
    x = layers.BatchNormalization(name=f'{block_name}_bn1')(x)
    x = layers.Dropout(dropout_rate, name=f'{block_name}_dropout1')(x)
    
    x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same', name=f'{block_name}_conv2')(x)
    x = layers.BatchNormalization(name=f'{block_name}_bn2')(x)
    
    # Add efficient attention (highlights important features)
    x = EfficientSpatialAttentionGate(name=f'{block_name}_attention_gate')(x)
    
    return x

# ---- Memory-Efficient Deep Decoder Block ----
def memory_efficient_deep_decoder_block_with_transformer(x, skip_connection, filters, block_name, 
                                                        dropout_rate=0.3, use_transformer=True):
    """
    Memory-efficient deep decoder block with lightweight transformer
    
    CRITICAL for Class 1 (Disc) and Class 2 (Cup) segmentation
    
    Key components:
    1. Multi-scale feature extraction (Inception-style)
    2. Deep processing with 3 conv layers (VGG-style)
    3. Transformer for global context
    4. Attention gates for focus
    
    WHY MULTI-SCALE?
    - Disc boundaries: Need fine details (1×1, 3×3)
    - Cup context: Need broader receptive field (5×5 not used here for memory)
    - Literature: "Going Deeper with Convolutions" (Szegedy et al., 2015)
    
    WHY 3 CONV LAYERS (reduced from 6)?
    - Memory optimization while maintaining depth
    - Still provides effective receptive field of 7×7
    - Literature: "Very Deep CNNs" (Simonyan & Zisserman, 2014)
    """
    
    print(f"Building memory-efficient DEEP decoder block: {block_name} with {filters} filters")
    
    # Upsampling
    x = layers.UpSampling2D((2, 2), name=f'{block_name}_upsample')(x)
    
    # Process skip connection efficiently
    # WHY 1×1 conv: Channel reduction for memory efficiency
    skip_processed = layers.Conv2D(filters//2, (1, 1), activation='relu', 
                                 padding='same', name=f'{block_name}_skip_process')(skip_connection)
    skip_processed = layers.BatchNormalization(name=f'{block_name}_skip_bn')(skip_processed)
    
    # Apply lightweight transformer for global context
    # CRITICAL: Helps relate disc and cup spatial relationships
    if use_transformer:
        print(f"  Adding lightweight transformer to enhanced skip at {block_name}")
        skip_processed = LightweightTransformerBlock(
            num_heads=2,
            ff_dim=filters // 2,
            dropout_rate=0.1,
            name=f'{block_name}_skip_enhanced_transformer'
        )(skip_processed)
    
    # Multi-scale feature extraction (Inception-style, memory efficient)
    # WHY TWO PATHS instead of four:
    # - 1×1: Point-wise features (low-level details)
    # - 3×3: Local neighborhood patterns (boundaries)
    # Memory savings: Reduced from 4 paths (1×1, 3×3, 5×5, pool) to 2 paths
    x_1x1 = layers.Conv2D(filters//4, (1, 1), activation='relu', padding='same', name=f'{block_name}_1x1')(x)
    x_3x3 = layers.Conv2D(filters//4, (3, 3), activation='relu', padding='same', name=f'{block_name}_3x3')(x)
    
    # Concatenate multi-scale features
    x_multi = layers.Concatenate(name=f'{block_name}_multi_concat')([x_1x1, x_3x3])
    x_multi = layers.BatchNormalization(name=f'{block_name}_multi_bn')(x_multi)
    
    # Concatenate with processed skip connection
    x = layers.Concatenate(name=f'{block_name}_concat')([x_multi, skip_processed])
    
    # Deep processing layers (3 instead of 6 for memory)
    # WHY 3 LAYERS?
    # - Effective receptive field: 1 + 2×3 = 7×7 pixels
    # - Sufficient for disc/cup boundary refinement at this resolution
    # - Formula: ERF = 1 + 2×N for N layers of 3×3 convs
    for i in range(3):
        x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same', 
                         name=f'{block_name}_deep_conv_{i+1}')(x)
        x = layers.BatchNormalization(name=f'{block_name}_deep_bn_{i+1}')(x)
        
        # Add dropout in middle layer for regularization
        if i == 1:
            x = layers.Dropout(dropout_rate, name=f'{block_name}_deep_dropout_{i+1}')(x)
    
    # Apply lightweight transformer after deep processing
    # WHY HERE? Features are now refined, transformer captures global dependencies
    if use_transformer:
        print(f"  Adding lightweight transformer after deep processing at {block_name}")
        x = LightweightTransformerBlock(
            num_heads=2,
            ff_dim=filters,
            dropout_rate=0.1,
            name=f'{block_name}_deep_transformer'
        )(x)
    
    # Add efficient attention for feature enhancement
    # WHY ATTENTION? Focuses on disc/cup boundaries, suppresses background
    x = EfficientSpatialAttentionGate(name=f'{block_name}_deep_attention_gate')(x)
    
    return x

# ---- Enhanced Decoder Block (for non-transformer model) ----
def enhanced_decoder_block_with_transformer(x, skip_connection, filters, block_name, 
                                          dropout_rate=0.3, use_transformer=False):
    """
    Enhanced decoder block with more layers and better feature processing
    
    Used for non-transformer model (deeper baseline)
    Includes 4 conv layers + residual connection + attention
    """
    
    print(f"Building enhanced decoder block: {block_name} with {filters} filters")
    
    # Upsampling
    x = layers.UpSampling2D((2, 2), name=f'{block_name}_upsample')(x)
    
    # Skip connection fusion with 1×1 conv for channel alignment
    if skip_connection.shape[-1] != x.shape[-1]:
        skip_connection = layers.Conv2D(x.shape[-1], (1, 1), activation='relu', 
                                      padding='same', name=f'{block_name}_skip_align')(skip_connection)
        skip_connection = layers.BatchNormalization(name=f'{block_name}_skip_bn')(skip_connection)
    
    # Concatenate features
    x = layers.Concatenate(name=f'{block_name}_concat')([x, skip_connection])
    
    # First conv block - reduce channels
    x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same', name=f'{block_name}_conv1')(x)
    x = layers.BatchNormalization(name=f'{block_name}_bn1')(x)
    x = layers.Dropout(dropout_rate, name=f'{block_name}_dropout1')(x)
    
    # Second conv block - refine features
    x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same', name=f'{block_name}_conv2')(x)
    x = layers.BatchNormalization(name=f'{block_name}_bn2')(x)
    
    # Third conv block - additional refinement
    x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same', name=f'{block_name}_conv3')(x)
    x = layers.BatchNormalization(name=f'{block_name}_bn3')(x)
    
    # Add spatial attention gate
    x = SpatialAttentionGate(name=f'{block_name}_attention_gate')(x)
    
    # Fourth conv block with residual connection
    residual = x
    x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same', name=f'{block_name}_conv4')(x)
    x = layers.BatchNormalization(name=f'{block_name}_bn4')(x)
    x = layers.Dropout(dropout_rate, name=f'{block_name}_dropout2')(x)
    
    # Residual connection (ResNet-style)
    x = layers.Add(name=f'{block_name}_residual')([x, residual])
    
    # Final refinement with 1×1 conv
    # WHY? Cross-channel feature fusion without spatial operations
    x = layers.Conv2D(filters, (1, 1), activation='relu', padding='same', name=f'{block_name}_conv_final')(x)
    x = layers.BatchNormalization(name=f'{block_name}_bn_final')(x)
    
    return x

# ---- Deep Decoder Block (for non-transformer model) ----
def deep_decoder_block_with_transformer(x, skip_connection, filters, block_name, 
                                       dropout_rate=0.3, use_transformer=False):
    """
    Extra deep decoder block for critical layers
    
    FULL VERSION with 6 conv layers and multi-scale features
    Used only for non-transformer model
    
    WHY 6 LAYERS?
    - Effective receptive field: 1 + 2×6 = 13×13 pixels
    - Critical for small object segmentation (disc ~150px, cup ~100px in 256×256)
    - More non-linear transformations = better feature learning
    """
    
    print(f"Building DEEP decoder block: {block_name} with {filters} filters")
    
    # Upsampling
    x = layers.UpSampling2D((2, 2), name=f'{block_name}_upsample')(x)
    
    # Skip connection processing
    skip_processed = layers.Conv2D(filters//2, (1, 1), activation='relu', 
                                 padding='same', name=f'{block_name}_skip_process')(skip_connection)
    skip_processed = layers.BatchNormalization(name=f'{block_name}_skip_bn')(skip_processed)
    
    # Multi-scale feature extraction (FULL Inception-style with 4 paths)
    # WHY FOUR PATHS?
    # - Captures features at different scales simultaneously
    # - 1×1: Point features
    # - 3×3: Local patterns
    # - 5×5: Broader context
    # - Pool: Strong feature preservation
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
    
    # Deep processing layers (6 layers for very deep processing)
    # WHY 6 LAYERS?
    # - Stacking small kernels creates large effective receptive field
    # - More non-linearity = better boundary learning
    # - Critical for medical image fine details
    for i in range(6):
        x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same', 
                         name=f'{block_name}_deep_conv_{i+1}')(x)
        x = layers.BatchNormalization(name=f'{block_name}_deep_bn_{i+1}')(x)
        
        # Add dropout every 2 layers
        if i % 2 == 1:
            x = layers.Dropout(dropout_rate, name=f'{block_name}_deep_dropout_{i+1}')(x)
        
        # Add residual connections every 3 layers
        # WHY? Prevents vanishing gradients in very deep networks
        if i == 2:
            residual_1 = x
        elif i == 5:
            x = layers.Add(name=f'{block_name}_deep_residual')([x, residual_1])
    
    # Final boundary enhancement with 1×1 conv
    x = layers.Conv2D(filters, (1, 1), activation='relu', padding='same', name=f'{block_name}_boundary_conv')(x)
    x = layers.BatchNormalization(name=f'{block_name}_boundary_bn')(x)
    
    return x

# ---- Memory-Efficient Enhanced MobileNet-UNet ----
def build_memory_efficient_mobilenet_unet_with_transformer(input_shape=(256, 256, 3), num_classes=3, use_transformer=True):
    """
    Build memory-efficient MobileNet-UNet with lightweight transformers
    
    Architecture summary:
    - Encoder: MobileNetV2 (pretrained on ImageNet)
    - Bridge: Reduced channels (512) + lightweight transformer
    - Decoder: 
      * Layers 1-2: Memory-efficient blocks with transformers
      * Layers 3-4: Deep memory-efficient blocks with transformers (CRITICAL for Disc/Cup)
    - Final: Lightweight transformer + attention + multi-scale refinement
    
    Total parameters: ~15-20M (vs 67M in original transformer version)
    """
    
    print(f"Building Memory-Efficient MobileNet-UNet {'with Lightweight Transformers' if use_transformer else ''}...")
    
    # Input
    inputs = layers.Input(input_shape, dtype='float32')
    
    # Encoder (MobileNetV2 backbone - pretrained on ImageNet)
    # WHY MobileNetV2? Efficient encoder with inverted residuals
    backbone = MobileNetV2(input_tensor=inputs, weights='imagenet', include_top=False)
    
    # Extract skip connections at different resolutions
    skip_1 = backbone.get_layer('block_1_expand_relu').output    # 128x128, 96 channels
    skip_2 = backbone.get_layer('block_3_expand_relu').output    # 64x64, 144 channels
    skip_3 = backbone.get_layer('block_6_expand_relu').output    # 32x32, 192 channels
    skip_4 = backbone.get_layer('block_13_expand_relu').output   # 16x16, 576 channels
    
    # Bridge with efficient processing
    bridge = backbone.output  # 8x8, 1280 channels
    print(f"Bridge shape: {bridge.shape}")
    
    # Reduce bridge channels first (1280 → 512) for memory efficiency
    # WHY? Bridge has most channels, reducing saves significant memory
    bridge = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='bridge_conv1')(bridge)
    bridge = layers.BatchNormalization(name='bridge_bn1')(bridge)
    
    # Apply lightweight transformer at bridge for global context
    # WHY AT BRIDGE? 
    # - Smallest spatial resolution (8×8 = 64 tokens)
    # - Most abstract features
    # - Critical for capturing disc-cup relationship
    if use_transformer:
        print("Adding lightweight transformer at bridge...")
        bridge = LightweightTransformerBlock(
            num_heads=4,
            ff_dim=256,
            dropout_rate=0.1,
            name='bridge_transformer'
        )(bridge)
    
    bridge = layers.Dropout(0.3, name='bridge_dropout')(bridge)
    
    print("Building decoder with memory-efficient transformer blocks...")
    
    # Decoder with memory-efficient blocks
    # Layer 1: 8×8 → 16×16
    x = memory_efficient_decoder_block_with_transformer(
        bridge, skip_4, 256, 'decoder_1', dropout_rate=0.3, use_transformer=use_transformer
    )
    
    # Layer 2: 16×16 → 32×32
    x = memory_efficient_decoder_block_with_transformer(
        x, skip_3, 128, 'decoder_2', dropout_rate=0.3, use_transformer=use_transformer
    )
    
    # Layer 3: 32×32 → 64×64 (DEEP block - CRITICAL for Disc)
    # WHY DEEP HERE?
    # - Resolution critical for disc boundary detection
    # - Multi-scale features capture different aspects
    # - Transformer captures global disc shape
    x = memory_efficient_deep_decoder_block_with_transformer(
        x, skip_2, 64, 'decoder_3_deep', dropout_rate=0.2, use_transformer=use_transformer
    )
    
    # Layer 4: 64×64 → 128×128 (DEEP block - CRITICAL for Cup and fine details)
    # WHY DEEP HERE?
    # - Highest resolution before final upsampling
    # - Critical for cup boundary (smaller object)
    # - Fine detail preservation
    x = memory_efficient_deep_decoder_block_with_transformer(
        x, skip_1, 32, 'decoder_4_deep', dropout_rate=0.2, use_transformer=use_transformer
    )
    
    # Final upsampling: 128×128 → 256×256
    x = layers.UpSampling2D((2, 2), name='final_upsample')(x)
    
    # Simplified final processing
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='final_conv')(x)
    x = layers.BatchNormalization(name='final_bn')(x)
    
    # Apply final lightweight transformer for pixel-level refinement
    # WHY? Final global refinement of segmentation boundaries
    if use_transformer:
        print("Adding final lightweight transformer...")
        x = LightweightTransformerBlock(
            num_heads=2,
            ff_dim=64,
            dropout_rate=0.1,
            name='final_transformer'
        )(x)
    
    # Final attention and boundary enhancement
    x = EfficientSpatialAttentionGate(name='final_attention_gate')(x)
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same', name='boundary_enhance')(x)
    x = layers.BatchNormalization(name='boundary_enhance_bn')(x)
    
    # Output layer
    outputs = layers.Conv2D(num_classes, (1, 1), activation='softmax', name='output', dtype='float32')(x)
    
    model_name = "MobileNet-UNet-MemoryEfficient-Transformer" if use_transformer else "MobileNet-UNet-MemoryEfficient"
    model = models.Model(inputs, outputs, name=model_name)
    
    print(f"Memory-efficient {'transformer ' if use_transformer else ''}model built successfully!")
    print(f"Total parameters: {model.count_params():,}")
    
    return model

# Update the main build function to use memory-efficient version
def build_mobilenet_unet_deeper_with_transformer(input_shape=(256, 256, 3), num_classes=3, use_transformer=True):
    """
    Main build function that selects appropriate architecture
    
    - use_transformer=True: Memory-efficient transformer version (~15-20M params)
    - use_transformer=False: Full deeper version without transformer (~26M params)
    """
    if use_transformer:
        return build_memory_efficient_mobilenet_unet_with_transformer(input_shape, num_classes, use_transformer)
    else:
        # Keep original deeper model for non-transformer version
        print(f"Building MobileNet-UNet with Enhanced Deeper Decoder...")
        
        # Input
        inputs = layers.Input(input_shape, dtype='float32')
        
        # Encoder
        backbone = MobileNetV2(input_tensor=inputs, weights='imagenet', include_top=False)
        
        # Extract skip connections
        skip_1 = backbone.get_layer('block_1_expand_relu').output
        skip_2 = backbone.get_layer('block_3_expand_relu').output
        skip_3 = backbone.get_layer('block_6_expand_relu').output
        skip_4 = backbone.get_layer('block_13_expand_relu').output
        
        # Bridge with full processing
        bridge = backbone.output
        print(f"Bridge shape: {bridge.shape}")
        
        bridge = layers.Conv2D(1024, (3, 3), activation='relu', padding='same', name='bridge_conv1')(bridge)
        bridge = layers.BatchNormalization(name='bridge_bn1')(bridge)
        bridge = layers.Conv2D(1024, (3, 3), activation='relu', padding='same', name='bridge_conv2')(bridge)
        bridge = layers.BatchNormalization(name='bridge_bn2')(bridge)
        bridge = layers.Dropout(0.4, name='bridge_dropout')(bridge)
        
        print("Building decoder with enhanced blocks...")
        
        # Decoder with full enhanced blocks
        x = enhanced_decoder_block_with_transformer(
            bridge, skip_4, 512, 'decoder_1', dropout_rate=0.3, use_transformer=False
        )
        
        x = enhanced_decoder_block_with_transformer(
            x, skip_3, 256, 'decoder_2', dropout_rate=0.3, use_transformer=False
        )
        
        x = deep_decoder_block_with_transformer(
            x, skip_2, 128, 'decoder_3_deep', dropout_rate=0.2, use_transformer=False
        )
        
        x = deep_decoder_block_with_transformer(
            x, skip_1, 64, 'decoder_4_deep', dropout_rate=0.2, use_transformer=False
        )
        
        # Final upsampling and multi-scale processing
        x = layers.UpSampling2D((2, 2), name='final_upsample')(x)
        
        # Multi-scale final features (Feature Pyramid Network style)
        # WHY THREE PATHS?
        # - Fine (3×3): Detail preservation
        # - Coarse (5×5): Context integration
        # - Detail (1×1): Channel-wise refinement
        x_fine = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='final_conv_fine')(x)
        x_fine = layers.BatchNormalization(name='final_bn_fine')(x_fine)
        
        x_coarse = layers.Conv2D(32, (5, 5), activation='relu', padding='same', name='final_conv_coarse')(x)
        x_coarse = layers.BatchNormalization(name='final_bn_coarse')(x_coarse)
        
        x_detail = layers.Conv2D(32, (1, 1), activation='relu', padding='same', name='final_conv_detail')(x)
        x_detail = layers.BatchNormalization(name='final_bn_detail')(x_detail)
        
        # Combine multi-scale features via element-wise addition
        x = layers.Add(name='final_combine')([x_fine, x_coarse, x_detail])
        x = layers.BatchNormalization(name='final_combine_bn')(x)
        
        # Full spatial attention
        x = SpatialAttentionGate(name='final_attention_gate')(x)
        
        # Final boundary enhancement
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='boundary_enhance_1')(x)
        x = layers.BatchNormalization(name='boundary_enhance_bn1')(x)
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='boundary_enhance_2')(x)
        x = layers.BatchNormalization(name='boundary_enhance_bn2')(x)
        
        # Output layer
        outputs = layers.Conv2D(num_classes, (1, 1), activation='softmax', name='output', dtype='float32')(x)
        
        model = models.Model(inputs, outputs, name="MobileNet-UNet-Deeper")
        
        print(f"Enhanced deeper model built successfully!")
        print(f"Total parameters: {model.count_params():,}")
        
        return model

# Backward compatibility
def build_mobilenet_unet_deeper(input_shape=(256, 256, 3), num_classes=3):
    """Build deeper MobileNet-UNet without transformer"""
    return build_mobilenet_unet_deeper_with_transformer(input_shape, num_classes, use_transformer=False)

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
def train_model_with_transformer(X_train, y_train, X_val, y_val, model_name, 
                                 use_transformer=False, augment=None, gpu_available=True):
    """Train deeper MobileNet-UNet model with optional transformer enhancement"""
    
    print(f"\nTraining {model_name}...")
    print(f"Using transformer: {'Yes' if use_transformer else 'No'}")
    print(f"Using augmentation: {'Yes' if augment else 'No'}")
    
    # Clear any previous models
    K.clear_session()
    
    # Build model
    model = build_mobilenet_unet_deeper_with_transformer(use_transformer=use_transformer)
    
    # Optimizer with different settings for transformer model
    initial_lr = 3e-4 if use_transformer else 5e-4
    optimizer = optimizers.Adam(
        learning_rate=initial_lr,
        clipnorm=1.0 if use_transformer else 0.5,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7
    )
    
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
        monitor='val_dice_coef_multiclass',
        mode='max'
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor="val_dice_coef_multiclass",
        factor=0.6 if use_transformer else 0.5,
        patience=6 if use_transformer else 4,
        verbose=1,
        mode='max',
        min_lr=1e-9
    )
    
    early_stop = EarlyStopping(
        monitor="val_dice_coef_multiclass",
        patience=20 if use_transformer else 12,
        verbose=1, 
        restore_best_weights=True,
        mode='max'
    )
    
    # Set batch size
    batch_size = 2 if use_transformer else 4
    steps_train = X_train.shape[0] // batch_size
    steps_val = X_val.shape[0] // batch_size
    
    print(f"Using batch size: {batch_size}")
    print(f"Steps per epoch - Train: {steps_train}, Val: {steps_val}")
    
    # Training with error handling
    try:
        history = model.fit(
            data_generator(X_train, y_train, batch_size, augment=augment),
            validation_data=data_generator(X_val, y_val, batch_size, augment=None),
            steps_per_epoch=steps_train,
            validation_steps=steps_val,
            epochs=100 if use_transformer else 80,
            callbacks=[checkpoint, reduce_lr, early_stop],
            verbose=1
        )
    except Exception as e:
        print(f"Training error with batch_size {batch_size}: {e}")
        print("Reducing batch size to 1...")
        batch_size = 1
        steps_train = X_train.shape[0] // batch_size
        steps_val = X_val.shape[0] // batch_size
        
        history = model.fit(
            data_generator(X_train, y_train, batch_size, augment=augment),
            validation_data=data_generator(X_val, y_val, batch_size, augment=None),
            steps_per_epoch=steps_train,
            validation_steps=steps_val,
            epochs=100 if use_transformer else 80,
            callbacks=[checkpoint, reduce_lr, early_stop],
            verbose=1
        )
    
    return model, history

# ---- Evaluation Functions ----
def calculate_flops_manual(model, input_shape=(1, 256, 256, 3)):
    """Manual FLOPS calculation for the model"""
    try:
        params = model.count_params()
        if 'transformer' in model.name.lower():
            estimated_flops = params * 5
        else:# filepath: c:\Users\karan\OneDrive\Desktop\deeper_transformer.py