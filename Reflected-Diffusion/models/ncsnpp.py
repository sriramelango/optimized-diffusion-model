# pylint: skip-file

# Import required modules for the NCSN++ model
from . import utils, layers, layerspp, normalization
import torch.nn as nn
import functools
import torch
import numpy as np
import torch.nn.functional as F

# Import specific layer types and utilities
ResnetBlockDDPM = layerspp.ResnetBlockDDPMpp
ResnetBlockBigGAN = layerspp.ResnetBlockBigGANpp
Combine = layerspp.Combine
conv3x3 = layerspp.conv3x3
conv1x1 = layerspp.conv1x1
get_act = layers.get_act
get_normalization = normalization.get_normalization
default_initializer = layers.default_init


# Register this model with the model registry
@utils.register_model(name='ncsnpp')
class NCSNpp(nn.Module):
    """
    NCSN++ (Noise Conditional Score Networks++) model for diffusion-based generation.
    
    This is a U-Net style architecture with:
    - Downsampling path: Extracts features at multiple resolutions
    - Bottleneck: Processes features at the lowest resolution
    - Upsampling path: Reconstructs features with skip connections
    - Time conditioning: Embeds noise level information
    - Class conditioning: Optional classifier-free guidance support
    
    Key architectural features:
    - Skip connections between corresponding down/up blocks
    - Attention blocks at specified resolutions
    - Fourier time embedding
    - Conditional generation support
    """

    def __init__(self, config):
        # Initialize the parent nn.Module class
        super().__init__()
        
        # Store the configuration for later use
        self.config = config
        
        # Extract all model configuration parameters from config
        # Base number of features (channels) for the model
        self.nf = nf = config.model.nf
        # Channel multipliers for each resolution level (e.g., [1, 2, 4, 8])
        self.ch_mult = ch_mult = config.model.ch_mult
        # Number of ResNet blocks per resolution level
        self.num_res_blocks = num_res_blocks = config.model.num_res_blocks
        # List of resolutions where attention blocks should be added
        self.attn_resolutions = attn_resolutions = config.model.attn_resolutions
        # Dropout rate for regularization
        self.dropout = dropout = config.model.dropout
        # Whether to use convolution for resampling operations
        self.resamp_with_conv = resamp_with_conv = config.model.resamp_with_conv
        # Type of time embedding (only 'fourier' is supported)
        self.embedding_type = embedding_type = config.model.embedding_type
        # Whether to enable class conditioning
        self.conditional = conditional = config.model.conditional
        # Probability of dropping class labels during training (for classifier-free guidance)
        self.cond_drop_prob = config.model.cond_drop_prob if hasattr(config.model, 'cond_drop_prob') else 0.0
        # Number of classes for conditioning
        self.num_classes = getattr(config.model, 'num_classes', 1)
        # Scale factor for weight initialization
        self.init_scale = config.model.init_scale
        # Whether to rescale skip connections
        self.skip_rescale = config.model.skip_rescale
        # Whether to use finite impulse response filter
        self.fir = config.model.fir
        # Kernel size for FIR filter
        self.fir_kernel = tuple(config.model.fir_kernel)
        # Input image size (assumed square)
        self.image_size = config.model.image_size
        # Image width (for non-square images, defaults to image_size)
        self.image_width = getattr(config.model, 'image_width', config.model.image_size)
        # Number of input channels
        self.channels = config.model.channels
        # Whether to scale output by noise level
        self.scale_by_sigma = getattr(config.model, 'scale_by_sigma', False)
        # Activation function to use throughout the model
        self.act = get_act(config)

        # ===== TIME EMBEDDING SETUP =====
        # Create time embedding layer to encode noise level information
        if self.embedding_type == 'fourier':
            # Use Fourier features for time embedding (more stable than positional encoding)
            self.time_embed = layerspp.GaussianFourierProjection(embedding_size=nf, scale=config.model.fourier_scale)
            # Fourier features have 2x the embedding size (real and imaginary parts)
            embed_dim = 2 * nf
        else:
            # Only Fourier embedding is currently supported
            raise NotImplementedError('Only fourier embedding supported')
        
        # MLP to process the time embeddings
        # Expand to 4x base features for rich time representation
        self.time_mlp = nn.Sequential(
            nn.Linear(embed_dim, nf * 4),
            self.act,
            nn.Linear(nf * 4, nf * 4),
        )

        # ===== CLASSIFIER-FREE GUIDANCE SETUP =====
        # Create class label embedding for conditional generation
        if conditional:
            # Embed class labels to the same dimension as time embeddings
            self.label_emb = nn.Linear(self.num_classes, nf * 4)

        # ===== INPUT PROCESSING =====
        # Initial convolution to process input images
        # Converts input channels to base number of features
        self.input_conv = conv3x3(self.channels, nf)

        # ===== DOWNSAMPLING PATH CONSTRUCTION =====
        # Create lists to store downsampling components
        self.down_blocks = nn.ModuleList()  # ResNet blocks for feature extraction
        self.down_attn = nn.ModuleList()    # Attention blocks for downsampling
        self.downsample = nn.ModuleList()   # Downsampling operations
        in_ch = nf  # Start with base number of features
        self.resolutions = [self.image_size]  # Track resolution changes
        
        # ===== SKIP CONNECTION TRACKING =====
        # Build skip connections that will be used in the upsampling path
        # Each down block produces a skip connection for the corresponding up block
        self.skip_channels = []
        
        # Iterate through each resolution level
        for i, mult in enumerate(ch_mult):
            # Calculate number of features at this resolution
            out_ch = nf * mult
            
            # Add ResNet blocks for this resolution level
            for j in range(num_res_blocks):
                # Create ResNet block with time conditioning
                self.down_blocks.append(ResnetBlockDDPM(self.act, in_ch, out_ch, temb_dim=nf*4, dropout=dropout, skip_rescale=self.skip_rescale, init_scale=self.init_scale))
                in_ch = out_ch
                
                # Add attention block if this resolution requires it
                if self.image_size // (2 ** i) in attn_resolutions:
                    self.down_attn.append(layerspp.AttnBlockpp(in_ch, skip_rescale=self.skip_rescale, init_scale=self.init_scale))
                else:
                    self.down_attn.append(None)
                
                # Store skip connection channel count for this block
                self.skip_channels.append(in_ch)
            
            # Store skip connection for the extra up block at this resolution
            # (There's one more up block than down blocks per resolution)
            self.skip_channels.append(in_ch)
            
            # Add downsampling operation (except for the last resolution)
            if i != len(ch_mult) - 1:
                self.downsample.append(layerspp.Downsample(in_ch, with_conv=resamp_with_conv, fir=self.fir, fir_kernel=self.fir_kernel))
                self.resolutions.append(self.resolutions[-1] // 2)  # Halve resolution
            else:
                self.downsample.append(None)
        
        # ===== VERIFY SKIP CONNECTION COUNT =====
        # Ensure we have the right number of skip connections for the upsampling path
        # Calculate total number of up blocks (num_res_blocks + 1 per resolution)
        total_up_blocks = sum([num_res_blocks + 1 for _ in ch_mult])
        print(f"[DEBUG] skip_channels list: {self.skip_channels}")
        print(f"[DEBUG] Number of skip connections: {len(self.skip_channels)}")
        print(f"[DEBUG] Number of up blocks: {total_up_blocks}")
        # Verify that skip connections match up blocks exactly
        assert len(self.skip_channels) == total_up_blocks, f"Skip connections ({len(self.skip_channels)}) must match up blocks ({total_up_blocks})"

        # ===== BOTTLENECK CONSTRUCTION =====
        # Process features at the lowest resolution (bottleneck)
        # First bottleneck ResNet block
        self.mid_block1 = ResnetBlockDDPM(self.act, in_ch, in_ch, temb_dim=nf*4, dropout=dropout, skip_rescale=self.skip_rescale, init_scale=self.init_scale)
        # Attention block at bottleneck (if required)
        self.mid_attn = layerspp.AttnBlockpp(in_ch, skip_rescale=self.skip_rescale, init_scale=self.init_scale) if self.image_size // (2 ** (len(ch_mult)-1)) in attn_resolutions else None
        # Second bottleneck ResNet block
        self.mid_block2 = ResnetBlockDDPM(self.act, in_ch, in_ch, temb_dim=nf*4, dropout=dropout, skip_rescale=self.skip_rescale, init_scale=self.init_scale)

        # ===== UPSAMPLING PATH CONSTRUCTION =====
        # Build the decoder (upsampling) path with skip connections
        self.up_blocks = nn.ModuleList()  # ResNet blocks for upsampling
        self.up_attn = nn.ModuleList()    # Attention blocks for upsampling
        self.upsample = nn.ModuleList()   # Upsampling operations
        # Reverse skip channels for upsampling order (last in, first out)
        skip_channels = list(reversed(self.skip_channels))
        
        # Iterate through resolutions in reverse order (from lowest to highest)
        for i, mult in reversed(list(enumerate(ch_mult))):
            # Calculate number of features at this resolution
            out_ch = nf * mult
            
            # Add ResNet blocks for this resolution level (one extra per resolution)
            for _ in range(num_res_blocks + 1):
                # Get skip connection channel count
                skip_ch = skip_channels.pop(0)
                # Input channels = current channels + skip connection channels
                input_ch = in_ch + skip_ch
                # Create ResNet block that processes concatenated features
                self.up_blocks.append(ResnetBlockDDPM(self.act, input_ch, out_ch, temb_dim=nf*4, dropout=dropout, skip_rescale=self.skip_rescale, init_scale=self.init_scale))
                in_ch = out_ch
                
                # Add attention block if this resolution requires it
                if self.image_size // (2 ** i) in attn_resolutions:
                    self.up_attn.append(layerspp.AttnBlockpp(in_ch, skip_rescale=self.skip_rescale, init_scale=self.init_scale))
                else:
                    self.up_attn.append(None)
            
            # Add upsampling operation (except for the first resolution)
            if i != 0:
                self.upsample.append(layerspp.Upsample(in_ch, with_conv=resamp_with_conv, fir=self.fir, fir_kernel=self.fir_kernel))
            else:
                self.upsample.append(None)

        # ===== OUTPUT LAYERS =====
        # Final processing layers to produce output
        # Group normalization for final features
        self.out_norm = nn.GroupNorm(num_groups=min(in_ch // 4, 32), num_channels=in_ch, eps=1e-6)
        # Activation function
        self.out_act = self.act
        # Final convolution to produce output with same channels as input
        self.out_conv = conv3x3(in_ch, self.channels, init_scale=self.init_scale)

    def forward(self, x, time_cond, class_labels=None):
        """
        Forward pass through the NCSN++ model.
        
        Args:
            x: Input tensor of shape (B, C, H, W) - noisy image or score
            time_cond: Time/noise level tensor of shape (B,) - noise level
            class_labels: Optional class labels for conditioning of shape (B, num_classes)
        
        Returns:
            Predicted noise or score tensor of same shape as input
        """
        
        # ===== CLASSIFIER-FREE GUIDANCE =====
        # During training, randomly drop class labels to enable classifier-free guidance
        # This allows the model to learn both conditional and unconditional generation
        if self.conditional and self.training and self.cond_drop_prob > 0:
            # Create random mask for dropping class labels
            mask = (torch.rand(x.shape[0], device=x.device) < self.cond_drop_prob).float().unsqueeze(1)
            # Zero out some class labels based on mask
            class_labels = class_labels * (1 - mask)
        
        # ===== TIME EMBEDDING =====
        # Embed the noise level using Fourier features
        if self.embedding_type == 'fourier':
            # Use log of time condition for numerical stability
            temb = self.time_embed(torch.log(time_cond))
        else:
            raise NotImplementedError('Only fourier embedding supported')
        
        # Process time embeddings through MLP
        temb = self.time_mlp(temb)
        
        # ===== CLASS CONDITIONING =====
        # Add class information to time embeddings if conditioning is enabled
        if self.conditional:
            temb = temb + self.label_emb(class_labels)
        
        # ===== INPUT PROCESSING =====
        # Initial convolution to process input images
        h = self.input_conv(x)
        # Store initial features for skip connection
        hs = [h]
        
        # ===== DOWNSAMPLING PATH =====
        # Process through encoder (downsampling) path
        d_idx = 0  # Index for down blocks
        for i in range(len(self.ch_mult)):
            # Process through ResNet blocks at this resolution
            for _ in range(self.num_res_blocks):
                # Apply ResNet block with time conditioning
                h = self.down_blocks[d_idx](h, temb)
                
                # Apply attention if present at this resolution
                if self.down_attn[d_idx] is not None:
                    h = self.down_attn[d_idx](h)
                
                # Store features for skip connection
                hs.append(h)
                d_idx += 1
            
            # Add one more skip connection per resolution for the extra up block
            hs.append(h)
            
            # Downsample to next resolution (except for the last resolution)
            if self.downsample[i] is not None:
                h = self.downsample[i](h)
        
        # ===== BOTTLENECK =====
        # Process features at the lowest resolution (bottleneck)
        # First bottleneck block
        h = self.mid_block1(h, temb)
        # Apply attention at bottleneck if present
        if self.mid_attn is not None:
            h = self.mid_attn(h)
        # Second bottleneck block
        h = self.mid_block2(h, temb)
        
        # ===== UPSAMPLING PATH =====
        # Process through decoder (upsampling) path with skip connections
        u_idx = 0  # Index for up blocks
        # Reverse skip channels for upsampling order
        skip_channels = list(reversed(self.skip_channels))
        
        # Iterate through resolutions in reverse order
        for i in range(len(self.ch_mult)):
            # Process through ResNet blocks at this resolution (one extra per resolution)
            for _ in range(self.num_res_blocks + 1):
                # Get skip connection from downsampling path
                h_skip = hs.pop()
                
                # Ensure spatial shapes match before concatenation
                # This handles cases where upsampling might not perfectly match downsampling
                if h.shape[2:] != h_skip.shape[2:]:
                    h = F.interpolate(h, size=h_skip.shape[2:], mode='nearest')
                
                # Get skip connection channel count
                skip_ch = skip_channels.pop(0)
                # Concatenate current features with skip connection
                h = torch.cat([h, h_skip], dim=1)
                
                # Apply ResNet block with time conditioning
                h = self.up_blocks[u_idx](h, temb)
                
                # Apply attention if present at this resolution
                if self.up_attn[u_idx] is not None:
                    h = self.up_attn[u_idx](h)
                
                u_idx += 1
            
            # Upsample to next resolution (except for the first resolution)
            if self.upsample[i] is not None:
                h = self.upsample[i](h)
        
        # ===== OUTPUT PROCESSING =====
        # Final normalization, activation, and convolution
        # Apply group normalization
        h = self.out_norm(h)
        # Apply activation function
        h = self.out_act(h)
        # Final convolution to produce output
        h = self.out_conv(h)
        
        # Optionally scale output by noise level (for some diffusion formulations)
        if self.scale_by_sigma:
            h = h / time_cond.view(x.shape[0], *([1] * (h.ndim - 1)))
        
        # Return the processed output
        return h
