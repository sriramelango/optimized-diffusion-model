# =============================================================================
# IMPORTS
# =============================================================================

# Standard library imports
import math
import copy
from functools import partial

# Third-party imports
import torch
from torch import nn, einsum, Tensor
import torch.nn.functional as F
from einops import rearrange
from models.utils import register_model

# =============================================================================
# CONSTANTS AND CONFIGURATION
# =============================================================================

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def exists(x):
    """
    Check if a value exists (is not None).
    
    Args:
        x: Value to check
        
    Returns:
        bool: True if x is not None, False otherwise
    """
    return x is not None


def default(val, d):
    """
    Return the first value if it exists, otherwise return the default value.
    
    Args:
        val: Primary value to check
        d: Default value or callable that returns the default value
        
    Returns:
        The primary value if it exists, otherwise the default value
    """
    if exists(val):
        return val
    return d() if callable(d) else d


def prob_mask_like(shape, prob, device):
    """
    Create a boolean mask tensor with specified probability of True values.
    
    Args:
        shape: Shape of the mask tensor
        prob: Probability of True values (between 0 and 1)
        device: Device to place the tensor on
        
    Returns:
        Tensor: Boolean mask tensor
    """
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob

# =============================================================================
# SMALL HELPER MODULES
# =============================================================================

class Residual(nn.Module):
    """
    Residual connection wrapper.
    
    This module adds a residual connection to any function f, such that
    the output is f(x) + x. This is commonly used in neural networks
    to help with gradient flow and training stability.
    """
    
    def __init__(self, fn):
        """
        Initialize the residual wrapper.
        
        Args:
            fn: Function to wrap with residual connection
        """
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        """
        Forward pass with residual connection.
        
        Args:
            x: Input tensor
            *args: Additional positional arguments for fn
            **kwargs: Additional keyword arguments for fn
            
        Returns:
            Tensor: f(x) + x
        """
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim, dim_out=None, scale_factor=2):
    """
    Create an upsampling module using nearest neighbor interpolation and convolution.
    
    Args:
        dim: Input dimension
        dim_out: Output dimension (defaults to dim if None)
        scale_factor: Upsampling factor (default: 2)
        
    Returns:
        nn.Sequential: Upsampling module
    """
    return nn.Sequential(
        nn.Upsample(scale_factor=scale_factor, mode='nearest'),
        nn.Conv1d(dim, default(dim_out, dim), 3, padding=1)
    )


def Final_upsample_to_target_length(dim_out, dim_in, target_length):
    """
    Create a final upsampling module to reach a specific target length.
    
    Args:
        dim_out: Output dimension
        dim_in: Input dimension
        target_length: Target sequence length
        
    Returns:
        nn.Sequential: Upsampling module to target length
    """
    return nn.Sequential(
        nn.Upsample(size=target_length, mode='nearest'),
        nn.Conv1d(dim_out, dim_in, 3, padding=1)
    )


def Downsample(dim, dim_out=None, kernel=4):
    """
    Create a downsampling module using strided convolution.
    
    Args:
        dim: Input dimension
        dim_out: Output dimension (defaults to dim if None)
        kernel: Convolution kernel size (default: 4)
        
    Returns:
        nn.Conv1d: Downsampling convolution layer
    """
    return nn.Conv1d(dim, default(dim_out, dim), kernel, 2, 1)


class RMSNorm(nn.Module):
    """
    Root Mean Square normalization layer.
    
    This normalization technique normalizes the input using the RMS of the
    feature dimension, which can be more stable than standard normalization.
    """
    
    def __init__(self, dim):
        """
        Initialize RMS normalization.
        
        Args:
            dim: Feature dimension to normalize over
        """
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1))

    def forward(self, x):
        """
        Apply RMS normalization.
        
        Args:
            x: Input tensor of shape (batch, features, length)
            
        Returns:
            Tensor: Normalized tensor
        """
        return F.normalize(x, dim=1) * self.g * (x.shape[1] ** 0.5)


class PreNorm(nn.Module):
    """
    Pre-normalization wrapper.
    
    This module applies normalization before passing the input to a function,
    which can improve training stability and convergence.
    """
    
    def __init__(self, dim, fn):
        """
        Initialize pre-normalization wrapper.
        
        Args:
            dim: Feature dimension for normalization
            fn: Function to apply after normalization
        """
        super().__init__()
        self.fn = fn
        self.norm = RMSNorm(dim)

    def forward(self, x):
        """
        Apply normalization followed by the wrapped function.
        
        Args:
            x: Input tensor
            
        Returns:
            Tensor: Output of fn(normalize(x))
        """
        x = self.norm(x)
        return self.fn(x)

# =============================================================================
# SINUSOIDAL POSITIONAL EMBEDDINGS
# =============================================================================

class SinusoidalPosEmb(nn.Module):
    """
    Sinusoidal positional embeddings for sequence modeling.
    
    This module generates sinusoidal positional embeddings that can be added
    to input sequences to provide position information to the model.
    """
    
    def __init__(self, dim, theta=10000):
        """
        Initialize sinusoidal positional embeddings.
        
        Args:
            dim: Dimension of the positional embeddings
            theta: Frequency parameter for the sinusoids (default: 10000)
        """
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        """
        Generate sinusoidal positional embeddings.
        
        Args:
            x: Input tensor of shape (batch_size, 1) containing positions
            
        Returns:
            Tensor: Positional embeddings of shape (batch_size, dim)
        """
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """
    Random or learned sinusoidal positional embeddings.
    
    This module provides an alternative to fixed sinusoidal embeddings by
    allowing the positional encoding to be either random or learned during
    training. Based on the implementation by @crowsonkb.
    """
    
    def __init__(self, dim, is_random=False):
        """
        Initialize random or learned sinusoidal positional embeddings.
        
        Args:
            dim: Dimension of the positional embeddings (must be even)
            is_random: If True, use random weights; if False, learn weights
        """
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad=not is_random)

    def forward(self, x):
        """
        Generate random or learned sinusoidal positional embeddings.
        
        Args:
            x: Input tensor of shape (batch_size,) containing positions
            
        Returns:
            Tensor: Positional embeddings of shape (batch_size, dim + 1)
        """
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered

# =============================================================================
# BUILDING BLOCK MODULES
# =============================================================================

class Block(nn.Module):
    """
    Basic building block with convolution, normalization, and activation.
    
    This module implements a standard neural network block consisting of
    a convolution layer, group normalization, and SiLU activation.
    """
    
    def __init__(self, dim, dim_out, groups=8):
        """
        Initialize the block.
        
        Args:
            dim: Input dimension
            dim_out: Output dimension
            groups: Number of groups for group normalization (default: 8)
        """
        super().__init__()
        self.proj = nn.Conv1d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        """
        Forward pass through the block.
        
        Args:
            x: Input tensor
            scale_shift: Optional scale and shift parameters for conditional normalization
            
        Returns:
            Tensor: Output after convolution, normalization, and activation
        """
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    """
    Residual block with time and class conditioning.
    
    This module implements a residual block that can be conditioned on both
    time embeddings and class embeddings, commonly used in diffusion models.
    """
    
    def __init__(self, dim, dim_out, *, time_emb_dim=None, classes_emb_dim=None, groups=8):
        """
        Initialize the residual block.
        
        Args:
            dim: Input dimension
            dim_out: Output dimension
            time_emb_dim: Dimension of time embeddings (optional)
            classes_emb_dim: Dimension of class embeddings (optional)
            groups: Number of groups for group normalization (default: 8)
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(int(time_emb_dim) + int(classes_emb_dim), dim_out * 2)
        ) if exists(time_emb_dim) or exists(classes_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None, class_emb=None):
        """
        Forward pass through the residual block.
        
        Args:
            x: Input tensor
            time_emb: Time embeddings (optional)
            class_emb: Class embeddings (optional)
            
        Returns:
            Tensor: Output after residual block processing
        """
        scale_shift = None

        if exists(self.mlp) and (exists(time_emb) or exists(class_emb)):
            cond_emb = tuple(filter(exists, (time_emb, class_emb)))
            cond_emb = torch.cat(cond_emb, dim=-1)
            try:
                cond_emb = self.mlp(cond_emb)
            except:
                print("wrong size for conditional embedding!")
            cond_emb = rearrange(cond_emb, 'b c -> b c 1')

            scale_shift = cond_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)

        return h + self.res_conv(x)

class LinearAttention(nn.Module):
    """
    Linear attention mechanism for efficient sequence processing.
    
    This module implements a linear attention mechanism that reduces the
    computational complexity from O(n²) to O(n) for sequence length n.
    """
    
    def __init__(self, dim, heads=4, dim_head=32):
        """
        Initialize linear attention.
        
        Args:
            dim: Input dimension
            heads: Number of attention heads (default: 4)
            dim_head: Dimension per attention head (default: 32)
        """
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Sequential(
            nn.Conv1d(hidden_dim, dim, 1),
            RMSNorm(dim)
        )

    def forward(self, x):
        """
        Apply linear attention.
        
        Args:
            x: Input tensor of shape (batch, dim, seq_len)
            
        Returns:
            Tensor: Output after linear attention
        """
        b, c, n = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h d) n -> b h d n', h=self.heads), qkv)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h e n -> b (h e) n')
        return self.to_out(out)

class Attention(nn.Module):
    """
    Standard multi-head attention mechanism.
    
    This module implements the standard scaled dot-product attention
    mechanism used in transformer architectures.
    """
    
    def __init__(self, dim, heads=4, dim_head=32):
        """
        Initialize multi-head attention.
        
        Args:
            dim: Input dimension
            heads: Number of attention heads (default: 4)
            dim_head: Dimension per attention head (default: 32)
        """
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
        """
        Apply multi-head attention.
        
        Args:
            x: Input tensor of shape (batch, dim, seq_len)
            
        Returns:
            Tensor: Output after attention
        """
        b, c, n = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h d) n -> b h d n', h=self.heads), qkv)

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b h d j -> b h d i', attn, v)

        out = rearrange(out, 'b h d n -> b (h d) n')
        return self.to_out(out)

# =============================================================================
# MODEL
# =============================================================================

@register_model(name='unet1d_gto')
class Unet1D(nn.Module):
    """
    1D U-Net architecture for diffusion models with classifier-free guidance.
    
    This module implements a 1D U-Net architecture that can be conditioned on
    class embeddings and supports classifier-free guidance for improved generation
    quality. The architecture follows the standard U-Net pattern with skip
    connections and attention mechanisms.
    """
    
    def __init__(self, config):
        """Initialize Unet1D from config object to match other models."""
        # Extract parameters from config
        dim = config.model.dim
        class_dim = config.model.class_dim
        seq_length = config.model.seq_length
        cond_drop_prob = getattr(config.model, 'cond_drop_prob', 0.5)
        mask_val = getattr(config.model, 'mask_val', 0.0)
        init_dim = getattr(config.model, 'init_dim', None)
        out_dim = getattr(config.model, 'out_dim', None)
        dim_mults = tuple(getattr(config.model, 'dim_mults', [1, 2, 4, 8]))
        embed_class_layers_dims = tuple(getattr(config.model, 'embed_class_layers_dims', [64, 64]))
        channels = getattr(config.model, 'channels', 3)
        self_condition = getattr(config.model, 'self_condition', False)
        resnet_block_groups = getattr(config.model, 'resnet_block_groups', 4)
        learned_variance = getattr(config.model, 'learned_variance', False)
        learned_sinusoidal_cond = getattr(config.model, 'learned_sinusoidal_cond', False)
        random_fourier_features = getattr(config.model, 'random_fourier_features', False)
        learned_sinusoidal_dim = getattr(config.model, 'learned_sinusoidal_dim', 16)
        sinusoidal_pos_emb_theta = getattr(config.model, 'sinusoidal_pos_emb_theta', 10000)
        attn_dim_head = getattr(config.model, 'attn_dim_head', 32)
        attn_heads = getattr(config.model, 'attn_heads', 4)
        
        super().__init__()
        """
        Initialize the 1D U-Net architecture.
        
        Args:
            dim: Base dimension for the network
            class_dim: Dimension of class embeddings
            seq_length: Length of input sequences
            cond_drop_prob: Probability of dropping conditioning during training
            mask_val: Value to use for masked conditioning
            init_dim: Initial dimension (defaults to dim)
            out_dim: Output dimension (defaults to channels)
            dim_mults: Multipliers for dimensions at each level
            embed_class_layers_dims: Dimensions for class embedding MLP
            channels: Number of input/output channels
            self_condition: Whether to use self-conditioning
            resnet_block_groups: Number of groups for ResNet blocks
            learned_variance: Whether to learn the variance
            learned_sinusoidal_cond: Whether to use learned sinusoidal conditioning
            random_fourier_features: Whether to use random Fourier features
            learned_sinusoidal_dim: Dimension for learned sinusoidal embeddings
            sinusoidal_pos_emb_theta: Theta parameter for sinusoidal embeddings
            attn_dim_head: Dimension per attention head
            attn_heads: Number of attention heads
        """
        super().__init__()

        # classifier free guidance stuff
        self.cond_drop_prob = cond_drop_prob
        self.mask_val = mask_val

        # determine dimensions
        self.dim = dim
        self.dim_mults = dim_mults
        self.embed_class_layers_dims = embed_class_layers_dims

        self.seq_length = seq_length

        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, self.dim)
        self.init_conv = nn.Conv1d(input_channels, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: self.dim * m, self.dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # Partial function is used to create a "partial" version of a function. You can think of it as pre-setting some arguments of a function.
        # Here, the ResnetBlock class is being partially set with the groups argument pre-defined. Any subsequent calls to block_klass will be the same as calling ResnetBlock with groups=resnet_block_groups.
        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings
        time_dim = self.dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(self.dim, theta=sinusoidal_pos_emb_theta)
            fourier_dim = self.dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # Class embeddings ##, define the conditional embedding 
        # embedded_classes_dim = self.dim * 4
        # self.classes_mlp = nn.Sequential(
        #     nn.Linear(class_dim, embedded_classes_dim),
        #     nn.GELU(),
        #     nn.Linear(embedded_classes_dim, embedded_classes_dim)
        # )

        self.classes_mlp = self.build_classes_mlp(class_dim, self.embed_class_layers_dims)
        embedded_classes_dim = self.embed_class_layers_dims[-1]

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            # Handle the case where seq_length = 1, no upsampling and downsampling, only conv1d
            if self.seq_length > 1:
                self.downs.append(nn.ModuleList([
                    block_klass(dim_in, dim_in, time_emb_dim=time_dim, classes_emb_dim=embedded_classes_dim),
                    block_klass(dim_in, dim_in, time_emb_dim=time_dim, classes_emb_dim=embedded_classes_dim),
                    Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                    Downsample(dim_in, dim_out) if not is_last else nn.Conv1d(dim_in, dim_out, 3, padding=1)
                ]))
            else:
                # If sequence length is already 1, avoid downsampling and just use convolutional layers
                self.downs.append(nn.ModuleList([
                    block_klass(dim_in, dim_in, time_emb_dim=time_dim, classes_emb_dim=embedded_classes_dim),
                    block_klass(dim_in, dim_in, time_emb_dim=time_dim, classes_emb_dim=embedded_classes_dim),
                    Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                    # nn.Conv1d(dim_in, dim_out, 3, padding=1)
                    Downsample(dim_in, dim_out, 2) if not is_last else nn.Conv1d(dim_in, dim_out, 3, padding=1)
                ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim, classes_emb_dim=embedded_classes_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim, dim_head=attn_dim_head, heads=attn_heads)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim, classes_emb_dim=embedded_classes_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            is_second_last = ind == (len(in_out) - 2)
            is_third_last = ind == (len(in_out) - 3)

            if self.seq_length > 1:
                self.ups.append(nn.ModuleList([
                    block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim, classes_emb_dim=embedded_classes_dim),
                    block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim, classes_emb_dim=embedded_classes_dim),
                    Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                    # Since the original data size not necessarily can be divided by two,
                    #  so if is the second last year of upsample, we need make sure it upsamples to the original size
                    # Upsample(dim_out, dim_in) if not is_last else nn.Conv1d(dim_out, dim_in, 3, padding=1),
                    # Jannik: Added first upsample to half of the sequence length for input of dimension 66 => 33 => 16 => 33 => 66
                    Final_upsample_to_target_length(dim_out, dim_in,
                                                    target_length=int(self.seq_length/2)) if is_third_last else
                    (Final_upsample_to_target_length(dim_out, dim_in,
                                                    target_length=self.seq_length) if is_second_last else
                    (nn.Conv1d(dim_out, dim_in, 3, padding=1) if is_last else
                     Upsample(dim_out, dim_in)))

                ]))
            else:
                self.ups.append(nn.ModuleList([
                    block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim, classes_emb_dim=embedded_classes_dim),
                    block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim, classes_emb_dim=embedded_classes_dim),
                    Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                    # nn.Conv1d(dim_out, dim_in, 3, padding=1),
                    (nn.Conv1d(dim_out, dim_in, 3, padding=1) if is_last else
                     Upsample(dim_out, dim_in, 1))
                ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = block_klass(self.dim * 2, self.dim, time_emb_dim=time_dim,
                                           classes_emb_dim=embedded_classes_dim)
        self.final_conv = nn.Conv1d(self.dim, self.out_dim, 1)

    def build_classes_mlp(self, class_dim, embed_class_layers_dims):
        """
        Build a multi-layer perceptron (MLP) for class embeddings.
        
        This function constructs a neural network with linear layers and GELU activations
        to transform class embeddings from the input dimension to the specified output dimensions.
        The final activation layer is removed to return raw logits.
        
        Args:
            class_dim (int): Input dimension of the class embeddings
            embed_class_layers_dims (list): List of output dimensions for each layer
            
        Returns:
            nn.Sequential: A sequential neural network for class embedding transformation
        """
        layers = []
        input_dim = copy.copy(class_dim)
        
        # Build layers with linear transformations and GELU activations
        for output_dim in embed_class_layers_dims:
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.GELU())
            input_dim = output_dim

        # Remove the last GELU activation to return raw logits
        layers.pop()

        return nn.Sequential(*layers)

    def forward_with_cond_scale(
            self,
            *args,
            cond_scale=1.,
            rescaled_phi=0.,
            **kwargs
    ):
        """
        Forward pass with classifier-free guidance scaling.
        
        This method implements classifier-free guidance by computing both conditional and unconditional
        outputs, then combining them using a scaling factor. It also supports rescaling based on
        standard deviation to maintain statistical properties.
        
        Args:
            *args: Positional arguments passed to the forward method
            cond_scale (float): Scaling factor for classifier-free guidance. 
                               When cond_scale=1, returns fully conditional output.
                               When cond_scale>1, increases the influence of conditioning.
            rescaled_phi (float): Weight for rescaling based on standard deviation.
                                 When rescaled_phi=0, no rescaling is applied.
                                 When rescaled_phi=1, fully uses rescaled output.
            **kwargs: Keyword arguments passed to the forward method
            
        Returns:
            torch.Tensor: Scaled and potentially rescaled logits
        """
        # Get fully conditional output (no dropout applied to conditioning)
        logits = self.forward(*args, cond_drop_prob=0., **kwargs)

        # If cond_scale is 1, return the conditional output directly
        if cond_scale == 1:
            return logits

        # Get unconditional output (full dropout applied to conditioning)
        null_logits = self.forward(*args, cond_drop_prob=1., **kwargs)

        # Apply classifier-free guidance scaling:
        # scaled_logits = unconditional + scale * (conditional - unconditional)
        # This is equivalent to: scale * conditional + (1 - scale) * unconditional
        scaled_logits = null_logits + (logits - null_logits) * cond_scale

        # If no rescaling is requested, return the scaled logits
        if rescaled_phi == 0.:
            return scaled_logits

        # Apply rescaling based on standard deviation to maintain statistical properties
        # This helps preserve the variance of the original conditional output
        std_fn = partial(torch.std, dim=tuple(range(1, scaled_logits.ndim)), keepdim=True)
        rescaled_logits = scaled_logits * (std_fn(logits) / (std_fn(scaled_logits) + 1e-6))
        
        # Safety check: if rescaling produces NaN values, fall back to scaled logits
        if torch.isnan(rescaled_logits).any():
            return scaled_logits

        # Combine rescaled and original scaled logits using rescaled_phi weight
        # If rescaled_phi = 1.0, fully use rescaled output
        # If rescaled_phi = 0.0, fully use original scaled output
        return rescaled_logits * rescaled_phi + scaled_logits * (1. - rescaled_phi)

    def forward(self, x, time, class_labels=None, cond_drop_prob=None):
        """
        Forward pass through the U-Net diffusion model with conditional guidance.
        
        This method implements the main forward pass of the diffusion model, processing
        the input through a U-Net architecture with time embeddings and conditional
        class information. It supports classifier-free guidance through conditional dropout.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, sequence_length)
            time (torch.Tensor): Time step tensor of shape (batch_size,)
            class_labels (torch.Tensor): Class labels tensor of shape (batch_size, num_classes)
            cond_drop_prob (float, optional): Conditional dropout probability for classifier-free guidance.
                                            If None, uses the default value from model initialization.
        
        Returns:
            torch.Tensor: Predicted noise or denoised output of same shape as input
        """
        # Extract batch size and device from input tensor
        batch, device = x.shape[0], x.device
        
        # Use default conditional dropout probability if not provided
        cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)

        # =============================================================================
        # CONDITIONAL GUIDANCE SETUP
        # =============================================================================
        
        # Handle class conditioning with classifier-free guidance
        if class_labels is not None:
            # Apply conditional dropout for classifier-free guidance
            if cond_drop_prob > 0:
                # Create mask to randomly drop conditioning information
                keep_mask = prob_mask_like((batch,), 1 - cond_drop_prob, device=device)
                
                # Apply mask to class embeddings: keep original class_labels or use mask value
                classes_emb = torch.where(
                    rearrange(keep_mask, 'b -> b 1'),
                    class_labels,
                    torch.tensor(self.mask_val, device=class_labels.device)  # Use mask value for dropped conditioning
                )
                
                # Embed class information into conditional variable c
                c = self.classes_mlp(classes_emb)
            else:
                # No dropout: use all class information
                c = self.classes_mlp(class_labels)
        else:
            # No class conditioning: use mask value as default
            default_class = torch.full((batch, 1), self.mask_val, device=device)
            c = self.classes_mlp(default_class)

        # =============================================================================
        # U-NET ENCODING PATH (DOWNWARD)
        # =============================================================================
        
        # Initial convolution layer
        x = self.init_conv(x)
        r = x.clone()  # Store residual connection for final skip connection
        
        # Generate time embeddings
        t = self.time_mlp(time)
        
        # Store intermediate features for skip connections
        h = []
        
        # Process through downsampling blocks
        for block1, block2, attn, downsample in self.downs:
            # First residual block with time and conditional embeddings
            x = block1(x, t, c)
            h.append(x)  # Store for skip connection
            
            # Second residual block with time and conditional embeddings
            x = block2(x, t, c)
            x = attn(x)  # Apply attention mechanism
            h.append(x)  # Store for skip connection
            
            # Downsample feature maps
            x = downsample(x)

        # =============================================================================
        # U-NET BOTTLENECK (MIDDLE)
        # =============================================================================
        
        # Middle residual blocks with attention
        x = self.mid_block1(x, t, c)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t, c)

        # =============================================================================
        # U-NET DECODING PATH (UPWARD)
        # =============================================================================
        
        # Process through upsampling blocks
        for block1, block2, attn, upsample in self.ups:
            # Concatenate with skip connection from encoding path
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t, c)  # First residual block
            
            # Concatenate with another skip connection
            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t, c)  # Second residual block
            x = attn(x)          # Apply attention mechanism
            x = upsample(x)      # Upsample feature maps
        
        # =============================================================================
        # FINAL OUTPUT
        # =============================================================================
        
        # Add final residual connection from input
        x = torch.cat((x, r), dim=1)
        
        # Final residual block and output convolution
        x = self.final_res_block(x, t, c)
        return self.final_conv(x)
