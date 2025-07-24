import math
import torch
from torch import nn
import torch.nn.functional as F
from functools import partial
import copy
from .utils import register_model

def exists(x):
    return x is not None

def default(val, d):
    return val if exists(val) else (d() if callable(d) else d)

def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

def Upsample(dim, dim_out=None, scale_factor=2):
    return nn.Sequential(
        nn.Upsample(scale_factor=scale_factor, mode='nearest'),
        nn.Conv1d(dim, default(dim_out, dim), 3, padding=1)
    )

def Final_upsample_to_target_length(dim_out, dim_in, target_length):
    return nn.Sequential(
        nn.Upsample(size=target_length, mode='nearest'),
        nn.Conv1d(dim_out, dim_in, 3, padding=1)
    )

def Downsample(dim, dim_out=None, kernel=4):
    return nn.Conv1d(dim, default(dim_out, dim), kernel, 2, 1)

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1))
    def forward(self, x):
        return F.normalize(x, dim=1) * self.g * (x.shape[1] ** 0.5)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = RMSNorm(dim)
    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        self.dim = dim
        self.theta = theta
    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    def __init__(self, dim, is_random=False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad=not is_random)
    def forward(self, x):
        x = x.unsqueeze(1)
        freqs = x * self.weights.unsqueeze(0) * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv1d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()
    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)
        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, classes_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(int(time_emb_dim) + int(classes_emb_dim), dim_out * 2)
        ) if exists(time_emb_dim) or exists(classes_emb_dim) else None
        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()
    def forward(self, x, time_emb=None, class_emb=None):
        scale_shift = None
        if exists(self.mlp) and (exists(time_emb) or exists(class_emb)):
            cond_emb = tuple(filter(exists, (time_emb, class_emb)))
            cond_emb = torch.cat(cond_emb, dim=-1)
            cond_emb = self.mlp(cond_emb)
            cond_emb = cond_emb.unsqueeze(-1)
            scale_shift = cond_emb.chunk(2, dim=1)
        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)
    def forward(self, x):
        b, c, n = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: t.reshape(b, self.heads, -1, n), qkv)
        # q, k, v: [b, heads, dim_head, n]
        q = q * self.scale
        sim = torch.einsum('bhcn,bhcm->bhnm', q, k)  # [b, heads, n, n]
        attn = sim.softmax(dim=-1)
        out = torch.einsum('bhnm,bhcm->bhcn', attn, v)
        out = out.reshape(b, -1, n)
        return self.to_out(out)

class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)
    def forward(self, x):
        b, c, n = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: t.reshape(b, self.heads, -1, n), qkv)
        q = q * self.scale
        sim = torch.einsum('bhcn,bhcm->bhnm', q, k)  # [b, heads, n, n]
        attn = sim.softmax(dim=-1)
        out = torch.einsum('bhnm,bhcm->bhcn', attn, v)
        out = out.reshape(b, -1, n)
        return self.to_out(out)

@register_model(name="unet1d")
def Unet1D_from_config(config):
    model_cfg = config.model
    return Unet1D(
        dim=model_cfg.dim,
        class_dim=model_cfg.class_dim,
        seq_length=model_cfg.seq_length,
        cond_drop_prob=getattr(model_cfg, 'cond_drop_prob', 0.5),
        mask_val=getattr(model_cfg, 'mask_val', 0.0),
        init_dim=getattr(model_cfg, 'init_dim', None),
        out_dim=getattr(model_cfg, 'out_dim', None),
        dim_mults=getattr(model_cfg, 'dim_mults', (1, 2, 4, 8)),
        embed_class_layers_dims=getattr(model_cfg, 'embed_class_layers_dims', (64, 64)),
        channels=getattr(model_cfg, 'channels', 1),
        self_condition=getattr(model_cfg, 'self_condition', False),
        resnet_block_groups=getattr(model_cfg, 'resnet_block_groups', 4),
        learned_variance=getattr(model_cfg, 'learned_variance', False),
        learned_sinusoidal_cond=getattr(model_cfg, 'learned_sinusoidal_cond', False),
        random_fourier_features=getattr(model_cfg, 'random_fourier_features', False),
        learned_sinusoidal_dim=getattr(model_cfg, 'learned_sinusoidal_dim', 16),
        sinusoidal_pos_emb_theta=getattr(model_cfg, 'sinusoidal_pos_emb_theta', 10000),
        attn_dim_head=getattr(model_cfg, 'attn_dim_head', 32),
        attn_heads=getattr(model_cfg, 'attn_heads', 4),
    )

class Unet1D(nn.Module):
    def __init__(
            self,
            dim,
            class_dim,
            seq_length,
            cond_drop_prob=0.5,
            mask_val=0.0,
            init_dim=None,
            out_dim=None,
            dim_mults=(1, 2, 4, 8),
            embed_class_layers_dims=(64, 64),
            channels=1,
            self_condition=False,
            resnet_block_groups=4,
            learned_variance=False,
            learned_sinusoidal_cond=False,
            random_fourier_features=False,
            learned_sinusoidal_dim=16,
            sinusoidal_pos_emb_theta=10000,
            attn_dim_head=32,
            attn_heads=4,
    ):
        super().__init__()
        self.cond_drop_prob = cond_drop_prob
        self.mask_val = mask_val
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
        block_klass = partial(ResnetBlock, groups=resnet_block_groups)
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
        self.classes_mlp = self.build_classes_mlp(class_dim, self.embed_class_layers_dims)
        embedded_classes_dim = self.embed_class_layers_dims[-1]
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            if self.seq_length > 1:
                self.downs.append(nn.ModuleList([
                    block_klass(dim_in, dim_in, time_emb_dim=time_dim, classes_emb_dim=embedded_classes_dim),
                    block_klass(dim_in, dim_in, time_emb_dim=time_dim, classes_emb_dim=embedded_classes_dim),
                    Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                    Downsample(dim_in, dim_out) if not is_last else nn.Conv1d(dim_in, dim_out, 3, padding=1)
                ]))
            else:
                self.downs.append(nn.ModuleList([
                    block_klass(dim_in, dim_in, time_emb_dim=time_dim, classes_emb_dim=embedded_classes_dim),
                    block_klass(dim_in, dim_in, time_emb_dim=time_dim, classes_emb_dim=embedded_classes_dim),
                    Residual(PreNorm(dim_in, LinearAttention(dim_in))),
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
                    Final_upsample_to_target_length(dim_out, dim_in, target_length=int(self.seq_length/2)) if is_third_last else (Final_upsample_to_target_length(dim_out, dim_in, target_length=self.seq_length) if is_second_last else (nn.Conv1d(dim_out, dim_in, 3, padding=1) if is_last else Upsample(dim_out, dim_in)))
                ]))
            else:
                self.ups.append(nn.ModuleList([
                    block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim, classes_emb_dim=embedded_classes_dim),
                    block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim, classes_emb_dim=embedded_classes_dim),
                    Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                    (nn.Conv1d(dim_out, dim_in, 3, padding=1) if is_last else Upsample(dim_out, dim_in, 1))
                ]))
        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)
        self.final_res_block = block_klass(self.dim * 2, self.dim, time_emb_dim=time_dim, classes_emb_dim=embedded_classes_dim)
        self.final_conv = nn.Conv1d(self.dim, self.out_dim, 1)
    def build_classes_mlp(self, class_dim, embed_class_layers_dims):
        layers = []
        input_dim = copy.copy(class_dim)
        for output_dim in embed_class_layers_dims:
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.GELU())
            input_dim = output_dim
        layers.pop()
        return nn.Sequential(*layers)
    def forward_with_cond_scale(self, *args, cond_scale=1., rescaled_phi=0., **kwargs):
        logits = self.forward(*args, cond_drop_prob=0., **kwargs)
        if cond_scale == 1:
            return logits
        null_logits = self.forward(*args, cond_drop_prob=1., **kwargs)
        scaled_logits = null_logits + (logits - null_logits) * cond_scale
        if rescaled_phi == 0.:
            return scaled_logits
        std_fn = partial(torch.std, dim=tuple(range(1, scaled_logits.ndim)), keepdim=True)
        rescaled_logits = scaled_logits * (std_fn(logits) / (std_fn(scaled_logits) + 1e-6))
        if torch.isnan(rescaled_logits).any():
            return scaled_logits
        return rescaled_logits * rescaled_phi + scaled_logits * (1. - rescaled_phi)
    def forward(self, x, time, class_labels=None, cond_drop_prob=None):
        batch, device = x.shape[0], x.device
        cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)
        if cond_drop_prob > 0:
            keep_mask = prob_mask_like((batch,), 1 - cond_drop_prob, device=device)
            classes_emb = torch.where(
                keep_mask.unsqueeze(1),
                class_labels,
                torch.full_like(class_labels, self.mask_val, device=device)
            )
            c = self.classes_mlp(classes_emb)
        else:
            c = self.classes_mlp(class_labels)
        x = self.init_conv(x)
        r = x.clone()
        t = self.time_mlp(time)
        h = []
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t, c)
            h.append(x)
            x = block2(x, t, c)
            x = attn(x)
            h.append(x)
            x = downsample(x)
        x = self.mid_block1(x, t, c)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t, c)
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t, c)
            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t, c)
            x = attn(x)
            x = upsample(x)
        x = torch.cat((x, r), dim=1)
        x = self.final_res_block(x, t, c)
        return self.final_conv(x) 