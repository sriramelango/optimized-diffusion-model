# pylint: skip-file

from . import utils, layers, layerspp, normalization
import torch.nn as nn
import functools
import torch
import numpy as np
import torch.nn.functional as F

ResnetBlockDDPM = layerspp.ResnetBlockDDPMpp
ResnetBlockBigGAN = layerspp.ResnetBlockBigGANpp
Combine = layerspp.Combine
conv3x3 = layerspp.conv3x3
conv1x1 = layerspp.conv1x1
get_act = layers.get_act
get_normalization = normalization.get_normalization
default_initializer = layers.default_init


@utils.register_model(name='ncsnpp')
class NCSNpp(nn.Module):
    """NCSN++ model"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.nf = nf = config.model.nf
        self.ch_mult = ch_mult = config.model.ch_mult
        self.num_res_blocks = num_res_blocks = config.model.num_res_blocks
        self.attn_resolutions = attn_resolutions = config.model.attn_resolutions
        self.dropout = dropout = config.model.dropout
        self.resamp_with_conv = resamp_with_conv = config.model.resamp_with_conv
        self.embedding_type = embedding_type = config.model.embedding_type
        self.conditional = conditional = config.model.conditional
        self.cond_drop_prob = config.model.cond_drop_prob if hasattr(config.model, 'cond_drop_prob') else 0.0
        self.num_classes = getattr(config.model, 'num_classes', 1)
        self.init_scale = config.model.init_scale
        self.skip_rescale = config.model.skip_rescale
        self.fir = config.model.fir
        self.fir_kernel = tuple(config.model.fir_kernel)
        self.image_size = config.model.image_size
        self.image_width = getattr(config.model, 'image_width', config.model.image_size)
        self.channels = config.model.channels
        self.scale_by_sigma = getattr(config.model, 'scale_by_sigma', False)
        self.act = get_act(config)

        # Time embedding
        if self.embedding_type == 'fourier':
            self.time_embed = layerspp.GaussianFourierProjection(embedding_size=nf, scale=config.model.fourier_scale)
            embed_dim = 2 * nf
        else:
            raise NotImplementedError('Only fourier embedding supported')
        self.time_mlp = nn.Sequential(
            nn.Linear(embed_dim, nf * 4),
            self.act,
            nn.Linear(nf * 4, nf * 4),
        )

        # Classifier-free guidance
        if conditional:
            self.label_emb = nn.Linear(self.num_classes, nf * 4)

        # Input conv
        self.input_conv = conv3x3(self.channels, nf)

        # Downsampling
        self.down_blocks = nn.ModuleList()
        self.down_attn = nn.ModuleList()
        self.downsample = nn.ModuleList()
        in_ch = nf
        self.resolutions = [self.image_size]
        
        # Build skip connections exactly matching up blocks
        self.skip_channels = []
        for i, mult in enumerate(ch_mult):
            out_ch = nf * mult
            for j in range(num_res_blocks):
                self.down_blocks.append(ResnetBlockDDPM(self.act, in_ch, out_ch, temb_dim=nf*4, dropout=dropout, skip_rescale=self.skip_rescale, init_scale=self.init_scale))
                in_ch = out_ch
                if self.image_size // (2 ** i) in attn_resolutions:
                    self.down_attn.append(layerspp.AttnBlockpp(in_ch, skip_rescale=self.skip_rescale, init_scale=self.init_scale))
                else:
                    self.down_attn.append(None)
                # Store skip connection for each down block
                self.skip_channels.append(in_ch)
            
            # Store skip connection for the extra up block at this resolution
            self.skip_channels.append(in_ch)
            
            if i != len(ch_mult) - 1:
                self.downsample.append(layerspp.Downsample(in_ch, with_conv=resamp_with_conv, fir=self.fir, fir_kernel=self.fir_kernel))
                self.resolutions.append(self.resolutions[-1] // 2)
            else:
                self.downsample.append(None)
        
        # Verify skip connection count matches up blocks
        total_up_blocks = sum([num_res_blocks + 1 for _ in ch_mult])
        print(f"[DEBUG] skip_channels list: {self.skip_channels}")
        print(f"[DEBUG] Number of skip connections: {len(self.skip_channels)}")
        print(f"[DEBUG] Number of up blocks: {total_up_blocks}")
        assert len(self.skip_channels) == total_up_blocks, f"Skip connections ({len(self.skip_channels)}) must match up blocks ({total_up_blocks})"

        # Bottleneck
        self.mid_block1 = ResnetBlockDDPM(self.act, in_ch, in_ch, temb_dim=nf*4, dropout=dropout, skip_rescale=self.skip_rescale, init_scale=self.init_scale)
        self.mid_attn = layerspp.AttnBlockpp(in_ch, skip_rescale=self.skip_rescale, init_scale=self.init_scale) if self.image_size // (2 ** (len(ch_mult)-1)) in attn_resolutions else None
        self.mid_block2 = ResnetBlockDDPM(self.act, in_ch, in_ch, temb_dim=nf*4, dropout=dropout, skip_rescale=self.skip_rescale, init_scale=self.init_scale)

        # Upsampling
        self.up_blocks = nn.ModuleList()
        self.up_attn = nn.ModuleList()
        self.upsample = nn.ModuleList()
        skip_channels = list(reversed(self.skip_channels))
        
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = nf * mult
            for _ in range(num_res_blocks + 1):
                skip_ch = skip_channels.pop(0)
                # The input channels should be in_ch + skip_ch
                input_ch = in_ch + skip_ch
                self.up_blocks.append(ResnetBlockDDPM(self.act, input_ch, out_ch, temb_dim=nf*4, dropout=dropout, skip_rescale=self.skip_rescale, init_scale=self.init_scale))
                in_ch = out_ch
                if self.image_size // (2 ** i) in attn_resolutions:
                    self.up_attn.append(layerspp.AttnBlockpp(in_ch, skip_rescale=self.skip_rescale, init_scale=self.init_scale))
                else:
                    self.up_attn.append(None)
            
            if i != 0:
                self.upsample.append(layerspp.Upsample(in_ch, with_conv=resamp_with_conv, fir=self.fir, fir_kernel=self.fir_kernel))
            else:
                self.upsample.append(None)

        # Output
        self.out_norm = nn.GroupNorm(num_groups=min(in_ch // 4, 32), num_channels=in_ch, eps=1e-6)
        self.out_act = self.act
        self.out_conv = conv3x3(in_ch, self.channels, init_scale=self.init_scale)

    def forward(self, x, time_cond, class_labels=None):
        # Classifier-free guidance
        if self.conditional and self.training and self.cond_drop_prob > 0:
            mask = (torch.rand(x.shape[0], device=x.device) < self.cond_drop_prob).float().unsqueeze(1)
            class_labels = class_labels * (1 - mask)
        # Time embedding
        if self.embedding_type == 'fourier':
            temb = self.time_embed(torch.log(time_cond))
        else:
            raise NotImplementedError('Only fourier embedding supported')
        temb = self.time_mlp(temb)
        # Class conditioning
        if self.conditional:
            temb = temb + self.label_emb(class_labels)
        # Input
        h = self.input_conv(x)
        hs = [h]
        # Downsampling
        d_idx = 0
        for i in range(len(self.ch_mult)):
            for _ in range(self.num_res_blocks):
                h = self.down_blocks[d_idx](h, temb)
                if self.down_attn[d_idx] is not None:
                    h = self.down_attn[d_idx](h)
                hs.append(h)
                d_idx += 1
            # Add one more skip connection per resolution for the extra up block
            hs.append(h)
            if self.downsample[i] is not None:
                h = self.downsample[i](h)
        # Bottleneck
        h = self.mid_block1(h, temb)
        if self.mid_attn is not None:
            h = self.mid_attn(h)
        h = self.mid_block2(h, temb)
        # Upsampling
        u_idx = 0
        skip_channels = list(reversed(self.skip_channels))
        for i in range(len(self.ch_mult)):
            for _ in range(self.num_res_blocks + 1):
                h_skip = hs.pop()
                # Ensure spatial shapes match before concatenation
                if h.shape[2:] != h_skip.shape[2:]:
                    h = F.interpolate(h, size=h_skip.shape[2:], mode='nearest')
                skip_ch = skip_channels.pop(0)
                h = torch.cat([h, h_skip], dim=1)
                h = self.up_blocks[u_idx](h, temb)
                if self.up_attn[u_idx] is not None:
                    h = self.up_attn[u_idx](h)
                u_idx += 1
            if self.upsample[i] is not None:
                h = self.upsample[i](h)
        h = self.out_norm(h)
        h = self.out_act(h)
        h = self.out_conv(h)
        if self.scale_by_sigma:
            h = h / time_cond.view(x.shape[0], *([1] * (h.ndim - 1)))
        return h
