#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File   :msvm_unet.py
@Author :CodeCat
@Date   :2025/8/27 11:46
"""
from __future__ import annotations
from collections import OrderedDict
from einops import rearrange
from functools import partial
import torch
from torch import nn
import torch.nn.functional as F
from models.vmamba import ENCODERS, VSSM
from models.vmamba.vmamba import VSSBlock, LayerNorm2d, Linear2d
from typing import List, Any, Sequence, Type, Optional


nonlinearity = partial(F.relu, inplace=True)


class DACblock(nn.Module):
    def __init__(self, channel):
        super(DACblock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=3, padding=3)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=5, padding=5)
        self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.conv1x1(self.dilate2(x)))
        dilate3_out = nonlinearity(self.conv1x1(self.dilate2(self.dilate1(x))))
        dilate4_out = nonlinearity(self.conv1x1(self.dilate3(self.dilate2(self.dilate1(x)))))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
        return out


class SPPblock(nn.Module):
    def __init__(self, in_channels):
        super(SPPblock, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=[3, 3], stride=3)
        self.pool3 = nn.MaxPool2d(kernel_size=[5, 5], stride=5)
        self.pool4 = nn.MaxPool2d(kernel_size=[6, 6], stride=6)

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1, padding=0)

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        layer1 = F.interpolate(self.conv(self.pool1(x)), size=(h, w), mode='bilinear', align_corners=False)
        layer2 = F.interpolate(self.conv(self.pool2(x)), size=(h, w), mode='bilinear', align_corners=False)
        layer3 = F.interpolate(self.conv(self.pool3(x)), size=(h, w), mode='bilinear', align_corners=False)
        layer4 = F.interpolate(self.conv(self.pool4(x)), size=(h, w), mode='bilinear', align_corners=False)

        out = torch.cat([layer1, layer2, layer3, layer4, x], 1)

        return out


class Context_Extractor(nn.Module):
    def __init__(self, in_channels=3):
        super(Context_Extractor, self).__init__()
        self.DACBlock = DACblock(channel=in_channels)
        self.RMPBlock = SPPblock(in_channels=in_channels)

    def forward(self, x):
        x = self.DACBlock(x)
        x = self.RMPBlock(x)
        return x


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            nn.Conv2d(gate_channels, gate_channels // reduction_ratio, 1, padding=0),
            nn.ReLU(),
            nn.Conv2d(gate_channels // reduction_ratio, gate_channels, 1, padding=0),
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                channel_att_raw = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(channel_att_raw)
            elif pool_type == 'max':

                channel_att_raw = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(channel_att_raw)
            elif pool_type == 'lp':
                channel_att_raw = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(channel_att_raw)
            else:
                raise Exception('Unsupported pool type: {}'.format(pool_type))
            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum += channel_att_raw
        scale = torch.sigmoid(channel_att_sum)
        return x * scale


class SpatialGate(nn.Module):
    def __init__(self, ):
        super(SpatialGate, self).__init__()
        self.spatial = layers.ConvBN(
            in_channels=2,
            out_channels=1,
            kernel_size=7,
            padding=3,
            stride=1
        )

    def forward(self, x):
        x_compress = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)
        return x * scale


class Attention(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(Attention, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        x_out = self.SpatialGate(x_out)
        return x_out


class Encoder(nn.Module):
    def __init__(self, name: str, in_channels: int = 3, **kwargs: Any) -> None:
        super(Encoder, self).__init__()
        vss_encoder: VSSM = ENCODERS[name](in_channels=in_channels, **kwargs)
        self.dims = vss_encoder.dims
        self.channel_first = vss_encoder.channel_first
        self.in_channels = in_channels
        self.layer0 = nn.Sequential(
            vss_encoder.patch_embed[0],
            vss_encoder.patch_embed[1],
            vss_encoder.patch_embed[2],
            vss_encoder.patch_embed[3],
            vss_encoder.patch_embed[4],
        )
        self.layer1 = nn.Sequential(
            vss_encoder.patch_embed[5],
            vss_encoder.patch_embed[6],
            vss_encoder.patch_embed[7],
        )
        self.layers = vss_encoder.layers
        self.downsamples = vss_encoder.downsamples

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        if x.shape[1] == 1:
            x = x.repeat(1, self.in_channels, 1, 1)

        ret = []
        x = self.layer0(x)
        x = self.layer1(x)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            ret.append(x if self.channel_first else x.permute(0, 3, 1, 2))
            x = self.downsamples[i](x)
        return ret

    @torch.no_grad()
    def freeze_params(self) -> None:
        for name, param in self.named_parameters():
            param.requires_grad = False

    @torch.no_grad()
    def unfreeze_params(self) -> None:
        for name, param in self.named_parameters():
            param.requires_grad = True


class MSConv(nn.Module):
    def __init__(self, dim: int, kernel_sizes: Sequence[int] = (1, 3, 5)) -> None:
        super(MSConv, self).__init__()
        self.dw_convs = nn.ModuleList([
            nn.Conv2d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim, bias=False)
            for kernel_size in kernel_sizes
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + sum([conv(x) for conv in self.dw_convs])


class MS_MLP(nn.Module):
    def __init__(
            self,
            in_features: int,
            hidden_features: Optional[int] = None,
            out_features: Optional[int] = None,
            act_layer: Type[nn.Module] = nn.GELU,
            drop: float = 0.,
            channels_first: bool = False,
            kernel_sizes: Sequence[int] = (1, 3, 5),
    ) -> None:
        super(MS_MLP, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        Linear = Linear2d if channels_first else nn.Linear

        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.multiscale_conv = MSConv(hidden_features, kernel_sizes=kernel_sizes)
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.multiscale_conv(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MSVSS(nn.Sequential):
    def __init__(
            self,
            dim: int,
            depth: int,
            drop_path: Sequence[float] | float = 0.0,
            use_checkpoint: bool = False,
            norm_layer: Type[nn.Module] = LayerNorm2d,
            channel_first: bool = True,
            ssm_d_state: int = 1,
            ssm_ratio: float = 1.0,
            ssm_dt_rank: str = "auto",
            ssm_act_layer: Type[nn.Module] = nn.SiLU,
            ssm_conv: int = 3,
            ssm_conv_bias: bool = False,
            ssm_drop_rate: float = 0.0,
            ssm_init: str = "v0",
            forward_type: str = "v05_noz",
            mlp_ratio: float = 4.0,
            mlp_act_layer: Type[nn.Module] = nn.GELU,
            mlp_drop_rate: float = 0.0,
            gmlp: bool = False,
    ) -> None:
        blocks = []
        for d in range(depth):
            blocks.append(VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[d] if isinstance(drop_path, Sequence) else drop_path,
                norm_layer=norm_layer,
                channel_first=channel_first,
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                gmlp=gmlp,
                use_checkpoint=use_checkpoint,
                customized_mlp=MS_MLP
            ))
        super(MSVSS, self).__init__(OrderedDict(
            blocks=nn.Sequential(*blocks),
        ))


class LKPE(nn.Module):
    def __init__(self, dim: int, dim_scale: int = 2, norm_layer: Type[nn.Module] = nn.LayerNorm):
        super(LKPE, self).__init__()
        self.dim = dim if dim != 772 else dim - 4
        self.expand = nn.Sequential(
            nn.Conv2d(dim, self.dim * 2, kernel_size=1, bias=True),
            nn.BatchNorm2d(self.dim * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.dim * 2, self.dim * 2, kernel_size=3, padding=1, groups=self.dim * 2, bias=True)
        )
        self.norm = norm_layer(self.dim // dim_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.expand(x)

        x = rearrange(x, pattern="b c h w -> b h w c")
        B, H, W, C = x.shape

        x = x.view(B, H, W, C)
        x = rearrange(x, pattern="b h w (p1 p2 c)-> b (h p1) (w p2) c", p1=2, p2=2, c=C // 4)
        x = x.view(B, -1, C // 4)
        x = self.norm(x)
        x = x.reshape(B, H * 2, W * 2, C // 4)

        x = rearrange(x, pattern="b h w c -> b c h w")
        return x


class FLKPE(nn.Module):
    def __init__(
            self,
            dim: int,
            num_classes: int,
            dim_scale: int = 4,
            norm_layer: Type[nn.Module] = nn.LayerNorm
    ):
        super(FLKPE, self).__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Sequential(
            nn.Conv2d(dim, dim * dim_scale ** 2, kernel_size=1, bias=True),
            nn.BatchNorm2d(dim * dim_scale ** 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim * dim_scale ** 2, dim * dim_scale ** 2, kernel_size=3, padding=1, groups=dim * dim_scale ** 2,
                      bias=True)
        )

        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)
        self.out = nn.Conv2d(self.output_dim, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.expand(x)

        x = rearrange(x, pattern="b c h w -> b h w c")
        B, H, W, C = x.shape

        x = rearrange(x, pattern="b h w (p1 p2 c)-> b (h p1) (w p2) c", p1=self.dim_scale, p2=self.dim_scale,
                      c=C // (self.dim_scale ** 2))
        x = x.view(B, -1, self.output_dim)
        x = self.norm(x)
        x = x.reshape(B, H * self.dim_scale, W * self.dim_scale, self.output_dim)

        x = rearrange(x, pattern="b h w c -> b c h w")
        return self.out(x)


class UpBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            depth: int,
            drop_path: Sequence[float] | float,
    ) -> None:
        super(UpBlock, self).__init__()
        self.up = LKPE(in_channels)
        self.concat_layer = Linear2d(2 * out_channels, out_channels)
        self.vss_layer = MSVSS(dim=out_channels, depth=depth, drop_path=drop_path)

    def forward(self, input: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        out = self.up(input)
        out = torch.cat(tensors=(out, skip), dim=1)
        out = self.concat_layer(out)
        out = self.vss_layer(out)
        return out


class UpAttBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            depth: int,
            drop_path: Sequence[float] | float,
    ) -> None:
        super(UpAttBlock, self).__init__()
        self.up = LKPE(in_channels)
        self.concat_layer = Linear2d(2 * out_channels, out_channels)
        self.vss_layer = MSVSS(dim=out_channels, depth=depth, drop_path=drop_path)
        self.attention = Attention(gate_channels=out_channels)

    def forward(self, input: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        out = self.up(input)
        out = torch.cat(tensors=(out, skip), dim=1)
        out = self.concat_layer(out)
        out = self.vss_layer(out)
        out = self.attention(out)
        return out


class Decoder(nn.Module):
    def __init__(
            self,
            dims: Sequence[int],
            num_classes: int,
            depths: Sequence[int] = (2, 2, 2, 2),
            drop_path_rate: float = 0.2,
    ) -> None:
        super(Decoder, self).__init__()
        dpr = [x.item() for x in torch.linspace(drop_path_rate, 0, (len(dims) - 1) * 2)]

        self.layers = nn.ModuleList()
        for i in range(1, len(dims)):
            self.layers.append(
                UpBlock(
                    in_channels=dims[i - 1],
                    out_channels=dims[i],
                    depth=depths[i],
                    drop_path=dpr[sum(depths[: i - 1]): sum(depths[: i])],
                ))

        self.out_layers = nn.Sequential(FLKPE(dims[-1], num_classes))

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        out = features[0]
        features = features[1:]
        for i, layer in enumerate(self.layers):
            out = layer(out, features[i])
        return self.out_layers[0](out)


class Decoder_A(nn.Module):
    def __init__(
            self,
            dims: Sequence[int],
            num_classes: int,
            depths: Sequence[int] = (2, 2, 2, 2),
            drop_path_rate: float = 0.2,
    ) -> None:
        super(Decoder_A, self).__init__()
        dpr = [x.item() for x in torch.linspace(drop_path_rate, 0, (len(dims) - 1) * 2)]

        self.layers = nn.ModuleList()
        for i in range(1, len(dims)):
            self.layers.append(
                UpBlock(
                    in_channels=dims[i - 1],
                    out_channels=dims[i],
                    depth=depths[i],
                    drop_path=dpr[sum(depths[: i - 1]): sum(depths[: i])],
                ))
        self.aux1_layers = nn.Sequential(FLKPE(dims[-2], num_classes, dim_scale=8))
        self.aux2_layers = nn.Sequential(FLKPE(dims[-3], num_classes, dim_scale=16))
        self.out_layers = nn.Sequential(FLKPE(dims[-1], num_classes))

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        outs = []
        out = features[0]
        features = features[1:]
        out = self.layers[0](out, features[0])
        outs.insert(0, self.aux2_layers[0](out))
        out = self.layers[1](out, features[1])
        outs.insert(0, self.aux1_layers[0](out))
        out = self.layers[2](out, features[2])
        outs.insert(0, self.out_layers[0](out))
        return outs


class Decoder_B(nn.Module):
    def __init__(
            self,
            dims: Sequence[int],
            num_classes: int,
            depths: Sequence[int] = (2, 2, 2, 2),
            drop_path_rate: float = 0.2,
    ) -> None:
        super(Decoder_B, self).__init__()
        dpr = [x.item() for x in torch.linspace(drop_path_rate, 0, (len(dims) - 1) * 2)]

        self.layers = nn.ModuleList()
        for i in range(1, len(dims)):
            self.layers.append(
                UpAttBlock(
                    in_channels=dims[i - 1],
                    out_channels=dims[i],
                    depth=depths[i],
                    drop_path=dpr[sum(depths[: i - 1]): sum(depths[: i])],
                ))
        self.aux1_layers = nn.Sequential(FLKPE(dims[-2], num_classes, dim_scale=8))
        self.aux2_layers = nn.Sequential(FLKPE(dims[-3], num_classes, dim_scale=16))
        self.out_layers = nn.Sequential(FLKPE(dims[-1], num_classes))

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        outs = []
        out = features[0]
        features = features[1:]
        out = self.layers[0](out, features[0])
        outs.insert(0, self.aux2_layers[0](out))
        out = self.layers[1](out, features[1])
        outs.insert(0, self.aux1_layers[0](out))
        out = self.layers[2](out, features[2])
        outs.insert(0, self.out_layers[0](out))
        return outs


class MSVMUNet(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            num_classes: int = 9,
            *,
            enc_name: str = "tiny_0230s"  # tiny_0230s, small_0229s
    ) -> None:
        """
        reference to: https://arxiv.org/pdf/2408.13735v1
        """
        super(MSVMUNet, self).__init__()
        self.in_channels = in_channels
        self.encoder = Encoder(enc_name, in_channels=in_channels)
        self.dims = self.encoder.dims
        self.decoder = Decoder(dims=self.dims[::-1], num_classes=num_classes)

    def forward(self, x: Tensor) -> Tensor | tuple[Tensor]:
        if x.shape[1] == 1:
            x = x.repeat(1, self.in_channels, 1, 1)
        return self.decoder(self.encoder(x)[::-1])

    @torch.no_grad()
    def freeze_encoder(self) -> None:
        self.encoder.freeze_params()

    @torch.no_grad()
    def unfreeze_encoder(self) -> None:
        self.encoder.unfreeze_params()


class MSVMUNet_A(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            num_classes: int = 9,
            *,
            enc_name: str = "tiny_0230s"  # tiny_0230s, small_0229s
    ) -> None:
        """
        MSVM-UNet-A: MSVM-UNet + Context Extractor
        reference to: https://arxiv.org/pdf/2408.13735v1
        """
        super(MSVMUNet_A, self).__init__()
        self.in_channels = in_channels
        self.encoder = Encoder(enc_name, in_channels=in_channels)
        self.dims = self.encoder.dims
        self.mid_layer = Context_Extractor(in_channels=self.dims[-1])
        self.dims[-1] += 4
        self.decoder = Decoder(dims=self.dims[::-1], num_classes=num_classes)

    def forward(self, x: Tensor) -> Tensor | tuple[Tensor]:
        if x.shape[1] == 1:
            x = x.repeat(1, self.in_channels, 1, 1)
        features = self.encoder(x)[::-1]
        features[0] = self.mid_layer(features[0])
        return self.decoder(features)

    @torch.no_grad()
    def freeze_encoder(self) -> None:
        self.encoder.freeze_params()

    @torch.no_grad()
    def unfreeze_encoder(self) -> None:
        self.encoder.unfreeze_params()


class MSVMUNet_B(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            num_classes: int = 9,
            *,
            enc_name: str = "tiny_0230s"  # tiny_0230s, small_0229s
    ) -> None:
        """
        MSVM-UNet-A: MSVM-UNet + Deep Supervision
        reference to: https://arxiv.org/pdf/2408.13735v1
        """
        super(MSVMUNet_B, self).__init__()
        self.in_channels = in_channels
        self.encoder = Encoder(enc_name, in_channels=in_channels)
        self.dims = self.encoder.dims
        self.decoder = Decoder_A(dims=self.dims[::-1], num_classes=num_classes)

    def forward(self, x: Tensor) -> Tensor | tuple[Tensor]:
        if x.shape[1] == 1:
            x = x.repeat(1, self.in_channels, 1, 1)
        features = self.encoder(x)[::-1]
        return self.decoder(features)

    @torch.no_grad()
    def freeze_encoder(self) -> None:
        self.encoder.freeze_params()

    @torch.no_grad()
    def unfreeze_encoder(self) -> None:
        self.encoder.unfreeze_params()


class MSVMUNet_C(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            num_classes: int = 9,
            *,
            enc_name: str = "tiny_0230s"  # tiny_0230s, small_0229s
    ) -> None:
        """
        MSVM-UNet-A: MSVM-UNet + Deep Supervision + Context Extractor
        reference to: https://arxiv.org/pdf/2408.13735v1
        """
        super(MSVMUNet_C, self).__init__()
        self.in_channels = in_channels
        self.encoder = Encoder(enc_name, in_channels=in_channels)
        self.dims = self.encoder.dims
        self.mid_layer = Context_Extractor(in_channels=self.dims[-1])
        self.dims[-1] += 4
        self.decoder = Decoder_A(dims=self.dims[::-1], num_classes=num_classes)

    def forward(self, x: Tensor) -> Tensor | tuple[Tensor]:
        if x.shape[1] == 1:
            x = x.repeat(1, self.in_channels, 1, 1)
        features = self.encoder(x)[::-1]
        features[0] = self.mid_layer(features[0])
        return self.decoder(features)

    @torch.no_grad()
    def freeze_encoder(self) -> None:
        self.encoder.freeze_params()

    @torch.no_grad()
    def unfreeze_encoder(self) -> None:
        self.encoder.unfreeze_params()


class MSVMUNet_D(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            num_classes: int = 9,
            *,
            enc_name: str = "tiny_0230s"  # tiny_0230s, small_0229s
    ) -> None:
        """
        MSVM-UNet-A: MSVM-UNet + Deep Supervision + Attention
        reference to: https://arxiv.org/pdf/2408.13735v1
        """
        super(MSVMUNet_D, self).__init__()
        self.in_channels = in_channels
        self.encoder = Encoder(enc_name, in_channels=in_channels)
        self.dims = self.encoder.dims
        self.decoder = Decoder_B(dims=self.dims[::-1], num_classes=num_classes)

    def forward(self, x: Tensor) -> Tensor | tuple[Tensor]:
        if x.shape[1] == 1:
            x = x.repeat(1, self.in_channels, 1, 1)
        features = self.encoder(x)[::-1]
        return self.decoder(features)

    @torch.no_grad()
    def freeze_encoder(self) -> None:
        self.encoder.freeze_params()

    @torch.no_grad()
    def unfreeze_encoder(self) -> None:
        self.encoder.unfreeze_params()


class MSVMUNet_E(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            num_classes: int = 9,
            *,
            enc_name: str = "tiny_0230s"  # tiny_0230s, small_0229s
    ) -> None:
        """
        MSVM-UNet-A: MSVM-UNet + Deep Supervision + Context Extractor + Attention
        reference to: https://arxiv.org/pdf/2408.13735v1
        """
        super(MSVMUNet_E, self).__init__()
        self.in_channels = in_channels
        self.encoder = Encoder(enc_name, in_channels=in_channels)
        self.dims = self.encoder.dims
        self.mid_layer = Context_Extractor(in_channels=self.dims[-1])
        self.dims[-1] += 4
        self.decoder = Decoder_B(dims=self.dims[::-1], num_classes=num_classes)

    def forward(self, x: Tensor) -> Tensor | tuple[Tensor]:
        if x.shape[1] == 1:
            x = x.repeat(1, self.in_channels, 1, 1)
        features = self.encoder(x)[::-1]
        features[0] = self.mid_layer(features[0])
        return self.decoder(features)

    @torch.no_grad()
    def freeze_encoder(self) -> None:
        self.encoder.freeze_params()

    @torch.no_grad()
    def unfreeze_encoder(self) -> None:
        self.encoder.unfreeze_params()




