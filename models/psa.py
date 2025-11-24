import torch
import torch.nn as nn


class ParallelPolarizedSelfAttention(nn.Module):
    def __init__(
        self, channel=512, use_bias=True, spatial_weight=0.5, channel_weight=0.5
    ):
        super(ParallelPolarizedSelfAttention, self).__init__()

        self.use_bias = use_bias
        self.spatial_weight = spatial_weight
        self.channel_weight = channel_weight

        self.ch_wv = nn.Conv2d(
            channel, channel // 2, kernel_size=(1, 1), bias=self.use_bias
        )
        self.ch_wq = nn.Conv2d(channel, 1, kernel_size=(1, 1), bias=self.use_bias)
        self.softmax_channel = nn.Softmax(1)
        self.ch_wz = nn.Conv2d(
            channel // 2, channel, kernel_size=(1, 1), bias=self.use_bias
        )
        self.ln = nn.LayerNorm(channel)
        self.sigmoid = nn.Sigmoid()

        self.sp_wv = nn.Conv2d(
            channel, channel // 2, kernel_size=(1, 1), bias=self.use_bias
        )
        self.sp_wq = nn.Conv2d(
            channel, channel // 2, kernel_size=(1, 1), bias=self.use_bias
        )
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.softmax_spatial = nn.Softmax(-1)

    def channel_pool(self, x):
        b, c, h, w = x.size()
        # Channel-only Self-Attention
        channel_wv = self.ch_wv(x)  # bs,c//2,h,w
        channel_wq = self.ch_wq(x)  # bs,1,h,w
        channel_wv = channel_wv.reshape(b, c // 2, -1)  # bs,c//2,h*w
        channel_wq = channel_wq.reshape(b, -1, 1)  # bs,h*w,1
        channel_wq = self.softmax_channel(channel_wq)
        channel_wz = torch.matmul(channel_wv, channel_wq).unsqueeze(-1)  # bs,c//2,1,1
        channel_weight = (
            self.sigmoid(
                self.ln(self.ch_wz(channel_wz).reshape(b, c, 1).permute(0, 2, 1))
            )
            .permute(0, 2, 1)
            .reshape(b, c, 1, 1)
        )  # bs,c,1,1
        channel_out = channel_weight * x
        return channel_out

    def spatial_pool(self, x):
        b, c, h, w = x.size()
        # Spatial-only Self-Attention
        spatial_wq = self.sp_wq(x)  # bs,c//2,h,w
        spatial_wq = self.agp(spatial_wq)  # bs,c//2,1,1
        spatial_wq = spatial_wq.permute(0, 2, 3, 1).reshape(b, 1, c // 2)  # bs,1,c//2
        spatial_wq = self.softmax_spatial(spatial_wq)

        spatial_wv = self.sp_wv(x)  # bs,c//2,h,w
        spatial_wv = spatial_wv.reshape(b, c // 2, -1)  # bs,c//2,h*w

        spatial_wz = torch.matmul(spatial_wq, spatial_wv)  # bs,1,h*w
        spatial_weight = self.sigmoid(spatial_wz.reshape(b, 1, h, w))  # bs,1,h,w
        spatial_out = spatial_weight * x
        return spatial_out

    def forward(self, x):
        channel_out = self.channel_pool(x)
        spatial_out = self.spatial_pool(x)
        out = self.spatial_weight * spatial_out + self.channel_weight * channel_out

        return out


class ParallelPolarizedSelfAttentionEdge(nn.Module):
    def __init__(
        self, channel=512, use_bias=True, spatial_weight=0.5, channel_weight=0.5
    ):
        super(ParallelPolarizedSelfAttentionEdge, self).__init__()

        self.use_bias = use_bias
        self.spatial_weight = spatial_weight
        self.channel_weight = channel_weight

        self.ch_wv = nn.Conv2d(
            channel, channel // 2, kernel_size=(1, 1), bias=self.use_bias
        )
        self.ch_wq = nn.Conv2d(channel, 1, kernel_size=(1, 1), bias=self.use_bias)
        self.softmax_channel = nn.Softmax(1)
        self.ch_wz = nn.Conv2d(
            channel // 2, channel, kernel_size=(1, 1), bias=self.use_bias
        )
        self.ln = nn.LayerNorm(channel)
        self.sigmoid = nn.Sigmoid()

        self.sp_wv = nn.Conv2d(
            channel, channel // 2, kernel_size=(1, 1), bias=self.use_bias
        )
        self.sp_wq = nn.Conv2d(
            channel, channel // 2, kernel_size=(1, 1), bias=self.use_bias
        )
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.softmax_spatial = nn.Softmax(-1)

    def channel_pool(self, x, x_edge):
        b, c, _, _ = x.size()
        # Channel-only Self-Attention
        channel_wv = self.ch_wv(x)  # bs,c//2,h,w
        # initially we tried to use x_edge for ch_wq but the results are not good
        channel_wq = self.ch_wq(x)  # bs,1,h,w
        channel_wv = channel_wv.reshape(b, c // 2, -1)  # bs,c//2,h*w
        channel_wq = channel_wq.reshape(b, -1, 1)  # bs,h*w,1
        channel_wq = self.softmax_channel(channel_wq)
        channel_wz = torch.matmul(channel_wv, channel_wq).unsqueeze(-1)  # bs,c//2,1,1
        channel_weight = (
            self.sigmoid(
                self.ln(self.ch_wz(channel_wz).reshape(b, c, 1).permute(0, 2, 1))
            )
            .permute(0, 2, 1)
            .reshape(b, c, 1, 1)
        )  # bs,c,1,1
        channel_out = channel_weight * x
        return channel_out

    def spatial_pool(self, x, x_edge):
        b, c, h, w = x.size()
        # Spatial-only Self-Attention
        spatial_wq = self.sp_wq(x_edge)  # bs,c//2,h,w
        spatial_wq = self.agp(spatial_wq)  # bs,c//2,1,1
        spatial_wq = spatial_wq.permute(0, 2, 3, 1).reshape(b, 1, c // 2)  # bs,1,c//2
        spatial_wq = self.softmax_spatial(spatial_wq)

        spatial_wv = self.sp_wv(x_edge)  # bs,c//2,h,w
        spatial_wv = spatial_wv.reshape(b, c // 2, -1)  # bs,c//2,h*w

        spatial_wz = torch.matmul(spatial_wq, spatial_wv)  # bs,1,h*w
        spatial_weight = self.sigmoid(spatial_wz.reshape(b, 1, h, w))  # bs,1,h,w
        spatial_out = spatial_weight * x
        return spatial_out

    def forward(self, x, x_edge):
        channel_out = self.channel_pool(x, x_edge)
        spatial_out = self.spatial_pool(x, x_edge)
        out = self.spatial_weight * spatial_out + self.channel_weight * channel_out

        return out
