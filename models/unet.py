import torch
import torch.nn as nn
import torch.nn.functional as F


class Double_Conv2d(nn.Module):
    def __init__(self, in_channel, out_channel, use_batch_norm):
        super(Double_Conv2d, self).__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.use_batch_norm = use_batch_norm

        if self.use_batch_norm:
            self.double_conv2d = torch.nn.Sequential(
                nn.Conv2d(
                    in_channels=self.in_channel,
                    out_channels=self.out_channel,
                    kernel_size=3,
                    padding=1,
                ),
                nn.BatchNorm2d(self.out_channel),
                nn.LeakyReLU(0.2),
                nn.Conv2d(
                    in_channels=self.out_channel,
                    out_channels=self.out_channel,
                    kernel_size=3,
                    padding=1,
                ),
                nn.BatchNorm2d(self.out_channel),
                nn.LeakyReLU(0.2),
            )
        else:
            self.double_conv2d = torch.nn.Sequential(
                nn.Conv2d(
                    in_channels=self.in_channel,
                    out_channels=self.out_channel,
                    kernel_size=3,
                    padding=1,
                ),
                nn.LeakyReLU(0.2),
                nn.Conv2d(
                    in_channels=self.out_channel,
                    out_channels=self.out_channel,
                    kernel_size=3,
                    padding=1,
                ),
                nn.LeakyReLU(0.2),
            )

    def forward(self, x):
        return self.double_conv2d(x)


class Encoder(nn.Module):
    def __init__(self, in_channel, multiplier, use_batch_norm):
        super(Encoder, self).__init__()

        self.in_channel = in_channel
        self.multiplier = multiplier
        self.use_batch_norm = use_batch_norm

        self.conv1 = Double_Conv2d(
            self.in_channel, 16 * self.multiplier, self.use_batch_norm
        )
        self.conv2 = Double_Conv2d(
            16 * self.multiplier, 32 * self.multiplier, self.use_batch_norm
        )
        self.conv3 = Double_Conv2d(
            32 * self.multiplier, 64 * self.multiplier, self.use_batch_norm
        )
        self.conv4 = Double_Conv2d(
            64 * self.multiplier, 128 * self.multiplier, self.use_batch_norm
        )
        self.conv5 = Double_Conv2d(
            128 * self.multiplier, 256 * self.multiplier, self.use_batch_norm
        )

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(F.max_pool2d(conv1, kernel_size=2))
        conv3 = self.conv3(F.max_pool2d(conv2, kernel_size=2))
        conv4 = self.conv4(F.max_pool2d(conv3, kernel_size=2))
        conv5 = self.conv5(F.max_pool2d(conv4, kernel_size=2))
        return [conv1, conv2, conv3, conv4, conv5]


class Decoder(nn.Module):
    def __init__(self, out_channel, multiplier, use_batch_norm):
        super(Decoder, self).__init__()

        self.out_channel = out_channel
        self.multiplier = multiplier
        self.use_batch_norm = use_batch_norm

        self.up6 = nn.ConvTranspose2d(
            256 * self.multiplier, 128 * self.multiplier, kernel_size=2, stride=2
        )
        self.conv6 = Double_Conv2d(
            256 * self.multiplier, 128 * self.multiplier, self.use_batch_norm
        )
        self.up7 = nn.ConvTranspose2d(
            128 * self.multiplier, 64 * self.multiplier, kernel_size=2, stride=2
        )
        self.conv7 = Double_Conv2d(
            128 * self.multiplier, 64 * self.multiplier, self.use_batch_norm
        )
        self.up8 = nn.ConvTranspose2d(
            64 * self.multiplier, 32 * self.multiplier, kernel_size=2, stride=2
        )
        self.conv8 = Double_Conv2d(
            64 * self.multiplier, 32 * self.multiplier, self.use_batch_norm
        )
        self.up9 = nn.ConvTranspose2d(
            32 * self.multiplier, 16 * self.multiplier, kernel_size=2, stride=2
        )
        self.conv9 = Double_Conv2d(
            32 * self.multiplier, 16 * self.multiplier, self.use_batch_norm
        )
        self.conv10 = nn.Conv2d(
            in_channels=16 * self.multiplier,
            out_channels=self.out_channel,
            kernel_size=1,
        )

    def forward(self, x=[]):
        conv5, conv4, conv3, conv2, conv1 = x[0], x[1], x[2], x[3], x[4]
        up6 = self.up6(conv5)
        up6 = torch.cat([up6, conv4], 1)
        conv6 = self.conv6(up6)

        up7 = self.up7(conv6)
        up7 = torch.cat([up7, conv3], 1)
        conv7 = self.conv7(up7)

        up8 = self.up8(conv7)
        up8 = torch.cat([up8, conv2], 1)
        conv8 = self.conv8(up8)

        up9 = self.up9(conv8)
        up9 = torch.cat([up9, conv1], 1)
        conv9 = self.conv9(up9)

        conv10 = self.conv10(conv9)
        return conv10


class GrayEdgeAttentionUNet(nn.Module):
    def __init__(self):
        super(GrayEdgeAttentionUNet, self).__init__()
        self.conv1 = Double_Conv2d(4, 32, False)
        self.conv2 = Double_Conv2d(32, 64, False)
        self.conv3 = Double_Conv2d(64, 128, False)
        self.conv4 = Double_Conv2d(128, 256, False)
        self.conv5 = Double_Conv2d(256, 512, False)
        self.up6 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv6 = Double_Conv2d(512, 256, False)
        self.up7 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv7 = Double_Conv2d(256, 128, False)
        self.up8 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv8 = Double_Conv2d(128, 64, False)
        self.up9 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv9 = Double_Conv2d(64, 32, False)
        self.conv10 = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=1)

    def forward(self, x, gray, edge):
        gray2 = F.max_pool2d(gray, kernel_size=2)
        gray3 = F.max_pool2d(gray2, kernel_size=2)
        gray4 = F.max_pool2d(gray3, kernel_size=2)
        gray5 = F.max_pool2d(gray4, kernel_size=2)

        x = torch.cat([x, edge], 1)

        conv1 = self.conv1(x)
        pool1 = F.max_pool2d(conv1, kernel_size=2)

        conv2 = self.conv2(pool1)
        pool2 = F.max_pool2d(conv2, kernel_size=2)

        conv3 = self.conv3(pool2)
        pool3 = F.max_pool2d(conv3, kernel_size=2)

        conv4 = self.conv4(pool3)
        pool4 = F.max_pool2d(conv4, kernel_size=2)

        conv5 = self.conv5(pool4)
        conv5 = conv5 * gray5

        up6 = self.up6(conv5)
        conv4 = conv4 * gray4
        up6 = torch.cat([up6, conv4], 1)
        conv6 = self.conv6(up6)

        up7 = self.up7(conv6)
        conv3 = conv3 * gray3
        up7 = torch.cat([up7, conv3], 1)
        conv7 = self.conv7(up7)

        up8 = self.up8(conv7)
        conv2 = conv2 * gray2
        up8 = torch.cat([up8, conv2], 1)
        conv8 = self.conv8(up8)

        up9 = self.up9(conv8)
        conv1 = conv1 * gray
        up9 = torch.cat([up9, conv1], 1)
        conv9 = self.conv9(up9)

        conv10 = self.conv10(conv9)
        out = F.pixel_shuffle(conv10, 1)

        return out


class UNet(nn.Module):
    def __init__(self, use_batch_norm=False):
        super(UNet, self).__init__()
        self.conv1 = Double_Conv2d(3, 32, use_batch_norm)
        self.conv2 = Double_Conv2d(32, 64, use_batch_norm)
        self.conv3 = Double_Conv2d(64, 128, use_batch_norm)
        self.conv4 = Double_Conv2d(128, 256, use_batch_norm)
        self.conv5 = Double_Conv2d(256, 512, use_batch_norm)
        self.up6 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv6 = Double_Conv2d(512, 256, use_batch_norm)
        self.up7 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv7 = Double_Conv2d(256, 128, use_batch_norm)
        self.up8 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv8 = Double_Conv2d(128, 64, use_batch_norm)
        self.up9 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv9 = Double_Conv2d(64, 32, use_batch_norm)
        self.conv10 = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        pool1 = F.max_pool2d(conv1, kernel_size=2)

        conv2 = self.conv2(pool1)
        pool2 = F.max_pool2d(conv2, kernel_size=2)

        conv3 = self.conv3(pool2)
        pool3 = F.max_pool2d(conv3, kernel_size=2)

        conv4 = self.conv4(pool3)
        pool4 = F.max_pool2d(conv4, kernel_size=2)

        conv5 = self.conv5(pool4)

        up6 = self.up6(conv5)
        up6 = torch.cat([up6, conv4], 1)
        conv6 = self.conv6(up6)

        up7 = self.up7(conv6)
        up7 = torch.cat([up7, conv3], 1)
        conv7 = self.conv7(up7)

        up8 = self.up8(conv7)
        up8 = torch.cat([up8, conv2], 1)
        conv8 = self.conv8(up8)

        up9 = self.up9(conv8)
        up9 = torch.cat([up9, conv1], 1)
        conv9 = self.conv9(up9)

        conv10 = self.conv10(conv9)
        out = F.pixel_shuffle(conv10, 1)

        return out


class UNet_Att(nn.Module):
    def __init__(self, use_bias=True, use_batch_norm=False):
        super(UNet_Att, self).__init__()
        self.main_encoder = Encoder(3, 2, use_batch_norm)
        self.up6 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv6 = Double_Conv2d(512, 256, use_batch_norm)
        self.up7 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv7 = Double_Conv2d(256, 128, use_batch_norm)
        self.up8 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv8 = Double_Conv2d(128, 64, use_batch_norm)
        self.up9 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv9 = Double_Conv2d(64, 32, use_batch_norm)
        self.conv10 = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=1)

        self.use_bias = use_bias

        from models.psa import ParallelPolarizedSelfAttention as PSA_P

        self.psa_1 = PSA_P(channel=32, use_bias=self.use_bias)
        self.psa_2 = PSA_P(channel=64, use_bias=self.use_bias)
        self.psa_3 = PSA_P(channel=128, use_bias=self.use_bias)
        self.psa_4 = PSA_P(channel=256, use_bias=self.use_bias)
        self.psa_5 = PSA_P(channel=512, use_bias=self.use_bias)

    def forward(self, x):
        [low_conv1, low_conv2, low_conv3, low_conv4, low_conv5] = self.main_encoder(x)

        conv5 = self.psa_5(low_conv5)
        conv4 = self.psa_4(low_conv4)
        conv3 = self.psa_3(low_conv3)
        conv2 = self.psa_2(low_conv2)
        conv1 = self.psa_1(low_conv1)

        up6 = self.up6(conv5)
        up6 = torch.cat([up6, conv4], 1)
        conv6 = self.conv6(up6)

        up7 = self.up7(conv6)
        up7 = torch.cat([up7, conv3], 1)
        conv7 = self.conv7(up7)

        up8 = self.up8(conv7)
        up8 = torch.cat([up8, conv2], 1)
        conv8 = self.conv8(up8)

        up9 = self.up9(conv8)
        up9 = torch.cat([up9, conv1], 1)
        conv9 = self.conv9(up9)

        conv10 = self.conv10(conv9)
        out = F.pixel_shuffle(conv10, 1)

        return out


class CCUNet(nn.Module):
    def __init__(
        self,
        psa_type=None,
        use_bias=True,
        use_batch_norm=True,
        spatial_weight=0.5,
        channel_weight=0.5,
    ):
        super(CCUNet, self).__init__()
        self.psa_type = psa_type
        self.use_bias = use_bias
        self.use_batch_norm = use_batch_norm
        self.spatial_weight = spatial_weight
        self.channel_weight = channel_weight

        if self.psa_type == "normal":
            from models.psa import ParallelPolarizedSelfAttention as PSA_P

            self.main_encoder = Encoder(3, 1, self.use_batch_norm)
            self.edge_encoder = Encoder(1, 1, self.use_batch_norm)
        elif self.psa_type == "edge":
            from models.psa import ParallelPolarizedSelfAttentionEdge as PSA_P

            self.main_encoder = Encoder(3, 2, self.use_batch_norm)
            self.edge_encoder = Encoder(1, 2, self.use_batch_norm)

        self.psa_1 = PSA_P(
            channel=32,
            use_bias=self.use_bias,
            spatial_weight=self.spatial_weight,
            channel_weight=self.channel_weight,
        )
        self.psa_2 = PSA_P(
            channel=64,
            use_bias=self.use_bias,
            spatial_weight=self.spatial_weight,
            channel_weight=self.channel_weight,
        )
        self.psa_3 = PSA_P(
            channel=128,
            use_bias=self.use_bias,
            spatial_weight=self.spatial_weight,
            channel_weight=self.channel_weight,
        )
        self.psa_4 = PSA_P(
            channel=256,
            use_bias=self.use_bias,
            spatial_weight=self.spatial_weight,
            channel_weight=self.channel_weight,
        )
        self.psa_5 = PSA_P(
            channel=512,
            use_bias=self.use_bias,
            spatial_weight=self.spatial_weight,
            channel_weight=self.channel_weight,
        )

        self.up6 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv6 = Double_Conv2d(512, 256, self.use_batch_norm)
        self.up7 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv7 = Double_Conv2d(256, 128, self.use_batch_norm)
        self.up8 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv8 = Double_Conv2d(128, 64, self.use_batch_norm)
        self.up9 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv9 = Double_Conv2d(64, 32, self.use_batch_norm)
        self.conv10 = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=1)

    def forward(self, low, edge):
        [low_conv1, low_conv2, low_conv3, low_conv4, low_conv5] = self.main_encoder(low)
        [edge_conv1, edge_conv2, edge_conv3, edge_conv4, edge_conv5] = (
            self.edge_encoder(edge)
        )

        if self.psa_type == "normal":
            conv5 = self.psa_5(torch.cat([low_conv5, edge_conv5], 1))
            conv4 = self.psa_4(torch.cat([low_conv4, edge_conv4], 1))
            conv3 = self.psa_3(torch.cat([low_conv3, edge_conv3], 1))
            conv2 = self.psa_2(torch.cat([low_conv2, edge_conv2], 1))
            conv1 = self.psa_1(torch.cat([low_conv1, edge_conv1], 1))
        elif self.psa_type == "edge":
            conv5 = self.psa_5(low_conv5, edge_conv5)
            conv4 = self.psa_4(low_conv4, edge_conv4)
            conv3 = self.psa_3(low_conv3, edge_conv3)
            conv2 = self.psa_2(low_conv2, edge_conv2)
            conv1 = self.psa_1(low_conv1, edge_conv1)

        up6 = self.up6(conv5)
        up6 = torch.cat([up6, conv4], 1)
        conv6 = self.conv6(up6)

        up7 = self.up7(conv6)
        up7 = torch.cat([up7, conv3], 1)
        conv7 = self.conv7(up7)

        up8 = self.up8(conv7)
        up8 = torch.cat([up8, conv2], 1)
        conv8 = self.conv8(up8)

        up9 = self.up9(conv8)
        up9 = torch.cat([up9, conv1], 1)
        conv9 = self.conv9(up9)

        conv10 = self.conv10(conv9)

        return conv10


class Encoder_Conv2d_NestedEdge(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Encoder_Conv2d_NestedEdge, self).__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel

        self.conv_1 = torch.nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channel,
                out_channels=self.out_channel,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(self.out_channel),
            nn.LeakyReLU(0.2),
        )
        self.conv_2 = torch.nn.Sequential(
            nn.Conv2d(
                in_channels=self.out_channel,
                out_channels=self.out_channel,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(self.out_channel),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        conv_1 = self.conv_1(x)
        conv_2 = self.conv_2(conv_1)
        return conv_1, conv_2


class Decoder_Conv2d_NestedEdge(nn.Module):
    def __init__(self, in_channel, out_channel, up_scale_factor, deconv_first=True):
        super(Decoder_Conv2d_NestedEdge, self).__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.up_scale_factor = up_scale_factor
        self.deconv_first = deconv_first

        self.conv_1 = torch.nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channel,
                out_channels=self.in_channel // 2,
                kernel_size=1,
            ),
            nn.BatchNorm2d(self.in_channel // 2),
            nn.LeakyReLU(0.2),
        )

        self.conv_2 = torch.nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channel,
                out_channels=self.in_channel // 2,
                kernel_size=1,
            ),
            nn.BatchNorm2d(self.in_channel // 2),
            nn.LeakyReLU(0.2),
        )

        self.conv_3 = torch.nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channel // 2,
                out_channels=self.out_channel,
                kernel_size=1,
            ),
            nn.BatchNorm2d(self.out_channel),
            nn.LeakyReLU(0.2),
        )

        self.deconv = nn.ConvTranspose2d(
            self.out_channel,
            self.out_channel,
            kernel_size=self.up_scale_factor,
            stride=self.up_scale_factor,
        )

    def forward(self, conv_1, conv_2):
        conv_1 = self.conv_1(conv_1)
        conv_2 = self.conv_2(conv_2)
        conv3 = self.conv_3(conv_1 + conv_2)
        if self.deconv_first:
            deconv = self.deconv(conv3)
            prob_map = torch.sigmoid(deconv)
            return deconv, prob_map
        else:
            prob_map = torch.sigmoid(conv3)
            deconv = self.deconv(conv3)
            return deconv, prob_map


class CCUNet_NestedEdge(nn.Module):
    def __init__(
        self,
        psa_type=None,
        use_bias=True,
        use_batch_norm=True,
        spatial_weight=0.5,
        channel_weight=0.5,
    ):
        super(CCUNet_NestedEdge, self).__init__()
        self.psa_type = psa_type
        self.use_bias = use_bias
        self.use_batch_norm = use_batch_norm
        self.spatial_weight = spatial_weight
        self.channel_weight = channel_weight

        if self.psa_type == "normal":
            from models.psa import ParallelPolarizedSelfAttention as PSA_P

            self.main_encoder = Encoder(3, 1, self.use_batch_norm)
            self.edge_encoder = Encoder(1, 1, self.use_batch_norm)
        elif self.psa_type == "edge":
            from models.psa import ParallelPolarizedSelfAttentionEdge as PSA_P

            self.main_encoder = Encoder(3, 2, self.use_batch_norm)
            self.edge_encoder = Encoder(1, 2, self.use_batch_norm)

        self.psa_1 = PSA_P(
            channel=32,
            use_bias=self.use_bias,
            spatial_weight=self.spatial_weight,
            channel_weight=self.channel_weight,
        )
        self.psa_2 = PSA_P(
            channel=64,
            use_bias=self.use_bias,
            spatial_weight=self.spatial_weight,
            channel_weight=self.channel_weight,
        )
        self.psa_3 = PSA_P(
            channel=128,
            use_bias=self.use_bias,
            spatial_weight=self.spatial_weight,
            channel_weight=self.channel_weight,
        )
        self.psa_4 = PSA_P(
            channel=256,
            use_bias=self.use_bias,
            spatial_weight=self.spatial_weight,
            channel_weight=self.channel_weight,
        )
        self.psa_5 = PSA_P(
            channel=512,
            use_bias=self.use_bias,
            spatial_weight=self.spatial_weight,
            channel_weight=self.channel_weight,
        )

        self.up6 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up7 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up8 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up9 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)

        self.conv6 = Encoder_Conv2d_NestedEdge(512, 256)
        self.conv7 = Encoder_Conv2d_NestedEdge(256, 128)
        self.conv8 = Encoder_Conv2d_NestedEdge(128, 64)
        self.conv9 = Encoder_Conv2d_NestedEdge(64, 32)
        self.conv10 = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=1)

        self.edge_conv6 = Decoder_Conv2d_NestedEdge(256, 1, 8)
        self.edge_conv7 = Decoder_Conv2d_NestedEdge(128, 1, 4)
        self.edge_conv8 = Decoder_Conv2d_NestedEdge(64, 1, 2)
        self.edge_conv9 = Decoder_Conv2d_NestedEdge(32, 1, 1)
        self.score_fuse = nn.Conv2d(4, 1, 1)

    def forward(self, low, edge):
        [low_conv1, low_conv2, low_conv3, low_conv4, low_conv5] = self.main_encoder(low)
        [edge_conv1, edge_conv2, edge_conv3, edge_conv4, edge_conv5] = (
            self.edge_encoder(edge)
        )

        if self.psa_type == "normal":
            conv5 = self.psa_5(torch.cat([low_conv5, edge_conv5], 1))
            conv4 = self.psa_4(torch.cat([low_conv4, edge_conv4], 1))
            conv3 = self.psa_3(torch.cat([low_conv3, edge_conv3], 1))
            conv2 = self.psa_2(torch.cat([low_conv2, edge_conv2], 1))
            conv1 = self.psa_1(torch.cat([low_conv1, edge_conv1], 1))
        elif self.psa_type == "edge":
            conv5 = self.psa_5(low_conv5, edge_conv5)
            conv4 = self.psa_4(low_conv4, edge_conv4)
            conv3 = self.psa_3(low_conv3, edge_conv3)
            conv2 = self.psa_2(low_conv2, edge_conv2)
            conv1 = self.psa_1(low_conv1, edge_conv1)

        # image enhancement decoder
        up6 = self.up6(conv5)
        up6 = torch.cat([up6, conv4], 1)
        conv6_1, conv6_2 = self.conv6(up6)

        up7 = self.up7(conv6_2)
        up7 = torch.cat([up7, conv3], 1)
        conv7_1, conv7_2 = self.conv7(up7)

        up8 = self.up8(conv7_2)
        up8 = torch.cat([up8, conv2], 1)
        conv8_1, conv8_2 = self.conv8(up8)

        up9 = self.up9(conv8_2)
        up9 = torch.cat([up9, conv1], 1)
        conv9_1, conv9_2 = self.conv9(up9)

        conv10 = self.conv10(conv9_2)

        # edge extraction decoder
        edge_conv6, edge_conv6_pm = self.edge_conv6(conv6_1, conv6_2)
        edge_conv7, edge_conv7_pm = self.edge_conv7(conv7_1, conv7_2)
        edge_conv8, edge_conv8_pm = self.edge_conv8(conv8_1, conv8_2)
        edge_conv9, edge_conv9_pm = self.edge_conv9(conv9_1, conv9_2)

        edge_fuse = torch.cat((edge_conv6, edge_conv7, edge_conv8, edge_conv9), dim=1)
        edge_fuse = self.score_fuse(edge_fuse)

        edge_fuse_pm = torch.sigmoid(edge_fuse)

        return conv10, [
            edge_conv6_pm,
            edge_conv7_pm,
            edge_conv8_pm,
            edge_conv9_pm,
            edge_fuse_pm,
        ]


class CCUNet_NestedEdge_v2(nn.Module):
    def __init__(
        self,
        psa_type=None,
        use_bias=True,
        use_batch_norm=True,
        spatial_weight=0.5,
        channel_weight=0.5,
        use_aux_loss=False,
        use_last_edge=False,
        last_edge_att=False,
    ):
        super(CCUNet_NestedEdge_v2, self).__init__()
        self.psa_type = psa_type
        self.use_bias = use_bias
        self.use_batch_norm = use_batch_norm
        self.spatial_weight = spatial_weight
        self.channel_weight = channel_weight
        self.use_aux_loss = use_aux_loss
        self.use_last_edge = use_last_edge
        self.last_edge_att = last_edge_att

        if self.psa_type == "normal":
            from models.psa import ParallelPolarizedSelfAttention as PSA_P

            self.main_encoder = Encoder(3, 1, self.use_batch_norm)
            self.edge_encoder = Encoder(1, 1, self.use_batch_norm)
        elif self.psa_type == "edge":
            from models.psa import ParallelPolarizedSelfAttentionEdge as PSA_P

            self.main_encoder = Encoder(3, 2, self.use_batch_norm)
            self.edge_encoder = Encoder(1, 2, self.use_batch_norm)

        self.psa_1 = PSA_P(
            channel=32,
            use_bias=self.use_bias,
            spatial_weight=self.spatial_weight,
            channel_weight=self.channel_weight,
        )
        self.psa_2 = PSA_P(
            channel=64,
            use_bias=self.use_bias,
            spatial_weight=self.spatial_weight,
            channel_weight=self.channel_weight,
        )
        self.psa_3 = PSA_P(
            channel=128,
            use_bias=self.use_bias,
            spatial_weight=self.spatial_weight,
            channel_weight=self.channel_weight,
        )
        self.psa_4 = PSA_P(
            channel=256,
            use_bias=self.use_bias,
            spatial_weight=self.spatial_weight,
            channel_weight=self.channel_weight,
        )
        self.psa_5 = PSA_P(
            channel=512,
            use_bias=self.use_bias,
            spatial_weight=self.spatial_weight,
            channel_weight=self.channel_weight,
        )

        self.decoder_psa_1 = PSA_P(
            channel=256,
            use_bias=self.use_bias,
            spatial_weight=self.spatial_weight,
            channel_weight=self.channel_weight,
        )
        self.decoder_psa_2 = PSA_P(
            channel=128,
            use_bias=self.use_bias,
            spatial_weight=self.spatial_weight,
            channel_weight=self.channel_weight,
        )
        self.decoder_psa_3 = PSA_P(
            channel=64,
            use_bias=self.use_bias,
            spatial_weight=self.spatial_weight,
            channel_weight=self.channel_weight,
        )
        self.decoder_psa_4 = PSA_P(
            channel=32,
            use_bias=self.use_bias,
            spatial_weight=self.spatial_weight,
            channel_weight=self.channel_weight,
        )

        self.up6 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up7 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up8 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up9 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)

        self.conv6 = Encoder_Conv2d_NestedEdge(512, 256)
        self.conv7 = Encoder_Conv2d_NestedEdge(256, 128)
        self.conv8 = Encoder_Conv2d_NestedEdge(128, 64)
        self.conv9 = Encoder_Conv2d_NestedEdge(64, 32)
        self.conv10 = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=1)

        self.edge_conv6 = Decoder_Conv2d_NestedEdge(256, 1, 8, deconv_first=False)
        self.edge_conv7 = Decoder_Conv2d_NestedEdge(128, 1, 4, deconv_first=False)
        self.edge_conv8 = Decoder_Conv2d_NestedEdge(64, 1, 2, deconv_first=False)

        if self.use_last_edge:
            self.edge_conv9 = Decoder_Conv2d_NestedEdge(32, 1, 1, deconv_first=False)

        self.score_fuse = nn.Conv2d(3, 1, 1)

    def forward(self, low, edge):
        [low_conv1, low_conv2, low_conv3, low_conv4, low_conv5] = self.main_encoder(low)
        [edge_conv1, edge_conv2, edge_conv3, edge_conv4, edge_conv5] = (
            self.edge_encoder(edge)
        )

        if self.psa_type == "normal":
            conv5 = self.psa_5(torch.cat([low_conv5, edge_conv5], 1))
            conv4 = self.psa_4(torch.cat([low_conv4, edge_conv4], 1))
            conv3 = self.psa_3(torch.cat([low_conv3, edge_conv3], 1))
            conv2 = self.psa_2(torch.cat([low_conv2, edge_conv2], 1))
            conv1 = self.psa_1(torch.cat([low_conv1, edge_conv1], 1))
        elif self.psa_type == "edge":
            conv5 = self.psa_5(low_conv5, edge_conv5)
            conv4 = self.psa_4(low_conv4, edge_conv4)
            conv3 = self.psa_3(low_conv3, edge_conv3)
            conv2 = self.psa_2(low_conv2, edge_conv2)
            conv1 = self.psa_1(low_conv1, edge_conv1)

        # image enhancement decoder
        up6 = self.up6(conv5)
        up6 = torch.cat([up6, conv4], 1)
        conv6_1, conv6_2 = self.conv6(up6)
        edge_conv6, edge_conv6_pm = self.edge_conv6(conv6_1, conv6_2)
        final_conv6 = self.decoder_psa_1(conv6_2, edge_conv6_pm * conv6_2)

        up7 = self.up7(final_conv6)
        up7 = torch.cat([up7, conv3], 1)
        conv7_1, conv7_2 = self.conv7(up7)
        edge_conv7, edge_conv7_pm = self.edge_conv7(conv7_1, conv7_2)
        final_conv7 = self.decoder_psa_2(conv7_2, edge_conv7_pm * conv7_2)

        up8 = self.up8(final_conv7)
        up8 = torch.cat([up8, conv2], 1)
        conv8_1, conv8_2 = self.conv8(up8)
        edge_conv8, edge_conv8_pm = self.edge_conv8(conv8_1, conv8_2)
        final_conv8 = self.decoder_psa_3(conv8_2, edge_conv8_pm * conv8_2)

        up9 = self.up9(final_conv8)
        up9 = torch.cat([up9, conv1], 1)
        conv9_1, conv9_2 = self.conv9(up9)

        if self.use_last_edge:
            # this only works when aux_loss is used
            edge_conv9, edge_conv9_pm = self.edge_conv9(conv9_1, conv9_2)

        if self.use_last_edge and self.last_edge_att:
            # this only works when last_edge is used
            conv10 = self.conv10(conv9_2 * edge_conv9_pm)
        else:
            conv10 = self.conv10(conv9_2)

        # edge prediction head
        edge_fuse = torch.cat((edge_conv6, edge_conv7, edge_conv8), dim=1)
        edge_fuse = self.score_fuse(edge_fuse)
        edge_fuse_pm = torch.sigmoid(edge_fuse)
        if self.training:
            if self.use_aux_loss:
                if self.use_last_edge:
                    return conv10, [
                        torch.sigmoid(edge_conv6),
                        torch.sigmoid(edge_conv7),
                        torch.sigmoid(edge_conv8),
                        torch.sigmoid(edge_conv9),
                        edge_fuse_pm,
                    ]
                else:
                    return conv10, [
                        torch.sigmoid(edge_conv6),
                        torch.sigmoid(edge_conv7),
                        torch.sigmoid(edge_conv8),
                        edge_fuse_pm,
                    ]
            else:
                return conv10, [edge_fuse_pm]
        else:
            return conv10, edge_fuse_pm
