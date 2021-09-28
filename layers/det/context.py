import megengine.functional as F
import megengine.module as M
import math

def last_zero_init(m):
    if isinstance(m, M.Sequential):
        M.init.zeros_(m[-1].weight)
        M.init.zeros_(m[-1].bias)
    else:
        M.init.zeros_(m.conv2.weight)
        M.init.zeros_(m.conv2.bias)


class Channel_conv(M.Module):
    def __init__(self,
                 inplanes,
                 planes
                 ):
        super(Channel_conv, self).__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.conv1 = M.Conv2d(self.inplanes, self.planes, kernel_size=1)
        self.norm1 = M.LayerNorm(self.planes)
        self.conv2 = M.Conv2d(self.planes, self.inplanes, kernel_size=1)

    def forward(self, x):

        x = self.conv1(x)
        x = self.norm1(x)
        x = F.relu(x)
        x = self.conv2(x)

        return x

class ContextBlock(M.Module):

    def __init__(self,
                 inplanes,
                 ratio,
                 pooling_type='att',
                 fusion_types=('channel_add', )):
        super(ContextBlock, self).__init__()
        assert pooling_type in ['avg', 'att']
        assert isinstance(fusion_types, (list, tuple))
        valid_fusion_types = ['channel_add', 'channel_mul']
        assert all([f in valid_fusion_types for f in fusion_types])
        assert len(fusion_types) > 0, 'at least one fusion should be used'
        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types
        self.avg_pool = M.AdaptiveAvgPool2d(1)
        if pooling_type == 'att':
            self.conv_mask = M.Conv2d(inplanes, 1, kernel_size=1)
        if 'channel_add' in fusion_types:
            # self.channel_add_conv = Channel_conv(self.inplanes, self.planes)
            self.channel_add_conv = M.Sequential(
                M.Conv2d(self.inplanes, self.planes, kernel_size=1),
                M.LayerNorm(self.planes),
                M.ReLU(),
                M.Conv2d(self.planes, self.inplanes, kernel_size=1)
            )

        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = Channel_conv(self.inplanes, self.planes)
        else:
            self.channel_mul_conv = None
        self.reset_parameters()

    def reset_parameters(self):

        if self.pooling_type == 'att':
            M.init.msra_normal_(self.conv_mask.weight, mode="fan_in", nonlinearity="relu")
            if self.conv_mask.bias is not None:
                fan_in, _ = M.init.calculate_fan_in_and_fan_out(self.conv_mask.weight)
                bound = 1 / math.sqrt(fan_in)
                M.init.uniform_(self.conv_mask.bias, -bound, bound)
            # kaiming_init(self.conv_mask, mode='fan_in')
            self.conv_mask.inited = True

        if self.channel_add_conv is not None:
            last_zero_init(self.channel_add_conv)
        if self.channel_mul_conv is not None:
            last_zero_init(self.channel_mul_conv)
        # print(self.channel_add_conv.conv2.weight)

    def spatial_pool(self, x):
        batch, channel, height, width = x.shape
        if self.pooling_type == 'att':
            input_x = x
            # [N, C, H * W]
            input_x =  F.reshape(input_x, (batch, channel, height * width))
            # [N, 1, C, H * W]
            input_x = F.expand_dims(input_x, 1)
            # [N, 1, H, W]
            context_mask = self.conv_mask(x)
            # [N, 1, H * W]
            context_mask = F.reshape(context_mask, (batch, 1, height * width))
            # [N, 1, H * W]
            context_mask = F.softmax(context_mask,axis=2)
            # [N, 1, H * W, 1]
            context_mask = F.expand_dims(context_mask, -1)
            # [N, 1, C, 1]
            context = F.matmul(input_x, context_mask)
            # [N, C, 1, 1]
            context = F.reshape(context, (batch, channel, 1, 1))
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)

        return context

    def forward(self, x):
        # [N, C, 1, 1]
        context = self.spatial_pool(x)
        out = x
        if self.channel_mul_conv is not None:
            # [N, C, 1, 1]
            channel_mul_term = F.sigmoid(self.channel_mul_conv(context))
            out = out * channel_mul_term
        if self.channel_add_conv is not None:
            # [N, C, 1, 1]
            channel_add_term = self.channel_add_conv(context)
            # print(channel_add_term.shape,out.shape)
            out = out + channel_add_term

        return out

# inp = F.zeros((8,256,64,64))
# norm = M.LayerNorm(256)
# gc = ContextBlock(256,1/4)
# print(norm(inp).shape)
# print(gc(inp).shape)




