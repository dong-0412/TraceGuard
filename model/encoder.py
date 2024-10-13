import torch
import torch.nn as nn
from options import HiDDenConfiguration
from model.conv_bn_relu import ConvBNRelu


class Encoder(nn.Module):
    """
    Inserts a watermark into an image.
    """
    def __init__(self, config: HiDDenConfiguration):
        super(Encoder, self).__init__()
        self.H = config.H
        self.W = config.W
        self.conv_channels = config.encoder_channels                #encoder_channels=64
        self.num_blocks = config.encoder_blocks                     #encoder_blocks=4

        layers = [ConvBNRelu(3, self.conv_channels)]                #一层3->64

        for _ in range(config.encoder_blocks-1):                    #其他层64->64
            layer = ConvBNRelu(self.conv_channels, self.conv_channels)
            layers.append(layer)

        self.conv_layers = nn.Sequential(*layers)#参数前面加上*号 ，意味着参数的个数不止一个.
        # 另外带一个星号（*）参数的函数传入的参数存储为一个元组（tuple），带两个（*）号则是表示字典（dict）
        self.after_concat_layer = ConvBNRelu(self.conv_channels + 3 + config.message_length,
                                             self.conv_channels)#64+3+30->64

        self.final_layer = nn.Conv2d(self.conv_channels, 3, kernel_size=1)#一个卷积64->3

    def forward(self, image, message):

        # First, add two dummy dimensions in the end of the message.
        # This is required for the .expand to work correctly
        expanded_message = message.unsqueeze(-1)
        expanded_message.unsqueeze_(-1)#添加两个虚拟维度

        expanded_message = expanded_message.expand(-1,-1, self.H, self.W)#消息扩展成图片的长宽，估计是（1,30,160,160）
        encoded_image = self.conv_layers(image)
        # concatenate expanded message and image
        concat = torch.cat([expanded_message, encoded_image, image], dim=1)#具体维度看一下，通道数结合一下30,64,3
        im_w = self.after_concat_layer(concat)
        im_w = self.final_layer(im_w)                                       #编码了信息的图像。
        return im_w
