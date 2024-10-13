import torch.nn as nn
from options import HiDDenConfiguration
from model.conv_bn_relu import ConvBNRelu


class Decoder(nn.Module):
    """
    Decoder module. Receives a watermarked image and extracts the watermark.
    The input image may have various kinds of noise applied to it,
    such as Crop, JpegCompression, and so on. See Noise layers for more.
    """
    def __init__(self, config: HiDDenConfiguration):

        super(Decoder, self).__init__()
        # self.channels = config.decoder_channels                 #decoder_channels=64
        #
        # layers = [ConvBNRelu(3, self.channels)]                 #3->64
        # for _ in range(config.decoder_blocks - 1):              #64->64
        #     layers.append(ConvBNRelu(self.channels, self.channels))
        #
        # # layers.append(block_builder(self.channels, config.message_length))
        # layers.append(ConvBNRelu(self.channels, config.message_length))         #64->30
        #
        # layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))#nn.AdaptiveAvgPool2d就是自适应平均池化，指定输出（H，W）
        # self.layers = nn.Sequential(*layers)

        #self.linear = nn.Linear(config.message_length, config.message_length)
        fc = [nn.Linear(config.noise_feature,256)]
        # fc.append(nn.Linear(config.noise_feature,512))
        # fc.append(nn.Linear(1024, 1024))
        # fc.append(nn.Linear(1024, 2048))
        # #fc.append(nn.ReLU())
        # fc.append(nn.Linear(2048, 2048))
        # fc.append(nn.Linear(2048, 1024))
        # fc.append(nn.Linear(1024, 512))
        # fc.append(nn.Linear(512, 256))
        fc.append(nn.ReLU())
        #fc.append(nn.ReLU())
        # fc.append(nn.Linear(256, 128))
        # fc.append(nn.Linear(256, 64))
        #fc.append(nn.Linear(64, 64))
        fc.append(nn.Linear(256, config.message_length))
        fc.append(nn.ReLU())
        # fc.append(nn.Linear(config.message_length,config.message_length))
        # fc.append(nn.Linear(config.message_length, config.message_length))
        # fc.append(nn.Linear(config.message_length, config.message_length))
        self.fc = nn.Sequential(*fc)
        # self.fc = nn.Sequential(nn.Linear(config.noise_feature, config.noise_feature), nn.ReLU(),
        #                         nn.Linear(config.noise_feature, config.message_length))

    #def forward(self, image_with_wm):
    def forward(self, embedding):
        #x = self.layers(image_with_wm)
        x = self.fc(embedding)
        # the output is of shape b x c x 1 x 1, and we want to squeeze out the last two dummy dimensions and make
        # the tensor of shape b x c. If we just call squeeze_() it will also squeeze the batch dimension when b=1.
        # x.squeeze_(3).squeeze_(2)  #池化之后降为
        # x = self.linear(x)
        return x
