import torch.nn as nn
from model.encoder import Encoder
from model.decoder import Decoder
from options import HiDDenConfiguration
from torchvision.models import resnet34, resnet18
#from noise_layers.noiser import Noiser
import random


#def To_image(encoded_image,image)

class HookTool:
    def __init__(self):
        self.fea = None

    def hook_fun(self, module, fea_in, fea_out):
        '''
        注意用于处理feature的hook函数必须包含三个参数[module, fea_in, fea_out]，参数的名字可以自己起，但其意义是
        固定的，第一个参数表示torch里的一个子module，比如Linear,Conv2d等，第二个参数是该module的输入，其类型是
        tuple；第三个参数是该module的输出，其类型是tensor。注意输入和输出的类型是不一样的，切记。
        '''
        self.fea = fea_in


def get_feas_by_hook(model):
    """
    提取Conv2d后的feature，我们需要遍历模型的module，然后找到Conv2d，把hook函数注册到这个module上；
    这就相当于告诉模型，我要在Conv2d这一层，用hook_fun处理该层输出的feature.
    由于一个模型中可能有多个Conv2d，所以我们要用hook_feas存储下来每一个Conv2d后的feature
    """
    fea_hooks = []
    for n, m in model.named_modules():
        # if isinstance(m, torch.nn.AdaptiveAvgPool2d):
        # if n == 'layer1.0.conv1' or n == 'layer2.0.conv1' or n == 'layer3.0.conv1' or n == 'fc.0':
        if n == 'fc' or n == "classifier" or n == "head":
            cur_hook = HookTool()
            m.register_forward_hook(cur_hook.hook_fun)
            fea_hooks.append(cur_hook)

    return fea_hooks


class EncoderDecoder(nn.Module):
    """
    Combines Encoder->Noiser->Decoder into single pipeline.
    The input is the cover image and the watermark message. The module inserts the watermark into the image
    (obtaining encoded_image), then applies Noise layers (obtaining noised_image), then passes the noised_image
    to the Decoder which tries to recover the watermark (called decoded_message). The module outputs
    a three-tuple: (encoded_image, noised_image, decoded_message)
    """
    def __init__(self, config: HiDDenConfiguration, noiser,me_model):

        super(EncoderDecoder, self).__init__()
        self.encoder = Encoder(config)
        self.noiser = noiser
        self.me_model = me_model
        self.decoder = Decoder(config)

    def forward(self, image, message):
        encoded_image = self.encoder(image, message)
        #noised_and_cover = self.noiser([encoded_image, image])
        # embedding, loggit = self.noiser(encoded_image)
        fea_hooks = get_feas_by_hook(self.noiser)
        me_fea_hooks = get_feas_by_hook(self.me_model)
        output = self.noiser(encoded_image)
        me_output = self.me_model(encoded_image)

        embedding = fea_hooks[0].fea[0]
        me_embedding = me_fea_hooks[0].fea[0]
        # embedding = output
        #noised_image = noised_and_cover[0]

        # embedding = embedding + 0.20 * random.gauss(0, 1)

        # decoded_message = self.decoder(output)

        decoded_message = self.decoder(embedding)               #解码信息
        me_decoded_message = self.decoder(me_embedding)  # 解码信息

        # surrogate_model = resnet18(pretrained=True).to('cuda')
        fea_hooks = get_feas_by_hook(self.noiser)
        s_me_fea_hooks = get_feas_by_hook(self.me_model)
        s_output = self.noiser(image)                   #干净图片输入
        s_me_output = self.me_model(image)              #干净图片输入盗版模型

        s_embedding = fea_hooks[0].fea[0]               #普通模型的me
        s_me_embedding = s_me_fea_hooks[0].fea[0]       #这个是me的embedding
        #s_decoded_message = self.decoder(s_embedding)
        # s_decoded_message = self.decoder(s_output)
##带信息的图像，干净图像的embedding，me模型干净图像的embedding，带信息的embedding，解码出的信息，me的解码信息
        return encoded_image,s_embedding, s_me_embedding, embedding, decoded_message, me_decoded_message
