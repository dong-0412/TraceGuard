# coding=utf-8
import numpy as np
import torch
import torch.nn as nn
import os
from model.discriminator import Discriminator
from vgg_loss import VGGLoss
from model.encoder import Encoder
from options import HiDDenConfiguration
from PIL import Image
import torchvision
from torchvision import transforms
from model.hidden import Hidden
from noise_layers.noiser import Noiser
from noise_argparser import NoiseArgParser
from utils import image_to_tensor


def main():
    trigger = Image.open('data/train/1/0.png')
    # trigger = (trigger + 1) / 2
    trigger_transform = transforms.Compose([
        transforms.Resize((7, 7)),
        transforms.ToTensor()
    ])
    trigger = trigger_transform(trigger).unsqueeze(0)
    root = "I:/lq/hiden_dataset/GTSRB/benign/test/"
    root_save = 'I:/lq/hiden_dataset/GTSRB/test_result/'

    hidden_config = HiDDenConfiguration(H=160, W=160,
                                        message_length=30,
                                        encoder_blocks=4, encoder_channels=64,
                                        decoder_blocks=7, decoder_channels=64,
                                        use_discriminator=False,
                                        use_vgg=False,
                                        discriminator_blocks=3, discriminator_channels=64,
                                        decoder_loss=1,
                                        encoder_loss=0.7,
                                        adversarial_loss=1e-3,
                                        enable_fp16=False
                                        )
    # encoder = Encoder(config)
    noise_config = []
    noiser = Noiser(noise_config, torch.device('cpu'))
    tb_logger = None
    encoder = Hidden(hidden_config, torch.device('cpu'), noiser, tb_logger)
    encoder.eval()
    # print(encoder)
    # encoder = torch.nn.DataParallel(encoder)
    model_x = torch.load("runs/1 2021.08.12--11-49-51/checkpoints/1--epoch-10.pyt")
    encoder.load_state_dict(model_x['enc-dec-model'])
    # print(encoder)

    # hidden_net.encoder_decoder.load_state_dict(checkpoint['enc-dec-model'])
    k = 0
    for file in os.listdir(root):
        m = 0
        full_file = os.path.join(root, file)
        new_path = 'result/' + file +'/'
        os.makedirs(new_path)
        for i in os.listdir(full_file):
            if m < 260:
                img = Image.open(full_file + '/' + i)
                img_transform = transforms.Compose([
                    transforms.Resize((160, 160)),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                    transforms.ToTensor()
                ])
                img = img_transform(img).unsqueeze(0)
                encoded_images = encoder(img, trigger)
                # encoded_images = (encoded_images + 1) / 2


                a = k * 260 + m
                out = i
                # toPIL = transforms.ToPILImage()
                # encoded_images = toPIL(encoded_images.float().squeeze(0))
                # encoded_images.save('G:/lq/benign/try/test/' + out)
                torchvision.utils.save_image(encoded_images, new_path + out, img.shape[0], normalize=False)
                m = m + 1
            else:
                break
        k = k + 1


if __name__ == "__main__":
    main()
