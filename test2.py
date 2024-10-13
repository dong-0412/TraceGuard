import os
import pprint
import argparse
import torch
import pickle           #序列化操作，处理文本，保存文件用的。
import utils
import logging
import sys
import numpy as np
from options import *
from model.hidden import Hidden
from torchvision.models import resnet34, resnet18

from timm.models import create_model
import torch.nn as nn

from noise_layers.noiser import get_noiser_model

from train import train

from torchvision import datasets, transforms


def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    parent_parser = argparse.ArgumentParser(description='Training of HiDDeN nets')
    new_run_parser = parent_parser
    new_run_parser.add_argument('--test', '-d', default=r'test_data/', type=str,
                                help='The directory where the data is stored.')#'F:\Datasets\Imagenet\Imagenet12_half\test''E:\bad_pretrain_encoder\SimCLR_backdoor\data\test' 'F:\Datasets\STL10\test'
    new_run_parser.add_argument('--message', '-m', default=30, type=int, help='The length in bits of the watermark.')#30
    new_run_parser.add_argument('--batch-size', '-b', default=32, type=int, help='The batch size.')#1
    new_run_parser.add_argument('--size', '-s', default=224, type=int,          #160
                                help='The size of the images (images are square so this is height and width).')
    new_run_parser.add_argument('--noiser-weight',
                                # default=r'E:\HiD\SimCLR\runs\ftal_imagenet12\epo 5.pth',
                                default=r'E:\HiD\sim_test\r18_vitb_imagenet\me_epo90.pth',
                                type=str, help='weight of Noise model')#'E:\bad_pretrain_encoder\SimCLR_backdoor\runs\benign\epo199.pth' 'F:\pretrain_backdoor\SimCLR\runs\change\benign_r18_adam_b50_lr0.0003\epo200.pth'

    # 'E:\bad_pretrain_encoder\SimCLR_backdoor\runs\benign\epo199.pth'
    args = parent_parser.parse_args()

    hidden_config = HiDDenConfiguration(H=args.size, W=args.size,           #图片长宽。
                                        message_length=args.message, learning_rate=0.001,       #水印长度。
                                        encoder_blocks=4, encoder_channels=64,
                                        decoder_blocks=7, decoder_channels=64,
                                        noise_feature = 512,                #补充
                                        # use_discriminator=False,
                                        use_vgg=False,
                                        discriminator_blocks=3, discriminator_channels=64,
                                        decoder_loss=1,
                                        encoder_loss=0.7,
                                        adversarial_loss=1e-3,
                                        enable_fp16=False
                                        )

    # noiser = get_noiser_model(args)
    # noiser.to(device)

    noiser = create_model("vit_base_patch32_224", pretrained=False).to(device)
    noiser.head = nn.Linear(768,512).to(device)

    # noiser = resnet18(pretrained=False)
    model_weight = torch.load(args.noiser_weight)
    # model_weight.pop('fc.weight')  # 丢掉FC层，因为输出不同，或_fc.weight
    # model_weight.pop('fc.bias')  # 丢掉FC层，因为输出不同，或_fc.bias
    noiser.load_state_dict(model_weight, strict=False)
    noiser.to(device)

    model = Hidden(hidden_config, device, noiser, None)    #encoder_decoder
    model.encoder_decoder.train()

    # model = Hidden(hidden_config, device, tb_logger)
    pth = torch.load(r"runs/stl-10 2022.10.22--21-46-34/checkpoints/--epoch-100.pyt")
    utils.model_from_checkpoint(model, pth)#runs/imagenet 2022.10.16--17-01-46/checkpoints/--epoch-28.pyt  runs/stl-10 2022.11.08--21-25-19/checkpoints/--epoch-7.pyt
    # model.encoder_decoder.noiser= get_noiser_model(args)
    # model.encoder_decoder.noiser.load_state_dict(torch.load(args.noiser_weight), strict=False)

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomCrop((hidden_config.H, hidden_config.W), pad_if_needed=True),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        'test': transforms.Compose([
            transforms.CenterCrop((hidden_config.H, hidden_config.W)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    }

    test_images = datasets.ImageFolder(args.test, data_transforms['test'])
    test_loader = torch.utils.data.DataLoader(test_images, batch_size=args.batch_size,
                                                    shuffle=False, num_workers=0)
    loss = 0
    sum = 0

    # for name, value in model.named_parameters():
    #         value.requires_grad = False
    #         print(name)
    #         print(value.requires_grad)


    for image, _ in test_loader:
        image = image.to(device)
        message = torch.Tensor(np.random.choice([0, 1], (image.shape[0], hidden_config.message_length))).to(device)
        losses, _ = model.validate_on_batch([image, message])  # 训练一个batch
        loss += losses['bitwise-error  ']
        sum += 1
        print("bit_error:{}".format(losses['bitwise-error  ']))

    print("bit_error_avg:{}".format(loss/sum))


if __name__ == '__main__':
    main()
