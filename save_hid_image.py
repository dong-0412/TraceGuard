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

from noise_layers.noiser import get_noiser_model

from train import train

from torchvision import datasets, transforms
import torch.nn.functional as F
from torchvision import datasets, transforms
import torchvision.utils
from PIL import Image

import matplotlib.pyplot as plt

from options import TrainingOptions

# def save_images(original_images, watermarked_images, folder, resize_to=None):
# def save_images(original_images, filename, resize_to=None):
#     # images = original_images[:original_images.shape[0], :, :, :].cpu()
#     # watermarked_images = watermarked_images[:watermarked_images.shape[0], :, :, :].cpu()
#     #
#     # # scale values to range [0, 1] from original range of [-1, 1]
#     # images = (images + 1) / 2#这一步是因为图像进行了归一化的结果。要变回从0-1
#     # watermarked_images = (watermarked_images + 1) / 2
#     #
#     # if resize_to is not None:
#     #     images = F.interpolate(images, size=resize_to)
#     #     watermarked_images = F.interpolate(watermarked_images, size=resize_to)
#     #
#     # stacked_images = torch.cat([images, watermarked_images], dim=0)
#     # #filename = os.path.join(folder, 'epoch-{}.png'.format(epoch))
#     # torchvision.utils.save_image(stacked_images, folder, original_images.shape[0], normalize=False)#这个函数省略了tensor到numpy的过程。
#     # print("save successfully")
#
#     images = original_images[:original_images.shape[0], :, :, :].cpu()
#     # watermarked_images = watermarked_images[:watermarked_images.shape[0], :, :, :].cpu()
#
#     # scale values to range [0, 1] from original range of [-1, 1]
#     images = (images + 1) / 2
#     # watermarked_images = (watermarked_images + 1) / 2
#
#     # if resize_to is not None:
#     #     images = F.interpolate(images, size=resize_to)
#     #     watermarked_images = F.interpolate(watermarked_images, size=resize_to)
#
#     # stacked_images = torch.cat([images, watermarked_images], dim=0)
#     # filename = os.path.join(folder )
#
#     for i in range(images.shape[0]):
#         save_dirs = filename +
#         torchvision.utils.save_image(images[:i, :, :, :], save_dirs)
#     # torchvision.utils.save_image(stacked_images, filename)


def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    parent_parser = argparse.ArgumentParser(description='Training of HiDDeN nets')
    new_run_parser = parent_parser
    new_run_parser.add_argument('--test', '-d', default=r'E:\bad_pretrain_encoder\SimCLR_backdoor\data\test', type=str,
                                help='The directory where the data is stored.')#'E:\bad_pretrain_encoder\SimCLR_backdoor\data\test''F:\Datasets\STL10\test'
    new_run_parser.add_argument('--message', '-m', default=30, type=int, help='The length in bits of the watermark.')#30
    new_run_parser.add_argument('--batch-size', '-b', default=32, type=int, help='The batch size.')#1
    new_run_parser.add_argument('--size', '-s', default=224, type=int,          #160
                                help='The size of the images (images are square so this is height and width).')
    new_run_parser.add_argument('--noiser-weight',
                                default=r'E:\bad_pretrain_encoder\SimCLR_backdoor\runs\benign\epo199.pth',
                                type=str, help='weight of Noise model')

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

    noiser = resnet18(pretrained=True)
    noiser.load_state_dict(torch.load(args.noiser_weight), strict=False)
    noiser.to(device)

    model = Hidden(hidden_config, device, noiser, None)    #encoder_decoder
    model.encoder_decoder.train()

    # model = Hidden(hidden_config, device, tb_logger)
    # pth = torch.load(r"runs/imagenet 2022.10.16--17-01-46/checkpoints/--epoch-28.pyt")
    # utils.model_from_checkpoint(model, pth)
    # model.encoder_decoder.noiser= get_noiser_model(args)
    # model.encoder_decoder.noiser.load_state_dict(torch.load(args.noiser_weight), strict=False)

    save_dir = r'E:\HiD\HiDFP\output_image/'

    # data_transforms = {
    #     'train': transforms.Compose([
    #         transforms.RandomCrop((hidden_config.H, hidden_config.W), pad_if_needed=True),
    #         transforms.ToTensor(),
    #         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    #     ]),
    #     'test': transforms.Compose([
    #         transforms.CenterCrop((hidden_config.H, hidden_config.W)),
    #         transforms.ToTensor(),
    #         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    #     ])
    # }
    #
    # test_images = datasets.ImageFolder(args.test, data_transforms['test'])
    # test_loader = torch.utils.data.DataLoader(test_images, batch_size=args.batch_size,
    #                                                 shuffle=False, num_workers=0)
    # loss = 0
    # sum = 0

    # for name, value in model.named_parameters():
    #         value.requires_grad = False
    #         print(name)
    #         print(value.requires_grad)


    # for image, _ in test_loader:
    #     image = image.to(device)
    #     message = torch.Tensor(np.random.choice([0, 1], (image.shape[0], hidden_config.message_length))).to(device)
    #     losses, _ = model.validate_on_batch([image, message])  # 训练一个batch
    #     loss += losses['bitwise-error  ']
    #     sum += 1
    #
    #     # a = _[0]
    #
    #
    #     save_images(_[0].cpu()[:, :, :, :], save_dir, resize_to=None)

    #     print("bit_error:{}".format(losses['bitwise-error  ']))
    #
    # print("bit_error_avg:{}".format(loss/sum))

    # root = "I:/lq/hiden_dataset/GTSRB/benign/test/"

    def get_data_loaders(hidden_config):
        """ Get torch data loaders for training and validation. The data loaders take a crop of the image,
        transform it into tensor, and normalize it."""
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomCrop((hidden_config.H, hidden_config.W), pad_if_needed=True),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        }

        train_images = datasets.ImageFolder(args.test, data_transforms['train'])
        train_loader = torch.utils.data.DataLoader(train_images, batch_size=args.batch_size, shuffle=False,
                                                   num_workers=0)
        return train_loader

    train_data = get_data_loaders(hidden_config)

    for image, _ in train_data:
        image = image.to(device)
        message = torch.Tensor(np.random.choice([0, 1], (image.shape[0], hidden_config.message_length))).to(device)
        losses, _ = model.train_on_batch([image, message])  # 训练一个batch
        print(losses)

        output = _[0].cpu()
        output = (output + 1) / 2

        unloader = transforms.ToPILImage()

        plt.imshow(unloader(output[0].squeeze(0)))
        plt.show()



    k = 0
    for file in os.listdir(args.test):
        m = 0
        full_file = os.path.join(args.test, file)
        new_path = save_dir + file +'/'
        # os.makedirs(new_path)
        for i in os.listdir(full_file):
            if m < 260:
                img = Image.open(full_file + '/' + i)
                # img.convert('RGB')
                plt.imshow(img)
                plt.show()
                img_transform = transforms.Compose([
                    transforms.CenterCrop((hidden_config.H, hidden_config.W)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ])
                img = img_transform(img).unsqueeze(0).to(device)
                message = torch.Tensor(np.random.choice([0, 1], (img.shape[0], hidden_config.message_length))).to(device)
                losses, _ = model.train_on_batch([img, message])  # 训练一个batch
                print(losses)

                out = i
                output = _[0].cpu()
                output = (output + 1) / 2

                unloader = transforms.ToPILImage()
                plt.close()

                plt.imshow(unloader(output.squeeze(0)))
                plt.show()

                # torchvision.utils.save_image(output, new_path + out, normalize=False)
                m = m + 1
            else:
                break
        k = k + 1
        print(k)


if __name__ == '__main__':
    main()
