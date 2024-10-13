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

from noise_layers.noiser import get_noiser_model
import torch.nn.functional as F
from PIL import Image
import datetime
from train import train
import torchvision.utils
from torchvision import datasets, transforms


def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    parent_parser = argparse.ArgumentParser(description='Training of HiDDeN nets')
    new_run_parser = parent_parser
    new_run_parser.add_argument('--test', '-d', default=r'E:\zyj\datasets\imagenet\train', type=str,
                                help='The directory where the data is stored.')
    new_run_parser.add_argument('--message', '-m', default=10, type=int, help='The length in bits of the watermark.')#30
    new_run_parser.add_argument('--batch-size', '-b', default=32, type=int, help='The batch size.')#1
    new_run_parser.add_argument('--size', '-s', default=160, type=int,          #160
                                help='The size of the images (images are square so this is height and width).')
    new_run_parser.add_argument('--noiser-weight', default=r'E:\zyj\project\HiDDeN\pth\imagenet\resnet18_32_0.01_epo24_cl86.70%_back.pth',
                                type=str, help='weight of Noise model')

    new_run_parser.add_argument('--encoder-image-path',
                                default=r'E:\zyj\datasets\ImageNet_HiDDeN\train',
                                type=str, help='image path after encoder')

    args = parent_parser.parse_args()

    hidden_config = HiDDenConfiguration(H=args.size, W=args.size,           #图片长宽。
                                        message_length=args.message, learning_rate=0.001,       #水印长度。
                                        encoder_blocks=4, encoder_channels=64,
                                        decoder_blocks=7, decoder_channels=64,
                                        noise_feature = 512,                #补充
                                        use_discriminator=False,
                                        use_vgg=False,
                                        discriminator_blocks=3, discriminator_channels=64,
                                        decoder_loss=1,
                                        encoder_loss=0.7,
                                        adversarial_loss=1e-3,
                                        enable_fp16=False
                                        )

    noiser = get_noiser_model(args)
    noiser.to(device)

    model = Hidden(hidden_config, device, noiser, None)    #encoder_decoder
    # model = Hidden(hidden_config, device, tb_logger)
    pth = torch.load(r"E:\zyj\project\HiDDeN\result\imagenet\HiDDeN_32_epoch50_training_loss0.21%.pth")
    utils.model_from_checkpoint(model, pth)
    #model.encoder_decoder.noiser= get_noiser_model(args)
    model.encoder_decoder.noiser.load_state_dict(torch.load(args.noiser_weight))

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
    # loss = 0
    # sum = 0
    j = 0
    for image, label in test_loader:
        image = image.to(device)
        message = torch.Tensor(np.random.choice([0, 1], (image.shape[0], hidden_config.message_length))).to(device)
        tensor = model.encoder_decoder.encoder(image, message)
        for i in range(args.batch_size):
            folder = os.path.join(args.encoder_image_path, r'{}\batch-{}.png'.format(label[i], j))
            watermarked_images = tensor[:tensor.shape[0], :, :, :].cpu()

            # scale values to range [0, 1] from original range of [-1, 1]
            #images = (images + 1) / 2  # 这一步是因为图像进行了归一化的结果。要变回从0-1
            watermarked_images = (watermarked_images + 1) / 2

            watermarked_images = F.interpolate(watermarked_images, size=(224, 224))

            #stacked_images = torch.cat([images, watermarked_images], dim=0)
            # filename = os.path.join(folder, 'epoch-{}.png'.format(epoch))
            torchvision.utils.save_image(watermarked_images[i], folder,
                                         normalize=False)  # 这个函数省略了tensor到numpy的过程。
            #print("save successfully")
            #utils.save_images(image[i].cpu(), tensor[i].cpu(), folder, resize_to=(224, 224))
            j = j+1
        print("batch successfully!!!")
    #print("successfully!!!")
        # imgs = image.permute(0,2,3,1)
        # images = imgs.cpu().detach().numpy()
        # for i in range(args.batch_size):
        #     res = images[i]
        #     img = Image.fromarray(np.uint8(res)).convert("RGB")
        #     #通过时间命名存储结果
        #     #timestamp = datetime.datetime.now().strftime("%M-%S")
        #     savepath = args.encoder_image_path + "/" +  "{}.jpg".format(str(j))
        #     j = j +1
        #     img.save(savepath)
        #     print("第{}张图片".format(str(j)))
        # losses, _ = model.validate_on_batch([image, message])  # 训练一个batch
        # loss += losses['bitwise-error  ']
        # sum += 1
        #print("bit_error:{}".format(losses['bitwise-error  ']))

    #print("bit_error_avg:{}".format(loss/sum))
    print("successfully!!!!")

if __name__ == '__main__':
    main()
