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

from scipy.stats import pearsonr
from scipy import spatial

from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import pylab
# import ana.tsne
from ana import tsne

#import pandas as pd

class HookTool:
    def __init__(self):
        self.fea = None

    def hook_fun(self, module, fea_in, fea_out):
        self.fea = fea_in

def get_feas_by_hook(model):
    fea_hooks = []
    for n, m in model.named_modules():
        # if isinstance(m, torch.nn.AdaptiveAvgPool2d):
        # if n == 'layer1.0.conv1' or n == 'layer2.0.conv1' or n == 'layer3.0.conv1' or n == 'fc.0':
        if n == 'fc':
            cur_hook = HookTool()
            m.register_forward_hook(cur_hook.hook_fun)
            fea_hooks.append(cur_hook)

    return fea_hooks


def draw_tsne(clean_img, img, me_img, pth_img):
    print("Run Y = tsne.tsne(X, no_dims, perplexity) to perform t-SNE on your dataset.")
    clean_img_list = [item for sublist in clean_img for item in sublist]
    img_list = [item for sublist in img for item in sublist]
    me_img_list = [item for sublist in me_img for item in sublist]
    pth_img_list = [item for sublist in pth_img for item in sublist]

    X = np.concatenate((np.array(img_list), np.array(me_img_list), np.array(pth_img_list)), axis=0)
    print(len(X))
    # X = np.array(X)
    # labels = np.loadtxt("txt/6class_withback.txt")

    Y = tsne.tsne_p(X, 2, 50, 20.0)

    pylab.scatter(Y[0:len(img_list), 0], Y[0:len(img_list), 1], 20, c='b', label="class0")
    pylab.scatter(Y[len(img_list):2 * len(img_list), 0], Y[len(img_list):2 * len(img_list), 1], 20, c='g', label="class1")
    pylab.scatter(Y[2 * len(img_list):3 * len(img_list), 0], Y[2 * len(img_list):3 * len(img_list), 1], 20, c='r', label="class2")

    # plt.scatter(Y[:, 0], Y[:, 1], 20,  label=labels)
    # pylab.legend(loc="best")
    pylab.show()

    pylab.close()
    X2 = np.concatenate((np.array(clean_img_list), np.array(img_list)), axis=0)
    Y2 = tsne.tsne_p(X2, 2, 50, 20.0)

    pylab.scatter(Y2[0:len(img_list), 0], Y2[0:len(img_list), 1], 20, c='r', label="class0")
    pylab.scatter(Y2[len(img_list):2 * len(img_list), 0], Y2[len(img_list):2 * len(img_list), 1], 20, c='g', label="class1")
    pylab.show()


def draw_pcc(clean_img, clean_me_img, clean_pth_img, img, me_img, pth_img):
    clean_img_list = [item for sublist in clean_img for item in sublist]
    clean_me_img_list = [item for sublist in clean_me_img for item in sublist]
    clean_pth_img_list = [item for sublist in clean_pth_img for item in sublist]

    img_list = [item for sublist in img for item in sublist]
    me_img_list = [item for sublist in me_img for item in sublist]
    pth_img_list = [item for sublist in pth_img for item in sublist]

    # X = np.concatenate((np.array(img_list), np.array(me_img_list), np.array(pth_img_list)), axis=0)
    # print(len(X))

    clean_list = []
    clean_me_list = []
    clean_pth_list = []

    me_list = []
    pth_list = []

    for i in range(len(img_list)):
        # # 皮尔逊相似度
        # pcc_clean = pearsonr(np.array(clean_img_list[i]), np.array(img_list[i]))
        # pcc_me = pearsonr(np.array(img_list[i]), np.array(me_img_list[i]))
        # pcc_pth = pearsonr(np.array(img_list[i]), np.array(pth_img_list[i]))

        # clean_list.append(pcc_clean[0])
        # me_list.append(pcc_me[0])
        # pth_list.append(pcc_pth[0])

        # cos相似度
        cos_clean = 1 - spatial.distance.cosine(np.array(clean_img_list[i]), np.array(img_list[i]))
        cos_clean_me = 1 - spatial.distance.cosine(np.array(clean_me_img_list[i]), np.array(me_img_list[i]))
        cos_clean_pth = 1 - spatial.distance.cosine(np.array(clean_pth_img_list[i]), np.array(pth_img_list[i]))

        cos_me = 1 - spatial.distance.cosine(np.array(img_list[i]), np.array(me_img_list[i]))
        cos_pth = 1 - spatial.distance.cosine(np.array(img_list[i]), np.array(pth_img_list[i]))
##########################################
        clean_list.append(cos_clean)
        clean_me_list.append(cos_clean_me)
        clean_pth_list.append(cos_clean_pth)

        me_list.append(cos_me)
        pth_list.append(cos_pth)

    clean_plt = np.array(clean_list)
    clean_me_plt = np.array(clean_me_list)
    clean_pth_plt = np.array(clean_pth_list)

    me_plt = np.array(me_list)
    pth_plt = np.array(pth_list)

    # plt.hist(clean_list, bins=50, density=True, color='r')

    plt.hist(clean_plt, bins=50, density=True, color='r', alpha=1)
    plt.hist(clean_me_plt, bins=50, density=True, color='g', alpha=1)
    plt.hist(clean_pth_plt, bins=50, density=True, color='b', alpha=1)

    # plt.hist(me_plt[0], bins=50, density=True, color='r')
    # plt.hist(me_plt[0], bins=50, density=True, color='r')
    # plt.hist(pth_plt[0], bins=50, density=True, color='g')


    plt.show()


def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    parent_parser = argparse.ArgumentParser(description='Training of HiDDeN nets')
    new_run_parser = parent_parser
    new_run_parser.add_argument('--test', '-d', default=r'E:\bad_pretrain_encoder\SimCLR_backdoor\data\test', type=str,
                                help='The directory where the data is stored.')
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
    loss_me = 0
    loss_pth = 0
    sum = 0

    # for name, value in model.named_parameters():
    #         value.requires_grad = False
    #         print(name)
    #         print(value.requires_grad)

    unloader = transforms.ToPILImage()

    clean_image_emb = []
    clean_me_image_emb = []
    clean_pth_image_emb = []

    image_emb = []
    me_image_emb = []
    pth_image_emb = []

    for image, _ in test_loader:
        # 原始图像可视化
        # image_show = (image + 1) / 2
        # image_show = torch.clip(image_show, 0, 1)
        # plt.imshow(unloader(image_show[0].squeeze(0)))
        # plt.show()

        image = image.to(device)
        message = torch.Tensor(np.random.choice([0, 1], (image.shape[0], hidden_config.message_length))).to(device)

        #——————版权模型———————

        noiser = resnet18(pretrained=False)
        noiser.load_state_dict(torch.load(args.noiser_weight), strict=False)
        noiser.to(device)

        model = Hidden(hidden_config, device, noiser, None)  # encoder_decoder
        model.encoder_decoder.train()
        pth = torch.load(r'E:\HiD\HiDFP\runs\imagenet 2022.10.16--17-01-46\checkpoints\--epoch-50.pyt')
        utils.model_from_checkpoint(model, pth)

        losses, _ = model.validate_on_batch([image, message])  # 训练一个batch
        loss += losses['bitwise-error  ']
        image_emb.append(_[1].cpu().numpy())

        #干净图像
        fea_hooks = get_feas_by_hook(noiser)
        clean_out = noiser(image)
        clean_emb = fea_hooks[0].fea[0]
        clean_image_emb.append(clean_emb.detach().cpu().numpy())

        print("——————版权模型———————")
        print("bit_error:{}".format(losses['bitwise-error  ']))


        #——————ME模型———————
        noiser = resnet18(pretrained=False)
        noiser.load_state_dict(torch.load(r'E:\HiD\sim_test\resnet_34_imagenet\me_epo99.pth'), strict=False)
        noiser.to(device)

        model = Hidden(hidden_config, device, noiser, None)  # encoder_decoder
        model.encoder_decoder.train()
        pth = torch.load(r'E:\HiD\HiDFP\runs\imagenet 2022.10.16--17-01-46\checkpoints\--epoch-50.pyt')
        utils.model_from_checkpoint(model, pth)

        losses_me, _me = model.validate_on_batch([image, message])  # 训练一个batch
        loss_me += losses_me['bitwise-error  ']
        me_image_emb.append(_me[1].cpu().numpy())

        #干净图像
        fea_hooks = get_feas_by_hook(noiser)
        clean_me_out = noiser(image)
        clean_me_emb = fea_hooks[0].fea[0]
        clean_me_image_emb.append(clean_me_emb.detach().cpu().numpy())

        print("——————ME模型———————")
        print("bit_error:{}".format(losses_me['bitwise-error  ']))


        # ——————监督预训练模型———————
        noiser = resnet18(pretrained=True)
        # noiser.load_state_dict(torch.load('E:\HiD\sim_test\imagenet_me\me_epo99.pth'), strict=False)
        noiser.to(device)

        model = Hidden(hidden_config, device, noiser, None)  # encoder_decoder
        model.encoder_decoder.train()
        pth = torch.load(r'E:\HiD\HiDFP\runs\imagenet 2022.10.16--17-01-46\checkpoints\--epoch-50.pyt')
        utils.model_from_checkpoint(model, pth)

        losses_pth, _pth = model.validate_on_batch([image, message])  # 训练一个batch
        loss_pth += losses_pth['bitwise-error  ']
        pth_image_emb.append(_pth[1].cpu().numpy())

        #干净图像
        fea_hooks = get_feas_by_hook(noiser)
        clean_pth_out = noiser(image)
        clean_pth_emb = fea_hooks[0].fea[0]
        clean_pth_image_emb.append(clean_pth_emb.detach().cpu().numpy())

        print("——————监督预训练模型———————")
        print("bit_error:{}".format(losses_pth['bitwise-error  ']))


        sum += 1

        # 编码图像可视化
        # output = _[0].cpu()
        # output = (output + 1) / 2
        # output = torch.clip(output, 0, 1)
        #
        # plt.close()
        #
        # plt.imshow(unloader(output[0].squeeze(0)))
        # plt.show()

    print("bit_error_avg:{}".format(loss/sum))

    # draw_tsne(clean_image_emb, image_emb, me_image_emb, pth_image_emb)
    draw_pcc(clean_image_emb, clean_me_image_emb, clean_pth_image_emb, image_emb, me_image_emb, pth_image_emb)




if __name__ == '__main__':
    main()
