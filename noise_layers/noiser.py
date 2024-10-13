import numpy as np
import torch.nn as nn
# from model.resnet34_g import resnet18
from torchvision.models import resnet34, resnet18,vgg16,densenet121
#from torchvision.models import resnet18
import torch
from noise_layers.identity import Identity
from noise_layers.jpeg_compression import JpegCompression
from noise_layers.quantization import Quantization
from collections import OrderedDict
from model import models_vit
from timm.models.layers import trunc_normal_

# class Noiser(nn.Module):
#     """
#     This module allows to combine different noise layers into a sequential noise module. The
#     configuration and the sequence of the noise layers is controlled by the noise_config parameter.
#     """
#     def __init__(self, noise_layers: list, device):
#         super(Noiser, self).__init__()
#         self.noise_layers = [Identity()]
#         for layer in noise_layers:
#             if type(layer) is str:
#                 if layer == 'JpegPlaceholder':
#                     self.noise_layers.append(JpegCompression(device))
#                 elif layer == 'QuantizationPlaceholder':
#                     self.noise_layers.append(Quantization(device))
#                 else:
#                     raise ValueError(f'Wrong layer placeholder string in Noiser.__init__().'
#                                      f' Expected "JpegPlaceholder" or "QuantizationPlaceholder" but got {layer} instead')
#             else:
#                 self.noise_layers.append(layer)
#         # self.noise_layers = nn.Sequential(*noise_layers)
#
#     def forward(self, encoded_and_cover):
#         random_noise_layer = np.random.choice(self.noise_layers, 1)[0]
#       return random_noise_layer(encoded_and_cover)

# class Noiser(nn.Module):
#     """
#     This module allows to combine different noise layers into a sequential noise module. The
#     configuration and the sequence of the noise layers is controlled by the noise_config parameter.
#     """
#     # def __init__(self, noise_layers: list, device):
#     #     super(Noiser, self).__init__()
#     def __init__(self):
#         super(Noiser, self).__init__()
#         self.model = resnet34()
#
#     def forward(self, encoded_and_cover):
#         feat, logit = self.model(encoded_and_cover)
#         return feat
#



# def get_noiser_model(args):
#     noiser = resnet18(pretrained=False)
#     fc_features = noiser.fc.in_features
#     noiser.fc = nn.Linear(fc_features, 12)
#     noiser.load_state_dict(torch.load(args.noiser_weight), strict=False)
#     # for p in noiser.parameters():
#     #     p.requires_grad = False
#
#     return noiser
def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed


def get_noiser_model(args):
    noiser = resnet34(pretrained=False)
    noiser = models_vit.__dict__["vit_base_patch16"](
        num_classes=1000,
        drop_path_rate=0.1,
        global_pool=False,
    )
    model_weight = torch.load(args.noiser_weight)
    model_weigth = model_weight['model']
    state_dict = noiser.state_dict()
    for k in ['head.weight', 'head.bias']:
        if k in model_weigth and model_weigth[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del model_weigth[k]

    # interpolate position embedding
    interpolate_pos_embed(noiser, model_weigth)
    # load pre-trained model
    msg = noiser.load_state_dict(model_weigth, strict=False)  # 返回缺失的key
    print("权重中缺失以下参数：")
    print(msg)

    # manually initialize fc layer
    trunc_normal_(noiser.head.weight, std=2e-5)


    # noiser = resnet18(pretrained=False)
    # #noiser = densenet121(pretrained=False)
    # #noiser.classifier = nn.Linear(25088, args.out_dim)
    # # a = torch.load(args.noiser_weight)
    # #
    # model_weight = torch.load(args.noiser_weight)
    # #
    # new_model_weight = OrderedDict()
    # for i, (k, v) in enumerate(model_weight['state_dict'].items()):
    #     # print(k)
    #     if "encoder_k" not in k or 'fc' in k:
    #         name = k
    #         print(f"delete:{name}")
    #     else:
    #         name = k[10:]  # 删除名字中的encoder_q
    #         new_model_weight[name] = v
    #         # freeze.append(name)
    #         print(name)
    # #model_weight = model_weight['noiser']
    # # model_weight.pop('classifier.weight')  # 丢掉FC层，因为输出不同，或_fc.weight
    # # model_weight.pop('classifier.bias')  # 丢掉FC层，因为输出不同，或_fc.weight
    # # model_weight.pop('fc.weight')  # 丢掉FC层，因为输出不同，或_fc.weight
    # # model_weight.pop('fc.bias')  # 丢掉FC层，因为输出不同，或_fc.bias
    # msg = noiser.load_state_dict(new_model_weight, strict=False)
    # print(msg)


    # fc_features = noiser.fc.in_features
    # noiser.fc = nn.Linear(fc_features, 12)


    # noiser.load_state_dict(torch.load(args.noiser_weight), strict=False)
    # for p in noiser.parameters():
    #     p.requires_grad = False

    return noiser

    # model = resnet18(pretrained=False).to(device)
    # sd = r'F:\pretrain_backdoor\bad_pretrain_encoder\SimCLR_backdoor\runs\imagenet_adv_half_15_255_r18\epo48.pth'
    # pretrained_dict = torch.load(sd)
    # model.load_state_dict(pretrained_dict, strict=False)

def get_pruning_noiser(args):
    #noiser = resnet18(pretrained=False)
    #path = args.noiser_weight
    noiser = torch.load(args.noiser_weight)
    print(noiser)
    return noiser