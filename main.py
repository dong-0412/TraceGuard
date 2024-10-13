import os
import pprint
import argparse
import torch
import pickle           #序列化操作，处理文本，保存文件用的。
import utils
import logging
import sys

from options import *
from model.hidden import Hidden
#from noise_layers.noiser import Noiser
#from noise_argparser import NoiseArgParser

from noise_layers.noiser import get_noiser_model,interpolate_pos_embed

from train import train
import numpy as np
from model import models_vit
from timm.models.layers import trunc_normal_


from torchvision.models import resnet34, resnet18,vgg16,densenet121
from timm.models import create_model
#为了可复现，记得定义一个seed。
def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        print("gpu cuda is available!")
        torch.cuda.manual_seed(0)
    else:
        print("cuda is not available! cpu is available!")
        torch.manual_seed(0)

    torch.cuda.empty_cache()


    parent_parser = argparse.ArgumentParser(description='Training of HiDDeN nets')
    # subparsers = parent_parser.add_subparsers(dest='command', help='Sub-parser for commands')
    # new_run_parser = subparsers.add_parser('new', help='starts a new run')
    new_run_parser = parent_parser
    #####################自监督微调，和后面训练的数据集一样。
########################simclr###############################
    new_run_parser.add_argument('-data', metavar='DIR', default=r'G:\datasets\imagenet\trace_img12_2400',      #G:\datasets\imagenet\FTtrace_img50_2500
                        help='path to dataset')
    new_run_parser.add_argument('--n-views', default=2, type=int, metavar='N',
                        help='Number of views for contrastive learning training.')
    new_run_parser.add_argument('--temperature', default=0.07, type=float,
                        help='softmax temperature (default: 0.07)')
    new_run_parser.add_argument('--out_dim', default=128, type=int,
                        help='feature dimension (default: 128)')

#################################################################



    new_run_parser.add_argument('--command', default='new', type=str, help='The batch size.')
    new_run_parser.add_argument('--learning-rate', default=0.0001, type=int, help='The learning rate.')
    new_run_parser.add_argument('--data-dir', '-d', default=r'G:\datasets\imagenet\trace_img12_2400', type=str,         #G:\datasets\imagenet\trace_img12_2400   G:\datasets\STL10\trace
                                help='The directory where the data is stored.')#E:\bad_pretrain_encoder\SimCLR_backdoor\data F:\Datasets\STL10
    new_run_parser.add_argument('--message', '-m', default=30, type=int, help='The length in bits of the watermark.')#30
    new_run_parser.add_argument('--batch-size', '-b', default=32, type=int, help='The batch size.')#1
    new_run_parser.add_argument('--epochs', '-e', default=50, type=int, help='Number of epochs to run the simulation.')#10
    new_run_parser.add_argument('--name', default='no_FT_vit_b_imagenet_mae', type=str, help='The name of the experiment.')##注意me模型一起改，自监督
    new_run_parser.add_argument('--size', '-s', default=224, type=int,          #160
                                help='The size of the images (images are square so this is height and width).')
    new_run_parser.add_argument('--continue-from-folder', '-c', default='', type=str,
                                help='The folder from where to continue a previous run. Leave blank if you are starting a new experiment.')
    # parser.add_argument('--tensorboard', dest='tensorboard', action='store_true',
    #                     help='If specified, use adds a Tensorboard log. On by default')
    new_run_parser.add_argument('--tensorboard', action='store_true',
                                help='Use to switch on Tensorboard logging.')
    new_run_parser.add_argument('--enable-fp16', dest='enable_fp16', action='store_true',
                                help='Enable mixed-precision training.')
    # nargs='*'表示参数可以设置0个或多个。调用这个参数noise，令参数的值为action的值，NoiseArgParser是初始化函数。
    # new_run_parser.add_argument('--noise', nargs='*', action=NoiseArgParser,
    #                             help="Noise layers configuration. Use quotes when specifying configuration, e.g. 'cropout((0.55, 0.6), (0.55, 0.6))'")
    #new_run_parser.add_argument('__noise-name', default='resnet34', help='Noise model using resnet34 encoder')

    new_run_parser.add_argument('--noiser-weight', default=r'D:\hid\mae-main\result\vit_base\checkpoint-200.pth',
                                type=str, help='weight of Noise model')#'E:\bad_pretrain_encoder\SimCLR_backdoor\runs\benign\epo199.pth'
    new_run_parser.set_defaults(tensorboard=False)
    new_run_parser.set_defaults(enable_fp16=False)

    # continue_parser = subparsers.add_parser('continue', help='Continue a previous run')
    # continue_parser.add_argument('--folder', '-f', required=True, type=str,
    #                              help='Continue from the last checkpoint in this folder.')
    # continue_parser.add_argument('--data-dir', '-d', required=False, type=str,
    #                              help='The directory where the data is stored. Specify a value only if you want to override the previous value.')
    # continue_parser.add_argument('--epochs', '-e', required=False, type=int,
    #                             help='Number of epochs to run the simulation. Specify a value only if you want to override the previous value.')
    # # continue_parser.add_argument('--tensorboard', action='store_true',
    # #                             help='Override the previous setting regarding tensorboard logging.')

    args = parent_parser.parse_args()
    checkpoint = None
    loaded_checkpoint_file_name = None

    if args.command == 'continue':              #改了代码，continue没改
        this_run_folder = args.folder
        options_file = os.path.join(this_run_folder, 'options-and-config.pickle')
        train_options, hidden_config, noise_config = utils.load_options(options_file)
        checkpoint, loaded_checkpoint_file_name = utils.load_last_checkpoint(os.path.join(this_run_folder, 'checkpoints'))
        train_options.start_epoch = checkpoint['epoch'] + 1
        if args.data_dir is not None:
            train_options.train_folder = os.path.join(args.data_dir, 'train')
            train_options.validation_folder = os.path.join(args.data_dir, 'val')
        if args.epochs is not None:
            if train_options.start_epoch < args.epochs:
                train_options.number_of_epochs = args.epochs
            else:
                print(f'Command-line specifies of number of epochs = {args.epochs}, but folder={args.folder} '
                      f'already contains checkpoint for epoch = {train_options.start_epoch}.')
                exit(1)

    else:
        assert args.command == 'new'
        start_epoch = 1
        train_options = TrainingOptions(
            batch_size=args.batch_size,
            number_of_epochs=args.epochs,
            train_folder=args.data_dir,  # args.data_dir注意这里搞个路径。
            me_train_folder=args.data,#args.data_dir注意这里搞个路径。
            #validation_folder=os.path.join(args.data_dir, 'test'),      #这里就用test数据集当validation
            runs_folder=os.path.join('.', r'runs'),
            start_epoch=start_epoch,
            experiment_name=args.name)#初始化训练参数。

        #noise_config = args.noise if args.noise is not None else []
        hidden_config = HiDDenConfiguration(H=args.size, W=args.size,           #图片长宽。
                                            message_length=args.message,learning_rate= args.learning_rate,        #水印长度。
                                            encoder_blocks=4, encoder_channels=64,
                                            decoder_blocks=7, decoder_channels=64,
                                            noise_feature = 768,                #补充
                                            # use_discriminator=False,
                                            use_vgg=False,
                                            discriminator_blocks=3, discriminator_channels=64,
                                            decoder_loss=1,           #用信息损失0.7
                                            encoder_loss=0.0,           #图片损失这里嵌入，估计图片损失
                                            me_decoder_loss=0.1,                #
                                            adversarial_loss=1e-3,
                                            enable_fp16=args.enable_fp16
                                            )

        this_run_folder = utils.create_folder_for_run(train_options.runs_folder, args.name)#保存结果的地方
        with open(os.path.join(this_run_folder, 'options-and-config.pickle'), 'wb+') as f:
            pickle.dump(train_options, f)   #将对象train_options保存到文件f中去。pickle.load读取字符串。
            #pickle.dump(noise_config, f)
            pickle.dump(hidden_config, f)


    logging.basicConfig(level=logging.INFO,
                        format='%(message)s',
                        handlers=[
                            logging.FileHandler(os.path.join(this_run_folder, f'{train_options.experiment_name}.log')),
                            logging.StreamHandler(sys.stdout)#输出信息的handlers
                        ])
    if (args.command == 'new' and args.tensorboard) or \
            (args.command == 'continue' and os.path.isdir(os.path.join(this_run_folder, 'tb-logs'))):
        logging.info('Tensorboard is enabled. Creating logger.')
        from tensorboard_logger import TensorBoardLogger
        tb_logger = TensorBoardLogger(os.path.join(this_run_folder, 'tb-logs'))
    else:
        tb_logger = None            #无可视化

    #noiser = Noiser(args.noise_model_name, device)           #噪声模块。
    noiser = get_noiser_model(args)
    noiser.to(device)

###############################################################################################

    #me_model = resnet18(pretrained=False)####注意me模型跟着noiser动
    me_model = create_model("vit_base_patch16_224",pretrained=False)
                            # pretrained=False,
                            # num_classes=1000,
                            # drop_path_rate=0.1,
                            # global_pool=False,)
    # me_model = models_vit.__dict__["vit_base_patch16"](
    #     num_classes=1000,
    #     drop_path_rate=0.1,
    #     global_pool=False,
    # )
    # model_weight = torch.load(args.noiser_weight)
    # model_weigth = model_weight['model']
    # state_dict = noiser.state_dict()
    # for k in ['head.weight', 'head.bias']:
    #     if k in model_weigth and model_weigth[k].shape != state_dict[k].shape:
    #         print(f"Removing key {k} from pretrained checkpoint")
    #         del model_weigth[k]
    #
    # # interpolate position embedding
    # interpolate_pos_embed(noiser, model_weigth)
    # trunc_normal_(noiser.head.weight, std=2e-5)
    #me_model = densenet121(pretrained=False)
    # me_model_weigth = torch.load(r'G:\pth\simclr_imagenet_r18.pth')
    # me_model_weigth.pop('fc.weight')  # 丢掉FC层，因为输出不同，或_fc.weight
    # me_model_weigth.pop('fc.bias')  # 丢掉FC层，因为输出不同，或_fc.bias
    # me_model.load_state_dict(me_model_weigth,strict=False)
##############################################################################################

    # for name, value in noiser.named_parameters():
    #         value.requires_grad = False
    #         print(name)
    #         print(value.requires_grad)

    print("sum of parameter of noiser:{}".format(utils.get_num_parameters(noiser)))
    print("training parameter of noiser:{}".format(utils.get_num_trainable_parameters(noiser)))
    print("non training parameter of noiser:{}".format(utils.get_num_non_trainable_parameters(noiser)))

    model = Hidden(hidden_config, device, noiser,me_model, tb_logger,args)    #encoder_decoder

    #############################################加载权重#############################################
    model_weigth = torch.load(r"G:\project\hidfp_ztx\HiDFP\runs\base\imagenet\vit_base_mae_imagenet_bs32 2023.03.13--13-07-16\checkpoints\--epoch-60.pyt")
    model.encoder_decoder.encoder.load_state_dict(model_weigth["encoder"])
    model.encoder_decoder.decoder.load_state_dict(model_weigth["decoder"])

    #################################################################################################
    # model = Hidden(hidden_config, device, tb_logger)

    utils.set_parameter_requires_grad(model.encoder_decoder.encoder,requires_grad=False)  # 不放心，在设置一次
    utils.set_parameter_requires_grad(model.encoder_decoder.decoder,requires_grad=False)  # 不放心，在设置一次
    utils.set_parameter_requires_grad(model.encoder_decoder.noiser, requires_grad=True)  # 不放心，在设置一次
    utils.set_parameter_requires_grad(model.encoder_decoder.me_model, requires_grad=True)  # 不放心，在设置一次
    #utils.set_parameter_requires_grad(model.encoder_decoder.me_model, requires_grad=False)  # 不放心，在设置一次
    print("sum of parameter of encoder_decoder:{}".format(utils.get_num_parameters(model.encoder_decoder)))
    print("training parameter of encoder_decoder:{}".format(utils.get_num_trainable_parameters(model.encoder_decoder)))
    print("non training parameter of encoder_decoder:{}".format(utils.get_num_non_trainable_parameters(model.encoder_decoder)))

    # pth = torch.load(r"runs/stl-10 2022.11.08--21-25-19/checkpoints/--epoch-7.pyt")
    # utils.model_from_checkpoint(model, pth)

    if args.command == 'continue':
        # if we are continuing, we have to load the model params
        assert checkpoint is not None
        logging.info(f'Loading checkpoint from file {loaded_checkpoint_file_name}')
        utils.model_from_checkpoint(model, checkpoint)

    logging.info('HiDDeN model: {}\n'.format(model.to_stirng()))#model.to_stirng()模型名字。
    logging.info('Model Configuration:\n')
    logging.info(pprint.pformat(vars(hidden_config)))#pprint分行打印，更漂亮。
    # logging.info('\nNoise configuration:\n')
    # logging.info(pprint.pformat(str(noise_config)))
    logging.info('\nTraining train_options:\n')
    logging.info(pprint.pformat(vars(train_options)))#vars以字典形式返回当前参数的值。
    ##############################唯一的message###########################
    #偶校验
    #one_message = torch.zeros([hidden_config.message_length]).to(device)
    #message = torch.Tensor(np.random.choice([0, 1], (int(hidden_config.message_length/3)))).to(device)
    message = torch.Tensor(np.random.choice([0, 1], (hidden_config.message_length))).to(device)
    # value = 0
    # for i in range(hidden_config.message_length-1):
    #     value += message[i]
    # if (value%2)==0:
    #     for j in range(hidden_config.message_length-1):
    #         one_message[j] = message[j]
    #     one_message[hidden_config.message_length-1] = 0
    # else:
    #     for j in range(hidden_config.message_length-1):
    #         one_message[j] = message[j]
    #     one_message[hidden_config.message_length-1] = 1

    # torch.save(message.to(torch.device('cpu')),'message.pth')

    ########################################################################

    train(model, device, hidden_config, train_options, this_run_folder, tb_logger,args,message)#训练


if __name__ == '__main__':
    main()
