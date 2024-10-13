import os
import time
import torch
import numpy as np
import utils
import logging
from collections import defaultdict

from options import *
from model.hidden import Hidden
from average_meter import AverageMeter
from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast

def train(model: Hidden,
          device: torch.device,
          hidden_config: HiDDenConfiguration,
          train_options: TrainingOptions,
          this_run_folder: str,
          tb_logger,
          args,
          one_message):           #tb_logger无可视化。
    """
    Trains the HiDDeN model
    :param model: The model
    :param device: torch.device object, usually this is GPU (if avaliable), otherwise CPU.
    :param hidden_config: The network configuration
    :param train_options: The training settings
    :param this_run_folder: The parent folder for the current training run to store training artifacts/results/logs.
    :param tb_logger: TensorBoardLogger object which is a thin wrapper for TensorboardX logger.
                Pass None to disable TensorboardX logging
    :return:
    """
    torch.save(one_message.to(torch.device('cpu')), os.path.join(this_run_folder, 'messages.th'))

    train_data,me_train_data = utils.get_data_loaders(hidden_config, train_options)
# ########################simclr_train_dataset##############################################
#
#     simclr_dataset = ContrastiveLearningDataset(args.data)
#
#     simclr_train_dataset = simclr_dataset.get_dataset(args.data, args.n_views)
#
#     simclr_train_loader = torch.utils.data.DataLoader(
#         simclr_train_dataset, batch_size=args.batch_size, shuffle=True,
#         num_workers=0, pin_memory=True, drop_last=True)
# #########################################################################################


    file_count = len(train_data.dataset)
    if file_count % train_options.batch_size == 0:
        steps_in_epoch = file_count // train_options.batch_size
    else:
        steps_in_epoch = file_count // train_options.batch_size + 1

    print_each = 16
    images_to_save = 5
    saved_images_size = (160, 160)
    message = torch.zeros([train_options.batch_size, hidden_config.message_length]).to(device)
    for i in range(message.shape[0]):
        message[i] = one_message
    # for i in range(message.shape[0]):
    #     for j in range(one_message.shape[0]):
    #         message[i][j*3] = one_message[j]
    #         message[i][(j * 3)+1] = one_message[j]
    #         message[i][(j * 3)+2] = one_message[j]
    print(message)

    for epoch in range(train_options.start_epoch, train_options.number_of_epochs + 1):
        logging.info('\nStarting epoch {}/{}'.format(epoch, train_options.number_of_epochs))
        logging.info('Batch size = {}\nSteps in epoch = {}'.format(train_options.batch_size, steps_in_epoch))
        training_losses = defaultdict(AverageMeter)#存在默认值的字典。
        simclr_losses = defaultdict(AverageMeter)
        me_losses = defaultdict(AverageMeter)
        epoch_start = time.time()
        step = 1
        # utils.set_parameter_requires_grad(model.encoder_decoder.noiser, requires_grad=True)  # 不放心，在设置一次
        # utils.set_parameter_requires_grad(model.encoder_decoder.me_model, requires_grad=False)  # 不放心，在设置一次
        # for images, _ in tqdm(simclr_train_loader):
        #     images = torch.cat(images, dim=0)
        #     images = images.cuda()
        #     #######这里的batch_size与其他的不一样。
        #     simclr_message = torch.Tensor(np.random.choice([0, 1], (images.shape[0], hidden_config.message_length))).to(device)
        #
        #     #with autocast(enabled=self.args.fp16_precision):
        #     simclr_loss = model.train_simclr([images,simclr_message])
        #     for name, loss in simclr_loss.items():
        #         simclr_losses[name].update(loss)
        #
        #     if step % print_each == 0 or step == steps_in_epoch:
        #         logging.info(
        #             'Epoch: {}/{} Step: {}/{}'.format(epoch, train_options.number_of_epochs, step, steps_in_epoch))
        #         utils.log_progress(simclr_losses)
        #         logging.info('-' * 40)
        #         step += 1
        utils.set_parameter_requires_grad(model.encoder_decoder.noiser, requires_grad=False)  # 不放心，在设置一次
        utils.set_parameter_requires_grad(model.encoder_decoder.me_model, requires_grad=True)  # 不放心，在设置一次
        for image, _ in me_train_data:
            image = image.to(device)
            #message = torch.Tensor(np.random.choice([0, 1], (image.shape[0], hidden_config.message_length))).to(device)
            me_loss = model.train_me([image, message])#训练一个batch

            for name, loss in me_loss.items():
                me_losses[name].update(loss)#更新损失字典
            if step % print_each == 0 or step == steps_in_epoch:
                logging.info(
                    'Epoch: {}/{} Step: {}/{}'.format(epoch, train_options.number_of_epochs, step, steps_in_epoch))
                utils.log_progress(me_losses)
                logging.info('-' * 40)
            step += 1

        utils.set_parameter_requires_grad(model.encoder_decoder.noiser, requires_grad=True)  # 不放心，在设置一次
        utils.set_parameter_requires_grad(model.encoder_decoder.me_model, requires_grad=False)  # 不放心，在设置一次

        for image, _ in train_data:
            image = image.to(device)
            #message = torch.Tensor(np.random.choice([0, 1], (image.shape[0], hidden_config.message_length))).to(device)
            step1_losses, _ = model.train_on_batch([image, message])#训练一个batch

            for name, loss in step1_losses.items():
                training_losses[name].update(loss)#更新损失字典
            if step % print_each == 0 or step == steps_in_epoch:
                logging.info(
                    'Epoch: {}/{} Step: {}/{}'.format(epoch, train_options.number_of_epochs, step, steps_in_epoch))
                utils.log_progress(training_losses)
                logging.info('-' * 40)
            step += 1
            #print(f"simclr_loss")



        train_duration = time.time() - epoch_start
        logging.info('Epoch {} training duration {:.2f} sec'.format(epoch, train_duration))
        logging.info('-' * 40)
        utils.save_checkpoint(model, epoch, os.path.join(this_run_folder, 'checkpoints'),
                              training_losses['bitwise-error'].avg)
        utils.write_losses(os.path.join(this_run_folder, 'train.csv'), training_losses, epoch, train_duration)

        # if epoch % 2 == 0:
        #     saveDir = r'E:\HiD\HiDFP\result\imagenet'
        #     utils.save_checkpoint(model, epoch, saveDir, training_losses['bitwise-error'].avg)



        # if tb_logger is not None:
        #     tb_logger.save_losses(training_losses, epoch)
        #     tb_logger.save_grads(epoch)
        #     tb_logger.save_tensors(epoch)
        #
        # #########################验证过程#############################
        # # 感觉没干什么验证，想好在写一下。
        # first_iteration = True
        # validation_losses = defaultdict(AverageMeter)
        # logging.info('Running validation for epoch {}/{}'.format(epoch, train_options.number_of_epochs))
        #
        # for image, _ in train_data:     #就用训练集来测试
        #     image = image.to(device)
        #     # message_val = torch.Tensor(np.random.choice([0, 1], (image.shape[0], hidden_config.message_length))).to(device)
        #     # for i in range(message_val.shape[0]):
        #     #     if message_val[i].equal(one_message):
        #     #         message_val[i] = message_val[i-1]       #可能会出错，但我赌它不出
        #     # message_val = torch.zeros([train_options.batch_size, int(hidden_config.message_length/3)]).to(device)
        #     # for i in range(message_val.shape[0]):
        #     #     message_val[i] = one_message
        #
        #     losses, (encoded_images, noised_images, decoded_messages) = model.validate_on_batch([image,message])
        #     for name, loss in losses.items():
        #         validation_losses[name].update(loss)
        #     if first_iteration:
        #         if hidden_config.enable_fp16:
        #             image = image.float()
        #             encoded_images = encoded_images.float()
        #         utils.save_images(image.cpu()[:images_to_save, :, :, :],
        #                           encoded_images[:images_to_save, :, :, :].cpu(),
        #                           epoch,
        #                           os.path.join(this_run_folder, 'images'), resize_to=saved_images_size)
        #         first_iteration = False
        #
        # utils.log_progress(validation_losses)
        # logging.info('-' * 40)
        # # utils.save_checkpoint(model, train_options.experiment_name, epoch, os.path.join(this_run_folder, 'checkpoints'))
        # utils.save_checkpoint(model, epoch, os.path.join(this_run_folder, 'checkpoints'), training_losses['bitwise-error'].avg)
        # utils.write_losses(os.path.join(this_run_folder, 'validation.csv'), validation_losses, epoch,
        #                    time.time() - epoch_start)