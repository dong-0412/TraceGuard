import numpy as np
import torch
import torch.nn as nn

from options import HiDDenConfiguration
from model.discriminator import Discriminator
from model.encoder_decoder import EncoderDecoder
from vgg_loss import VGGLoss
#from noise_layers.noiser import Noiser
import torch.nn.functional as F


class Hidden:
    def __init__(self, configuration: HiDDenConfiguration, device: torch.device, noiser,me_model, tb_logger,args):
        """
        :param configuration: Configuration for the net, such as the size of the input image, number of channels in the intermediate layers, etc.
        :param device: torch.device object, CPU or GPU
        :param noiser: Object representing stacked noise layers.
        :param tb_logger: Optional TensorboardX logger object, if specified -- enables Tensorboard logging
        """
        super(Hidden, self).__init__()

        self.encoder_decoder = EncoderDecoder(configuration, noiser,me_model).to(device)
        # self.discriminator = Discriminator(configuration).to(device)                #鉴别器
        self.optimizer_enc_dec = torch.optim.Adam(self.encoder_decoder.parameters(),lr=configuration.learning_rate)#默认学习率是0.001
        # self.optimizer_discrim = torch.optim.Adam(self.discriminator.parameters(),lr=configuration.learning_rate)

        if configuration.use_vgg:
            self.vgg_loss = VGGLoss(3, 1, False)
            self.vgg_loss.to(device)
        else:
            self.vgg_loss = None                    #不用vgg

        self.config = configuration
        self.device = device

        self.bce_with_logits_loss = nn.BCEWithLogitsLoss().to(device)#模型输出（没经过sigmoid)与标签计算损失。（二分类问题）
        self.mse_loss = nn.MSELoss().to(device)         #均方损失函数。（检测图片损失。）
        self.criterion = torch.nn.CrossEntropyLoss().to(device)
        self.me_loss = nn.CosineEmbeddingLoss(margin=0.0, size_average=None, reduce=None, reduction='mean')

##########################################sinclr####################################\
        #self.batch_size = args.batch_size
        self.args = args

        # # Defined the labels used for training the discriminator/adversarial loss
        # self.cover_label = 1
        # self.encoded_label = 0

        self.tb_logger = tb_logger
        if tb_logger is not None:
            from tensorboard_logger import TensorBoardLogger
            encoder_final = self.encoder_decoder.encoder._modules['final_layer']
            encoder_final.weight.register_hook(tb_logger.grad_hook_by_name('grads/encoder_out'))
            decoder_final = self.encoder_decoder.decoder._modules['linear']
            decoder_final.weight.register_hook(tb_logger.grad_hook_by_name('grads/decoder_out'))
            # discrim_final = self.discriminator._modules['linear']
            # discrim_final.weight.register_hook(tb_logger.grad_hook_by_name('grads/discrim_out'))

#训练网络在一个batch上面。
    def train_on_batch(self, batch: list):
        """
        Trains the network on a single batch consisting of images and messages
        :param batch: batch of training data, in the form [images, messages]
        :return: dictionary of error metrics from Encoder, Decoder, and Discriminator on the current batch
        """
        images, messages = batch

        batch_size = images.shape[0]
        self.encoder_decoder.train()
        # self.discriminator.train()
        with torch.enable_grad():                       #注意一下这个会不会让noiser变得可以计算梯度。
            # # ---------------- Train the discriminator -----------------------------
            # self.optimizer_discrim.zero_grad()
            # # train on cover
            # d_target_label_cover = torch.full((batch_size, 1), self.cover_label, device=self.device)#真实图片标签
            # d_target_label_encoded = torch.full((batch_size, 1), self.encoded_label, device=self.device)#编码图片标签
            # g_target_label_encoded = torch.full((batch_size, 1), self.cover_label, device=self.device)
            #
            # d_on_cover = self.discriminator(images)#输出一串数字，长度batch_size
            # d_loss_on_cover = self.bce_with_logits_loss(d_on_cover.float(), d_target_label_cover.float())#二分类损失
            # d_loss_on_cover.backward()
            #
            # # train on fake
            # encoded_images, noised_images, decoded_messages = self.encoder_decoder(images, messages)
            encoded_images, s_embedding,s_me_embedding,embedding, decoded_messages, me_decoded_messages = self.encoder_decoder(images, messages)#此时编码和解码器不进行更新。
            # d_on_encoded = self.discriminator(encoded_images.detach())
            # d_loss_on_encoded = self.bce_with_logits_loss(d_on_encoded.float(), d_target_label_encoded.float())#二分损失
            #
            # d_loss_on_encoded.backward()
            # self.optimizer_discrim.step()
            # #训练出一个有效的二分类器来区分真实图片和编码图片。

            # --------------Train the generator (encoder-decoder) ---------------------
            self.optimizer_enc_dec.zero_grad()
            # # target label for encoded images should be 'cover', because we want to fool the discriminator
            # d_on_encoded_for_enc = self.discriminator(encoded_images)
            # g_loss_adv = self.bce_with_logits_loss(d_on_encoded_for_enc.float(), g_target_label_encoded.float())#希望产生编码图片能骗过分类器。

            if self.vgg_loss == None:
                g_loss_enc = self.mse_loss(encoded_images, images)
            else:
                vgg_on_cov = self.vgg_loss(images)              #目前这两个损失没有。
                vgg_on_enc = self.vgg_loss(encoded_images)
                g_loss_enc = self.mse_loss(vgg_on_cov, vgg_on_enc)

            g_loss_dec = self.mse_loss(decoded_messages, messages)  #消息损失。
            me_g_loss_dec = self.mse_loss(decoded_messages, me_decoded_messages)  #me模型的解码损失
            ##########################################me损失#############################
            # target = torch.tensor([1], dtype=torch.float).cuda()
            # me_loss = self.me_loss(s_embedding, s_me_embedding, target)                     #让盗版输出和原来的相差很近
            ###############################################################################
            # g_loss = self.mse_loss(s_decoded_messages, torch.Tensor(np.random.choice([0, 1], (decoded_messages.size(0), 30))).to('cuda'))
            # g_loss = self.mse_loss(s_decoded_messages, s_messages)

            g_loss = self.config.decoder_loss * g_loss_dec+self.config.me_decoder_loss * me_g_loss_dec    #me_decoder_loss这个损失占比不能太大
            # g_loss = self.config.encoder_loss * g_loss_enc + self.config.decoder_loss * g_loss_dec + self.config.decoder_loss * g_loss

            # print("==================查看是否noiser能够更新，避免出错==========")
            # for name, parms in self.encoder_decoder.noiser.named_parameters():
            #     print('-->name:', name)
            #     print('-->para:', parms)
            #     print('-->grad_requirs:', parms.requires_grad)
            #     #print('-->grad_value:', parms.grad)
            #     print("======================================================")

            g_loss.backward()
            self.optimizer_enc_dec.step()

        decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)#round()四舍五入，clip（0,1）控制上下界
        bitwise_avg_err = np.sum(np.abs(decoded_rounded - messages.detach().cpu().numpy())) / (
                batch_size * messages.shape[1])#np.abs计算绝对值

        losses = {
            'loss           ': g_loss.item(),
            'encoder_mse    ': g_loss_enc.item(),
            'dec_mse        ': g_loss_dec.item(),
            'me_g_loss_dec  ': me_g_loss_dec.item(),
            'bitwise-error  ': bitwise_avg_err,                 #水印与真实水印差距，不计算梯度。
            # 'adversarial_bce': g_loss_adv.item(),
            # 'discr_cover_bce': d_loss_on_cover.item(),
            # 'discr_encod_bce': d_loss_on_encoded.item()
        }
        return losses, (encoded_images, embedding, decoded_messages)

    def validate_on_batch(self, batch: list):
        """
        Runs validation on a single batch of data consisting of images and messages
        :param batch: batch of validation data, in form [images, messages]
        :return: dictionary of error metrics from Encoder, Decoder, and Discriminator on the current batch
        """
        # if TensorboardX logging is enabled, save some of the tensors.
        # if self.tb_logger is not None:
        #     encoder_final = self.encoder_decoder.encoder._modules['final_layer']
        #     self.tb_logger.add_tensor('weights/encoder_out', encoder_final.weight)
        #     decoder_final = self.encoder_decoder.decoder._modules['linear']
        #     self.tb_logger.add_tensor('weights/decoder_out', decoder_final.weight)
        #     discrim_final = self.discriminator._modules['linear']
        #     self.tb_logger.add_tensor('weights/discrim_out', discrim_final.weight)

        # images, messages = batch
        #第一个是输入图像，messages是30位的，messages_val是10位的。
        images, messages= batch

        batch_size = images.shape[0]

        self.encoder_decoder.train()
        # self.encoder_decoder.eval()

        # self.discriminator.eval()
        with torch.no_grad():
            # d_target_label_cover = torch.full((batch_size, 1), self.cover_label, device=self.device)#造标签
            # d_target_label_encoded = torch.full((batch_size, 1), self.encoded_label, device=self.device)
            # g_target_label_encoded = torch.full((batch_size, 1), self.cover_label, device=self.device)
            # #编码图片与原图片之间的损失。
            # d_on_cover = self.discriminator(images)
            # d_loss_on_cover = self.bce_with_logits_loss(d_on_cover.float(), d_target_label_cover.float())

            encoded_images,s_embedding,s_me_embedding, embedding, decoded_messages, me_decoded_messages = self.encoder_decoder(
                images, messages)
            # encoded_images_val,s_embedding_val,s_me_embedding_val, embedding_val, decoded_messages_val, s_decoded_messages_val = self.encoder_decoder(
            #     images, messages_val)

            # d_on_encoded = self.discriminator(encoded_images)
            # d_loss_on_encoded = self.bce_with_logits_loss(d_on_encoded.float(), d_target_label_encoded.float())

            # d_on_encoded_for_enc = self.discriminator(encoded_images)
            # g_loss_adv = self.bce_with_logits_loss(d_on_encoded_for_enc.float(), g_target_label_encoded.float())

            if self.vgg_loss is None:
                g_loss_enc = self.mse_loss(encoded_images, images)
            else:
                vgg_on_cov = self.vgg_loss(images)
                vgg_on_enc = self.vgg_loss(encoded_images)
                g_loss_enc = self.mse_loss(vgg_on_cov, vgg_on_enc)

            g_loss_dec = self.mse_loss(decoded_messages, messages)  # 隐藏信息的损失。
            ##########################################me损失#############################
            target = torch.tensor([1], dtype=torch.float).cuda()
            me_loss = self.me_loss(s_embedding, s_me_embedding, target)
            g_loss = self.config.encoder_loss * g_loss_enc + self.config.decoder_loss * g_loss_dec + self.config.me_loss * me_loss

        decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
        #decoded_rounded_val = decoded_messages_val.detach().cpu().numpy().round().clip(0, 1)
        # a = decoded_rounded - messages.detach().cpu().numpy()
        # dec_round = torch.zeros([decoded_rounded.shape[0], int(decoded_rounded.shape[1]/3)])
        # for i in range(decoded_rounded.shape[0]):
        #     for j in range(int(decoded_rounded.shape[1]/3)):
        #         val = decoded_rounded[i][j*3]+decoded_rounded[i][j*3+1]+decoded_rounded[i][j*3+2]
        #         if val >= 2:
        #             dec_round[i][j]=1

        # bitwise_avg_err = np.sum(np.abs(dec_round.detach().cpu().numpy() - messages_val.detach().cpu().numpy())) / (
        #         batch_size * messages_val.shape[1])
        bitwise_avg_err = np.sum(np.abs(decoded_rounded - messages.detach().cpu().numpy())) / (
                batch_size * messages.shape[1])
        # bitwise_avg_val_err = np.sum(np.abs(decoded_rounded_val - messages_val.detach().cpu().numpy())) / (
        #         batch_size * messages_val.shape[1])  # 这个随机的，我希望越大越好，错的越多越好

        losses = {
            'loss           ': g_loss.item(),  # 图片，指纹，鉴别器损失和。
            'encoder_mse    ': g_loss_enc.item(),  # 图片损失
            'dec_mse        ': g_loss_dec.item(),  # 计算指纹与指纹之间的损失
            'me_loss        ': me_loss.item(),
            'one_message_bitwise-error  ': bitwise_avg_err,  # 这个是指纹与真实指纹之间的差距输出。
            #'random_message_bitwise_err': bitwise_avg_val_err
            # 'adversarial_bce': g_loss_adv.item(),                   #下面三个损失都是二分器的损失
            # 'discr_cover_bce': d_loss_on_cover.item(),
            # 'discr_encod_bce': d_loss_on_encoded.item()
        }
        return losses, (encoded_images, embedding, decoded_messages)
    # def to_stirng(self):
    #     return '{}\n{}'.format(str(self.encoder_decoder), str(self.discriminator))
    def to_stirng(self):
        return '{}\n'.format(str(self.encoder_decoder))


    def info_nce_loss(self, features):

        labels = torch.cat([torch.arange(self.args.batch_size) for i in range(self.args.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.cuda()

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        logits = logits / self.args.temperature
        return logits, labels

    def train_simclr(self, batch: list):
        images,messages = batch
        encoded_image,s_embedding, s_me_embedding, embedding, decoded_message, me_decoded_message = self.encoder_decoder(images, messages)

        logits, labels = self.info_nce_loss(s_embedding)
        simclr_loss = self.criterion(logits, labels)

        self.optimizer_enc_dec.zero_grad()

        simclr_loss.backward()
        self.optimizer_enc_dec.step()
        losses = {
            'simclr_loss': simclr_loss.item(),  # 图片，指纹，鉴别器损失和。
        }

        return losses
    def train_me(self,batch:list):
        images, messages = batch
        ##带信息的图像，干净图像的embedding，me模型干净图像的embedding，带信息的embedding，解码出的信息，me的解码信息
        encoded_image,s_embedding, s_me_embedding, embedding, decoded_message, me_decoded_message = self.encoder_decoder(
            images, messages)

        # logits, labels = self.info_nce_loss(s_embedding)
        # simclr_loss = self.criterion(logits, labels)

        self.optimizer_enc_dec.zero_grad()
        ##########################################me损失#############################
        target = torch.tensor([1], dtype=torch.float).cuda()
        me_loss = self.me_loss(s_embedding, s_me_embedding, target)
        me_loss.backward()
        #simclr_loss.backward()
        self.optimizer_enc_dec.step()
        me_loss = {
            'me_loss': me_loss.item(),  # 图片，指纹，鉴别器损失和。
        }

        return me_loss


    # def save_checkpoint(self,path1, path2):
    #     torch.save(self.encoder_decoder.state_dict(), path1)
    #     torch.save(self.discriminator.state_dict(), path2)