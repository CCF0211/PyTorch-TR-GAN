import numbers
import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import json
import ast
import numpy as np
from random import random, shuffle
from scipy import ndimage
from .networks import smooth_l1_loss


class TimeRecurrentGanModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.set_defaults(no_dropout=True)  # default TR-GAN did not use dropout
        if is_train:
            parser.add_argument('--T_length', type=int, default=3, help='Number of sessions of input MRI image')
            parser.add_argument('--T_test', type=int, default=1, help='Number of sessions of fake MRI image to test')
            parser.add_argument('--interval', type=int, default=6,
                                help='The interval between two adjacent time points, in month')
            parser.add_argument('--swap', default=True, type=ast.literal_eval,
                                help='if true, use random swap input in training stage')
            parser.add_argument('--swap_patchs', type=int, default=8,
                                help='number of crops each axis before input to net G')
            parser.add_argument('--swap_neighbors', type=int, default=2,
                                help='number of neighbors when swap input to net G')
            parser.add_argument('--swap_epochs', type=int, default=50,
                                help='swap operator is implemented only in the first [swap_epochs] epochs')
            parser.add_argument('--D_Preprocess', type=str, default='random_crop',
                                help='data preprocess before input netD [None | random_crop | equality_crop ]')
            parser.add_argument('--num_crop_D', type=int, default=64,
                                help='number of crops before input to net D, only used if D_Preprocess is random_crop')
            parser.add_argument('--crop_patchs', type=int, default=8,
                                help='number of crops each axis before input to net D, only used if D_Preprocess is equality_crop')
            parser.add_argument('--lambda_list', default=[0.3, 0.5, 1], help='weight for different session loss')

        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)

        if type(opt.lambda_list) == str:
            opt.lambda_list = json.loads(opt.lambda_list)
        if type(opt.lambda_list) == int:
            opt.lambda_list = [opt.lambda_list, 1]
        assert opt.T_length == len(
            opt.lambda_list), 'The length of lambda_list ({}) should equal to T_length({})'.format(
            len(opt.lambda_list), opt.T_length)
        self.T_length = opt.T_length
        self.T_test = opt.T_test
        self.interval = opt.interval
        self.n_epochs = opt.n_epochs
        self.phase = opt.phase
        self.swap = opt.swap
        self.swap_patchs = opt.swap_patchs
        self.swap_neighbors = opt.swap_neighbors
        self.swap_epochs = opt.swap_epochs
        self.crop_patchs = opt.crop_patchs
        self.D_Preprocess = opt.D_Preprocess
        self.netG_name = opt.netG
        self.lambda_list = opt.lambda_list
        self.loss_names = []
        self.visual_names = []
        self.model_names = ['G']
        self.num_crop_D = opt.num_crop_D
        self.opt = opt

        if self.phase == 'train':
            # In train phase, we just forward [T_length] times
            self.loop_length = self.T_length
        else:
            # In test or inference phase, we forward [T_length] + [T_test] -1 times to synthesis [T_test] sessions fake MRI
            self.loop_length = self.T_length + self.T_test - 1

        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        for i in range(self.loop_length):
            self.visual_names.append('real_ses_M{}'.format((i + 1) * opt.interval))
        for i in range(self.loop_length):
            self.visual_names.append('fake_ses_M{}'.format((i + 1) * opt.interval))
        for i in range(self.loop_length):
            self.visual_names.append('input_ses_M{}'.format((i + 1) * opt.interval))
        for i in range(self.loop_length):
            self.visual_names.append('diff_ses_M{}'.format((i + 1) * opt.interval))
        for i in range(self.loop_length):
            self.visual_names.append('state_ses_M{}'.format((i + 1) * opt.interval))

        if self.isTrain:
            for i in range(self.loop_length):
                ses_next = (i + 1) * self.interval
                # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
                self.model_names.append('D_ses_M{}'.format(ses_next))
                if self.swap:
                    self.model_names.append('D_swap_ses_M{}'.format(ses_next))
                # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
                self.loss_names.append('G_ses_M{}'.format(ses_next))
                self.loss_names.append('D_ses_M{}'.format(ses_next))
                self.loss_names.append('Recon_ses_M{}'.format(ses_next))
                if self.swap:
                    self.loss_names.append('G_swap_ses_M{}'.format(ses_next))
                    self.loss_names.append('D_swap_ses_M{}'.format(ses_next))

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G (G_W), D_ses_M{} (D_GAN), D_swap_ses_M{} (D_SWAP)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define discriminators
            if self.D_Preprocess == 'random_crop':
                in_chanel = opt.output_nc * opt.num_crop_D
            elif self.D_Preprocess == 'equality_crop':
                in_chanel = opt.output_nc * opt.crop_patchs * opt.crop_patchs * opt.crop_patchs
            else:
                in_chanel = opt.output_nc
            self.netD_list = []
            if self.swap:
                self.netD_swap_list = []
            for i in range(self.loop_length):
                ses_next = (i + 1) * self.interval
                net = networks.define_D(in_chanel, opt.ndf, opt.netD, opt.n_layers_D, opt.norm,
                                        opt.init_type, opt.init_gain, self.gpu_ids)
                exec('self.netD_ses_M{}=net'.format(ses_next))
                self.netD_list.append(net)
                if self.swap:
                    net = networks.define_D(in_chanel, opt.ndf, opt.netD, opt.n_layers_D, opt.norm,
                                            opt.init_type, opt.init_gain, self.gpu_ids)
                    exec('self.netD_swap_ses_M{}=net'.format(ses_next))
                    self.netD_swap_list.append(net)

            self.fake_images_pool_list = [ImagePool(opt.pool_size) for _ in range(
                self.loop_length)]  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            # define swap loss functions
            self.criterionSWAP = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            # reconstruction loss
            self.criterionRecon = smooth_l1_loss()

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_list = []
            if self.swap:
                self.optimizer_D_swap_list = []
            for i in range(self.loop_length):
                self.optimizer_D_list.append(
                    torch.optim.Adam(self.netD_list[i].parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                )
                if self.swap:
                    self.optimizer_D_swap_list.append(
                        torch.optim.Adam(self.netD_swap_list[i].parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                    )
            self.optimizers.append(self.optimizer_G)
            self.optimizers += self.optimizer_D_list
            if self.swap:
                self.optimizers += self.optimizer_D_swap_list

    def set_input(self, input, epoch=10000):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.
            epoch (int): current epoch
        """
        self.image_paths = input['paths_list']
        self.dataset_session_length = input['images'].shape[1]
        self.real_all_ses_images = []
        self.swapped_all_ses_images = []
        for i in range(self.dataset_session_length):
            ses_now = i * self.interval
            image = input['images']
            image_ori = image[:, i, :, :, :, :]
            if self.swap and self.phase == 'train':
                image_swapped = self.__swap_data__(image_ori, N_patch=self.swap_patchs)
                image_swapped = image_swapped.to(self.device)
                self.swapped_all_ses_images.append(image_swapped)
                exec('self.swap_ses_M{}=image_swapped'.format(ses_now))
            image_ori = image_ori.to(self.device)
            self.real_all_ses_images.append(image_ori)
            exec('self.real_ses_M{}=image_ori'.format(ses_now))

        # swap operator is implemented only in the first [swap_epochs] epochs
        if epoch < self.swap_epochs and self.swap and self.phase == 'train':
            Prob = random()  # random input original or swapped MRI
        else:
            Prob = 0  # input original MRI
            self.swap = False  # do not update params

        if Prob > 0.5:
            self.input_all_ses_images = self.swapped_all_ses_images
            self.swapped_input = True
        else:
            self.input_all_ses_images = self.real_all_ses_images
            self.swapped_input = False

        for i in range(self.dataset_session_length):
            image = self.input_all_ses_images[i]
            exec('self.input_ses_M{}=image'.format(i * self.interval))

        # set h for netG
        self.h = torch.zeros(self.real_all_ses_images[0].shape, dtype=self.real_all_ses_images[0].dtype,
                             device=self.real_all_ses_images[0].device)
        self.fake_images_list = []
        self.loss_D_list = []
        self.loss_D_swap_list = []

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""

        for i in range(self.loop_length):
            ses_next = (i + 1) * self.interval
            if i <= self.T_length - 1:  # only input first self.T_length sessions real MRI
                input_img = self.input_all_ses_images[i]
            else:
                input_img = fake_img
            fake_img, self.h = self.netG(input_img, self.h)
            self.fake_images_list.append(fake_img)
            exec('self.fake_ses_M{}=fake_img'.format(ses_next))
            exec('self.state_ses_M{}=self.h'.format(ses_next))
            if self.phase != 'inference':
                try:
                    diff_i = (fake_img - self.real_all_ses_images[i + 1]) / 2
                    exec('self.diff_ses_M{}=diff_i'.format(ses_next))
                except:
                    print('Session {} can not calculate diff image!'.format(ses_next))
        assert len(self.fake_images_list) == self.loop_length, 'fake_images_list over length!'

    def backward_D_basic(self, netD, real, fake, i):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator
            i (tensor int)      -- which session index to synthesis

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(self.image_preprocess(real, crop_patchs=self.crop_patchs, num_crop_D=self.num_crop_D))
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(self.image_preprocess(fake.detach(), crop_patchs=self.crop_patchs, num_crop_D=self.num_crop_D))
        loss_D_fake = self.criterionGAN(pred_fake, False)

        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5 * self.lambda_list[i]
        loss_D.backward()
        return loss_D

    def backward_D_swap_basic(self, netD_swap, fake, i):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD_swap (network) -- the swap discriminator D_swap
            fake (tensor array) -- images generated by a generator
            i (tensor int)      -- which session index to synthesis

        Return the discriminator loss.
        We also call loss_D_swap.backward() to calculate the gradients.
        """
        # Swap
        pred_swap = netD_swap(
            self.image_preprocess(fake.detach(), crop_patchs=self.crop_patchs, num_crop_D=self.num_crop_D))

        loss_D_swap = self.criterionSWAP(pred_swap, not self.swapped_input)
        loss_D_swap = loss_D_swap * self.lambda_list[i]
        loss_D_swap.backward()
        return loss_D_swap

    def backward_D(self):
        """Calculate GAN loss for discriminator D and D_swap"""
        for i in range(self.loop_length):
            ses_next = (i + 1) * self.interval
            fake_i = self.fake_images_pool_list[i].query(self.fake_images_list[i])
            if self.swap:
                loss_D_swap = self.backward_D_swap_basic(self.netD_swap_list[i], self.fake_images_list[i], i)
                exec('self.loss_D_swap_ses_M{}=loss_D_swap'.format(ses_next))
                self.loss_D_swap_list.append(loss_D_swap)
            loss_D = self.backward_D_basic(self.netD_list[i], self.real_all_ses_images[i], fake_i, i)
            exec('self.loss_D_ses_M{}=loss_D'.format(ses_next))
            self.loss_D_list.append(loss_D)

    def backward_G(self):
        """Calculate the loss for generators G"""

        # GAN loss D(G(X))
        self.loss_G_list = []
        for i in range(self.loop_length):
            loss_G = self.criterionGAN(
                self.netD_list[i](self.image_preprocess(self.fake_images_list[i], crop_patchs=self.crop_patchs,
                                                        num_crop_D=self.num_crop_D)), True)

            loss_G = loss_G * self.lambda_list[i]
            exec('self.loss_G_ses_M{}=loss_G'.format((i + 1) * self.interval))
            self.loss_G_list.append(loss_G)
        # G swap loss: D_swap(G(x))
        if self.swap:
            self.loss_G_swap_list = []
            for i in range(self.loop_length):
                loss_G_swap = self.criterionSWAP(
                    self.netD_swap_list[i](self.image_preprocess(self.fake_images_list[i], crop_patchs=self.crop_patchs,
                                                                 num_crop_D=self.num_crop_D)), self.swapped_input)

                loss_G_swap = loss_G_swap * self.lambda_list[i]
                exec('self.loss_G_swap_ses_M{}=loss_G_swap'.format((i + 1) * self.interval))
                self.loss_G_swap_list.append(loss_G_swap)
        else:
            self.loss_G_swap_list = [0]

        # reconstruction loss
        self.loss_Recon_list = []
        for i in range(self.loop_length):
            loss_Recon = self.criterionRecon(self.fake_images_list[i], self.real_all_ses_images[i + 1]) * 0.5 * \
                         self.lambda_list[i]

            exec('self.loss_Recon_ses_M{}=loss_Recon'.format((i + 1) * self.interval))
            self.loss_Recon_list.append(loss_Recon)

        # combined loss and calculate gradients
        self.loss_G = sum(self.loss_G_list) + sum(self.loss_G_swap_list) + sum(self.loss_Recon_list)

        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()  # compute fake images.
        # G
        self.set_requires_grad(self.netD_list, False)  # Ds require no gradients when optimizing Gs
        if self.swap:
            self.set_requires_grad(self.netD_swap_list, False)  # D_swaps require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        self.backward_G()  # calculate gradients for G
        self.optimizer_G.step()  # update G's weights
        # D
        self.set_requires_grad(self.netD_list, True)
        if self.swap:
            self.set_requires_grad(self.netD_swap_list, True)
        for i in range(self.loop_length):
            self.optimizer_D_list[i].zero_grad()  # set D's gradients to zero
            if self.swap:
                self.optimizer_D_swap_list[i].zero_grad()  # set D_swap's gradients to zero
        self.backward_D()  # calculate gradients for all D

        for i in range(self.loop_length):
            self.optimizer_D_list[i].step()  # update all D's weights
            if self.swap:
                self.optimizer_D_swap_list[i].step()  # update all D_swap's weights

    def image_preprocess(self, data, crop_patchs=8, num_crop_D=10):
        """
        Image preprocess before input to discriminator
        """
        if self.D_Preprocess == 'random_crop':
            return self.__random_center_crop_multi__(data, num_crop_D=num_crop_D)
        elif self.D_Preprocess == 'equality_crop':
            return self.__equality_crop__(data, crop_patchs=crop_patchs)
        else:
            return data

    def __random_center_crop_multi__(self, data, num_crop_D):
        """
        FOR MSL Module:
        Randomly crops the data into num_crop_D  patches with different shapes
        args:
            data : torch tensor
            num_crop_D (Np in paper) : number of patches tp crop
        """
        device = data.device
        image = data.detach().cpu().squeeze().numpy()
        target_indexs = np.where(image > -1)
        [img_d, img_h, img_w] = image.shape
        [max_D, max_H, max_W] = np.max(np.array(target_indexs), axis=1)
        [min_D, min_H, min_W] = np.min(np.array(target_indexs), axis=1)
        [target_depth, target_height, target_width] = np.array([max_D, max_H, max_W]) - np.array([min_D, min_H, min_W])
        data_list = []
        for i in range(num_crop_D):
            Z_min = int(min_D + target_depth * 0.5 * random())
            Y_min = int(min_H + target_height * 0.5 * random())
            X_min = int(min_W + target_width * 0.5 * random())

            Z_max = Z_min + int(target_depth * random())
            Y_max = Y_min + int(target_height * random())
            X_max = X_min + int(target_width * random())

            if Z_max - Z_min < 10:
                Z_max = Z_min + 10
            if Y_max - Y_min < 10:
                Y_max = Y_min + 10
            if X_max - X_min < 10:
                X_max = X_min + 10

            Z_min = np.max([0, Z_min])
            Y_min = np.max([0, Y_min])
            X_min = np.max([0, X_min])

            Z_max = np.min([img_d, Z_max])
            Y_max = np.min([img_h, Y_max])
            X_max = np.min([img_w, X_max])

            Z_min = int(Z_min)
            Y_min = int(Y_min)
            X_min = int(X_min)

            Z_max = int(Z_max)
            Y_max = int(Y_max)
            X_max = int(X_max)

            data_crop = self.__resize_data__(image[Z_min: Z_max, Y_min: Y_max, X_min: X_max])
            data_crop = torch.tensor(data_crop).unsqueeze(0).unsqueeze(0)
            data_list.append(data_crop)
        data_out = torch.cat(data_list, 1)
        return data_out.to(device)

    def __resize_data__(self, data, target_size=[32, 32, 32]):
        """
        Resize the data to the input size
        """
        [depth, height, width] = data.shape
        scale = [target_size[0] * 1.0 / depth, target_size[1] * 1.0 / height, target_size[2] * 1.0 / width]
        data = ndimage.interpolation.zoom(data, scale, order=0)

        return data

    def __equality_crop__(self, data, crop_patchs):
        device = data.device
        data = data.cpu().squeeze()
        # depthcut, highcut, widthcut = data.shape
        if isinstance(crop_patchs, numbers.Number):
            N_patch = [int(crop_patchs), int(crop_patchs), int(crop_patchs)]
        else:
            assert len(crop_patchs) == 3, "Please provide only three dimensions (d, h, w) for size."
            N_patch = crop_patchs
        images_list = self.__crop_image__(data, N_patch)
        tensor_list = [torch.tensor(images_list[i]).unsqueeze(0).unsqueeze(0) for i in range(len(images_list))]
        data_out = torch.cat(tensor_list, dim=1)
        return data_out.to(device)

    def __swap_data__(self, img, N_patch=8, k_neighbor=2):
        """
        FOR SWAP Module:
        Crop image in to N_patch*N_patch*N_patch patchs, and random swap them during its k_neighbor neighbor patchs
        args:
            img : torch tensor
            N_patch (Ns in paper) : list[N_patch_d, N_patch_h, N_patch_w] or int N_patch
            k_neighbor (K in paper) : int swap patchs during its k_neighbor neighbor patchs
        """
        device = img.device
        img = img.cpu().squeeze()
        if isinstance(N_patch, numbers.Number):
            self.N_patch = [int(N_patch), int(N_patch), int(N_patch)]
        else:
            assert len(N_patch) == 3, "Please provide only two dimensions (d, h, w) for size."
            self.N_patch = N_patch
        images = self.__crop_image__(img, self.N_patch)
        tmpx = []
        tmpy = []
        tmpz = []
        count_x = 0
        count_y = 0
        count_z = 0
        k = k_neighbor
        RAN = 2 * k
        for i in range(self.N_patch[2] * self.N_patch[1] * self.N_patch[0]):
            tmpx.append(images[i])
            count_x += 1
            if len(tmpx) >= k:
                tmp1 = tmpx[count_x - RAN:count_x]
                shuffle(tmp1)
                tmpx[count_x - RAN:count_x] = tmp1
            if count_x == self.N_patch[2]:
                tmpy.append(tmpx)
                count_x = 0
                count_y += 1
                tmpx = []
            if len(tmpy) >= k:
                tmp2 = tmpy[count_y - RAN:count_y]
                shuffle(tmp2)
                tmpy[count_y - RAN:count_y] = tmp2
            if count_y == self.N_patch[1]:
                tmpz.append(tmpy)
                count_x = 0
                count_y = 0
                count_z += 1
                tmpx = []
                tmpy = []
            if len(tmpz) >= k:
                tmp3 = tmpz[count_z - RAN:count_z]
                shuffle(tmp3)
                tmpz[count_z - RAN:count_z] = tmp3
        random_im = []
        for liney in tmpz:
            for linex in liney:
                random_im.extend(linex)

        # random.shuffle(images)
        depth, high, width = img.shape
        id = int(depth / self.N_patch[0])
        ih = int(high / self.N_patch[1])
        iw = int(width / self.N_patch[2])
        swaped_data = torch.zeros([id * self.N_patch[0], ih * self.N_patch[1], iw * self.N_patch[2]])
        x = 0
        y = 0
        z = 0
        for i in random_im:
            i = self.__resize_data__(i, target_size=[id, ih, iw])
            swaped_data[z * id:(z + 1) * id, y * ih:(y + 1) * ih, x * iw:(x + 1) * iw] = torch.tensor(i)
            x += 1
            if x == self.N_patch[2]:
                x = 0
                y += 1
            if y == self.N_patch[1]:
                x = 0
                y = 0
                z += 1

        return swaped_data.unsqueeze(0).unsqueeze(0).to(device)

    def __crop_image__(self, image, cropnum):
        depth, high, width = image.shape
        crop_z = [int((depth / cropnum[0]) * i) for i in range(cropnum[0] + 1)]
        crop_y = [int((high / cropnum[1]) * i) for i in range(cropnum[1] + 1)]
        crop_x = [int((width / cropnum[2]) * i) for i in range(cropnum[2] + 1)]
        im_list = []
        for k in range(len(crop_z) - 1):
            for j in range(len(crop_y) - 1):
                for i in range(len(crop_x) - 1):
                    im_list.append(image[crop_z[k]:min(crop_z[k + 1], depth), crop_y[j]:min(crop_y[j + 1], high),
                                   crop_x[i]:min(crop_x[i + 1], width)])
        return im_list
