import torch
import math
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
import tqdm
import datetime
from util.test_and_inference import test_data_loss
from util.GPUManager import GPUManager
import numpy as np
import random


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)  # gpu
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.enabled = False


if __name__ == '__main__':

    set_seed(2020)
    opt = TrainOptions().parse()  # get training options

    # choice useful GPU
    gm = GPUManager()
    device = gm.auto_choice()
    opt.gpu_ids = [device]
    opt.phase = 'train'
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)  # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    opt.suffix = opt.model + opt.netG + '_exp_T_' + str(opt.T_length) + '_' + datetime.datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S")
    # opt.name = opt.suffix

    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)  # create a visualizer that display/save images and plots
    total_iters = 0  # the total number of training iterations
    opt.display_env = 'train' + opt.suffix
    best_l1_loss = 10000
    best_epoch = None
    for epoch in range(opt.epoch_count,
                       opt.n_epochs + opt.n_epochs_decay + 1):  # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        tq = tqdm.tqdm(total=math.ceil(dataset_size / opt.batch_size))
        tq.set_description('Epoch {}'.format(epoch))
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()  # timer for data loading per iteration
        epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()  # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        model.update_learning_rate()  # update learning rates in the beginning of every epoch.
        model.log_epoch(epoch, opt.n_epochs, opt.n_epochs_decay)  # log epoch to use changed loss
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data, epoch)  # unpack data from dataset and apply preprocessing
            model.optimize_parameters()  # calculate loss functions, get gradients, update network weights

            if total_iters % opt.display_freq == 0:  # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result, save_3D_MRI=True)

            if total_iters % opt.print_freq == 0:  # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:  # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)
            tq.update(1)
            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:  # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks(epoch)
        print('End of epoch %d / %d \t Time Taken: %d sec' % (
            epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
        eval_start_time = time.time()
        # test on train dataset
        train_l1_loss = test_data_loss(model, dataset, epoch, opt, phase=opt.phase, save_results=False, save_freq=50,
                                       all_metric=False, is_train=True)
        print('Eval {} dataset, Time Taken: {} sec'.format(opt.phase, time.time() - eval_start_time))
        if train_l1_loss < best_l1_loss:
            best_l1_loss = train_l1_loss
            model.save_networks(epoch, delete_epoch=best_epoch)
            best_epoch = epoch
            print('Find better L1 loss and saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
