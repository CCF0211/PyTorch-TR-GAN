import torch
from util.visualizer import save_images, save_images_to_CAPS_folder
import os
from util import html
import numpy as np

from util.metrics import msssim_3d, psnr3d


def test_data_loss(model, dataset, epoch, opt, phase='test', save_results=True, save_freq=1, all_metric=True,
                   is_train=False):
    # create a website
    # hard-code some parameters for test
    ori_display_id = opt.display_id
    ori_visual_names = model.visual_names
    ori_phase = model.phase
    ori_swap = model.swap
    # model.phase = 'test'
    opt.display_id = -1  # no visdom display; the test code saves the results to a HTML file.
    opt.phase = phase
    loss_fn_l1 = torch.nn.L1Loss()
    loss_fn_l2 = torch.nn.MSELoss()
    if opt.model == 'time_recurrent_gan' or opt.model == 'TRv2_gan':
        model.visual_names = []
        if is_train:
            for i in range(opt.T_length):
                model.visual_names.append('real_ses_M{}'.format((i + 1) * opt.interval))
                model.visual_names.append('diff_ses_M{}'.format((i + 1) * opt.interval))
            for i in range(opt.T_length + opt.T_test - 1):
                model.visual_names.append('fake_ses_M{}'.format((i + 1) * opt.interval))
                model.visual_names.append('state_ses_M{}'.format((i + 1) * opt.interval))
        else:
            for i in range(opt.T_length + opt.T_test - 1):
                model.visual_names.append('real_ses_M{}'.format((i + 1) * opt.interval))
                model.visual_names.append('fake_ses_M{}'.format((i + 1) * opt.interval))
                model.visual_names.append('state_ses_M{}'.format((i + 1) * opt.interval))
                model.visual_names.append('diff_ses_M{}'.format((i + 1) * opt.interval))

    if save_results and epoch % save_freq == 0:  # save result every <save_epoch_freq> epochs:
        web_dir = os.path.join(opt.results_dir, opt.name, opt.phase,
                               '{}_epoch_{}'.format(opt.phase, epoch))  # define the website directory
        if opt.load_iter > 0:  # load_iter is 0 by default
            web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
        print('creating web directory', web_dir)
        webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, epoch))

    for i, data in enumerate(dataset):  # inner loop within one epoch
        with torch.no_grad():
            model.set_input(data)  # get the model.dataset_session_length
        break

    loss_l1_list = [[] for _ in range(model.dataset_session_length - 1)]
    loss_l1_mean_list = [0 for _ in range(model.dataset_session_length - 1)]
    loss_l1_std_list = [0 for _ in range(model.dataset_session_length - 1)]
    loss_l2_list = [[] for _ in range(model.dataset_session_length - 1)]
    loss_l2_mean_list = [0 for _ in range(model.dataset_session_length - 1)]
    loss_l2_std_list = [0 for _ in range(model.dataset_session_length - 1)]
    msssim_list = [[] for _ in range(model.dataset_session_length - 1)]
    msssim_mean_list = [0 for _ in range(model.dataset_session_length - 1)]
    msssim_std_list = [0 for _ in range(model.dataset_session_length - 1)]
    psnr_list = [[] for _ in range(model.dataset_session_length - 1)]
    psnr_mean_list = [0 for _ in range(model.dataset_session_length - 1)]
    psnr_std_list = [0 for _ in range(model.dataset_session_length - 1)]

    if opt.eval:
        model.eval()
    for i, data in enumerate(dataset):  # inner loop within one epoch
        with torch.no_grad():
            model.set_input(data)
            model.forward()
            model.compute_visuals()

            visuals = model.get_current_visuals()  # get image results
            img_path = model.get_image_paths()  # get image paths
            for ii in range(len(model.real_all_ses_images) - 1):
                Recon_test_loss_l1 = loss_fn_l1(model.fake_images_list[ii],
                                                model.real_all_ses_images[ii + 1])
                Recon_test_loss_l2 = loss_fn_l2(model.fake_images_list[ii],
                                                model.real_all_ses_images[ii + 1])
                loss_l1_list[ii].append(Recon_test_loss_l1.cpu().numpy())
                loss_l2_list[ii].append(Recon_test_loss_l2.cpu().numpy())
                if all_metric:
                    msssim_loss = msssim_3d(model.fake_images_list[ii],
                                            model.real_all_ses_images[ii + 1])
                    psnr_loss = psnr3d(model.fake_images_list[ii],
                                       model.real_all_ses_images[ii + 1])
                    msssim_list[ii].append(msssim_loss.cpu().numpy())
                    psnr_list[ii].append(psnr_loss)

            if save_results and epoch % save_freq == 0:  # save result every <save_epoch_freq> epochs:
                save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize,
                            normalize=opt.normalize, save_3D_MRI=True)

    for i in range(model.dataset_session_length - 1):
        loss_l1_mean_list[i] = np.mean(loss_l1_list[i])
        loss_l1_std_list[i] = np.std(loss_l1_list[i])
        loss_l2_mean_list[i] = np.mean(loss_l2_list[i])
        loss_l2_std_list[i] = np.std(loss_l2_list[i])
        if all_metric:
            msssim_mean_list[i] = np.mean(msssim_list[i])
            msssim_std_list[i] = np.std(msssim_list[i])
            psnr_mean_list[i] = np.mean(psnr_list[i])
            psnr_std_list[i] = np.std(psnr_list[i])

        print('{}_M{}_mean_l1_recon_loss:{}'.format(opt.phase, (i + 1) * opt.interval, loss_l1_mean_list[i]))
        print('{}_M{}_std_l1_recon_loss:{}'.format(opt.phase, (i + 1) * opt.interval, loss_l1_std_list[i]))
        print('{}_M{}_mean_l2_recon_loss:{}'.format(opt.phase, (i + 1) * opt.interval, loss_l2_mean_list[i]))
        print('{}_M{}_std_l2_recon_loss:{}'.format(opt.phase, (i + 1) * opt.interval, loss_l2_std_list[i]))
        if all_metric:
            print('{}_M{}_mean_msssim_loss:{}'.format(opt.phase, (i + 1) * opt.interval, msssim_mean_list[i]))
            print('{}_M{}_std_msssim_loss:{}'.format(opt.phase, (i + 1) * opt.interval, msssim_std_list[i]))
            print('{}_M{}_mean_psnr_loss:{}'.format(opt.phase, (i + 1) * opt.interval, psnr_mean_list[i]))
            print('{}_M{}_std_psnr_loss:{}'.format(opt.phase, (i + 1) * opt.interval, psnr_std_list[i]))

    opt.phase = ori_phase  # set to original phase
    opt.display_id = ori_display_id
    model.visual_names = ori_visual_names
    model.phase = ori_phase
    model.swap = ori_swap
    torch.cuda.empty_cache()
    if save_results and epoch % save_freq == 0:  # save result every <save_epoch_freq> epochs:
        webpage.save()  # save the HTML

    return loss_l1_mean_list[-1]


def inference(model, dataset, opt, phase='inference', fake_CAPS_path=None):
    # create a website
    # hard-code some parameters for test
    opt.display_id = -1  # no visdom display; the test code saves the results to a HTML file.
    opt.phase = phase
    # ses = 'ses-M' + str(opt.T_length * opt.interval)
    ses_list = []
    model.visual_names = []
    for i in range(opt.T_test):
        model.visual_names.append('fake_ses_M{}'.format((i + opt.T_length) * opt.interval))
        ses_list.append('ses-M' + str((i + opt.T_length) * opt.interval))
    print(model.visual_names)
    web_dir = os.path.join(opt.results_dir, opt.name, opt.phase,
                           '{}_eval_{}'.format(opt.phase, opt.eval))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, eval= %s' % (opt.name, opt.phase, opt.eval))
    num_count = 0
    if opt.eval:
        model.eval()
    for i, data in enumerate(dataset):  # inner loop within one epoch
        with torch.no_grad():
            model.set_input(data)
            model.forward()
            model.compute_visuals()

            visuals = model.get_current_visuals()  # get image results
            img_path = model.get_image_paths()  # get image paths

            for ses in ses_list:
                num_count = save_images_to_CAPS_folder(webpage, visuals, img_path, ses, num_count,
                                                       aspect_ratio=opt.aspect_ratio,
                                                       width=opt.display_winsize,
                                                       normalize=opt.normalize, save_3D_MRI=True,
                                                       fake_CAPS_path=fake_CAPS_path,
                                                       html_dir=web_dir, opt=opt)
    webpage.save()  # save the HTML
