from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
import datetime
from util.test_and_inference import inference
from util.GPUManager import GPUManager

if __name__ == '__main__':
    opt = TrainOptions().parse()  # get training options
    # choice useful GPU
    gm = GPUManager()
    device = gm.auto_choice()
    opt.gpu_ids = [device]
    # settings
    opt.checkpoints_dir = "/root/Downloads/model_dict/T3/TR-GAN-68-t1/"
    opt.results_dir = "/root/Downloads/inference"

    print('log dir:{}'.format(opt.checkpoints_dir))
    # model settings
    opt.netD = "pixel"
    opt.netG = "Recurrent_unet_32"
    opt.gan_mode = "vanilla"
    opt.ndf = 64
    opt.ngf = 64
    opt.T_length = 3
    opt.T_test = 3
    opt.num_crop_D = 128
    opt.D_Preprocess = 'random_crop'
    opt.caps_dir = '/root/Downloads/CAPS_only_image/'
    opt.segm_mode = 't1_linear'

    opt.phase = 'inference'
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)  # get the number of images in the dataset.
    print('The number of inference images = %d' % dataset_size)

    # hard-code some parameters for test
    opt.num_threads = 0  # test code only supports num_threads = 1
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True  # no flip; comment this line if results on flipped images are needed.

    description = 'Infer_' + opt.model + opt.netG + '_exp_T_' + str(
        opt.T_length) + '_' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    opt.name = description

    fake_CAPS_path = "/root/Downloads/fake_T1-linear_CAPS_TR-GAN-26-T{}-infer{}".format(opt.T_length,
                                                                                        opt.T_test)  # set for save fake images

    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)  # create a visualizer that display/save images and plot

    # load model dict
    model.save_dir = "/root/Downloads/model_dict/T3/TR-GAN-68-t1/"
    model.load_networks(epoch=100)

    # inference dataset
    inference(model, dataset, opt, fake_CAPS_path=fake_CAPS_path)
