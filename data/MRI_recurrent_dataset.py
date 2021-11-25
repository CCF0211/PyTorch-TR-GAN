"""MRI Recurrent Dataset

"""
from data.base_dataset import BaseDataset
import pandas as pd
import torch
from scipy import ndimage
import numpy as np
import os


class MRIRecurrentDataset(BaseDataset):
    """MRI Recurrent Dataset class."""

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument('--segm_mode', type=str, choices=['t1-spm-graymatter', 't1_linear'],
                            default='t1-spm-graymatter', help='type of dataset')
        parser.add_argument('--subjects_tsv_path', type=str, default='./data_splits',
                            help='TSV path with subjects/sessions to process.')
        parser.add_argument('--caps_dir', default='/root/Downloads/CAPS_only_spm_segm_origin',
                            help='caps dir where saved your MRI images. CAPS format please see: https://aramislab.paris.inria.fr/clinica/docs/public/latest/CAPS/Introduction/')
        parser.add_argument('--normalize', default=True, help='normalize input MRI images to [-1, 1].')
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)

        self.T_length = opt.T_length
        self.T_test = opt.T_test
        self.interval = opt.interval
        self.segm_mode = opt.segm_mode
        self.caps_directory = opt.caps_dir
        self.normalize = opt.normalize
        self.phase = opt.phase
        self.subjects_tsv_path = opt.subjects_tsv_path
        self.all_sessions_df = pd.read_csv(os.path.join(opt.subjects_tsv_path, 'all_sessions.tsv'), sep='\t')
        self.all_train_df = pd.read_csv(os.path.join(opt.subjects_tsv_path, 'train', 'all_train_sessions.tsv'),
                                        sep='\t')
        self.all_test_df = pd.read_csv(os.path.join(opt.subjects_tsv_path, 'test', 'all_test_sessions.tsv'),
                                       sep='\t')

        # set ses_list  ['00', '06', '12', ...]
        # In train phase, we just use previous [T_length] + 1 sessions real MRI to train model
        if self.phase == 'train':
            self.all_need_length = self.T_length + 1
        # In inference phase, we just use previous [T_length] real MRI to synthesis future sessions MRI
        elif self.phase == 'inference':
            self.all_need_length = self.T_length
        # In test phase, we need previous [T_length] + [T_test] sessions real MRI to test the synthesis performance of last [T_test] sessions fake MRI
        elif self.phase == 'test':
            self.all_need_length = self.T_length + self.T_test

        self.useful_ses_list = [str(i * self.interval) for i in range(self.all_need_length)]

        for i in range(len(self.useful_ses_list)):
            if len(self.useful_ses_list[i]) == 1:
                self.useful_ses_list[i] = 'ses-M0' + self.useful_ses_list[i]
            else:
                self.useful_ses_list[i] = 'ses-M' + self.useful_ses_list[i]

        print('In {} phase, we use {} session data!'.format(self.phase, self.useful_ses_list))

        if self.phase == 'train':
            self.df_ses = self.get_useful_ses_df(self.all_train_df, self.useful_ses_list)
        elif self.phase == 'test':
            self.df_ses = self.get_useful_ses_df(self.all_test_df, self.useful_ses_list)
        elif self.phase == 'inference':
            self.df_ses = self.get_useful_ses_df(self.all_sessions_df, self.useful_ses_list)
        self.ses_paths_list = self.get_all_need_path(self.df_ses)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.

        """
        paths_list = [self.ses_paths_list[i][index] for i in range(self.all_need_length)]
        diagnosis = self.ses_label[index]
        selected_data_list = []
        pad = torch.nn.ReplicationPad3d((4, 3, 0, 0, 4, 3))
        for i in range(self.all_need_length):
            if self.segm_mode == "t1_linear":
                path_now = self.ses_paths_list[i][index]
                data_now = torch.load(path_now)  # (1, 169, 208, 179)
                data_now = data_now.squeeze(0)  # (169, 208, 179)
                data_now = self.__resize_data__(data_now)  # (128, 128, 128)
                data_now = (data_now - np.min(data_now)) / (np.max(data_now) - np.min(data_now))
                data_now = torch.from_numpy(data_now)
                data_now = data_now.unsqueeze(0)  # (1, 128, 128, 128)
            else:
                path_now = self.ses_paths_list[i][index]
                data_now = torch.load(path_now)  # (1, 121, 145, 121)
                data_now = data_now[:, :, 8:-9, :]  # [1, 121, 128, 121]
                data_now = data_now.unsqueeze(0)  # [1, 1, 121, 128, 121]
                data_now = pad(data_now)  # [1, 1, 128, 128, 128]
                data_now = data_now.squeeze(0)  # [1, 128, 128, 128]
            if self.normalize:
                data_now = self._Normalize3D(data_now, [0.5], [0.5])
            selected_data_list.append(data_now)

        images = torch.stack(selected_data_list, 0)
        # [self.all_need_length, 1, 128, 128, 128] = session * channel * x * y * z
        if diagnosis == 'CN':
            diagnosis_label = 0
        elif diagnosis == 'MCI':
            diagnosis_label = 1
        elif diagnosis == 'AD':
            diagnosis_label = 2
        return {'images': images, 'paths_list': paths_list, 'diagnosis': diagnosis_label}

    def __len__(self):
        """Return the total number of images."""
        return self.data_size

    def __resize_data__(self, data, target_size=[128, 128, 128]):
        """
        Resize the data to the target_size
        """
        [depth, height, width] = data.shape
        scale = [target_size[0] * 1.0 / depth, target_size[1] * 1.0 / height, target_size[2] * 1.0 / width]
        data = ndimage.interpolation.zoom(data, scale, order=0)

        return data

    def _Normalize3D(self, tensor, mean, std):
        """
        Normalize the data by given mean and std
        """
        dtype = tensor.dtype
        mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
        std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
        tensor.sub_(mean[:, None, None, None]).div_(std[:, None, None, None])
        return tensor

    def get_all_need_path(self, df_ses):
        """
        get all need paths depend on the given sessions df
        """
        self.df = df_ses['participant_id'].drop_duplicates().reset_index(drop=True)
        self.data_size = len(self.df)
        ses_paths_list = [[] for _ in range(self.all_need_length)]

        for ses_i in range(self.all_need_length):
            for row_j in range(len(self.df)):
                ses_paths_list[ses_i].append(
                    self._get_path(participant=self.df.iloc[row_j], session=self.useful_ses_list[ses_i]))
        return ses_paths_list

    def get_useful_ses_df(self, ori_df, ses_list):
        """
        Return a df that has all sessions in ses_list
        """
        ses_df = pd.DataFrame(columns={"participant_id": "", "session_id": "", "diagnosis": ""})
        ori_df = ori_df.reset_index(drop=True)
        ori_subjects_df = ori_df['participant_id'].drop_duplicates().reset_index(drop=True)
        self.ses_label = []
        for i in range(ori_subjects_df.shape[0]):  # loop for every subject
            subject_df = ori_df.loc[ori_df['participant_id'] == ori_subjects_df[i]].reset_index(drop=True)
            flag = True
            subject_ses_df = pd.DataFrame(columns={"participant_id": "", "session_id": "", "diagnosis": ""})
            diagnosis = subject_df["diagnosis"].loc[0]
            for ses in ses_list:
                filted_df = subject_df.loc[subject_df['session_id'] == ses]
                if filted_df.shape[0] == 0:  # can not find this session
                    flag = False
                    break
                else:  # find this session
                    subject_ses_df = subject_ses_df.append(filted_df).reset_index(drop=True)
            if flag:
                ses_df = ses_df.append(subject_ses_df).drop_duplicates().reset_index(drop=True)
                self.ses_label.append(diagnosis)
        print('Find {} subjects'.format(ses_df['participant_id'].drop_duplicates().reset_index(drop=True).shape[0]))
        save_name = os.path.join(self.subjects_tsv_path, self.phase, 'Existing_T0_to_T' + str(
            self.all_need_length) + '_' + self.phase + '.tsv')
        ses_df.to_csv(save_name, sep='\t', index=False)

        return ses_df

    def _get_path(self, participant, session):
        """
        get the data path according CAPS format. see: https://aramislab.paris.inria.fr/clinica/docs/public/latest/CAPS/Introduction/
        """
        import os
        from os import path
        import nibabel as nib
        FILENAME_TYPE = {'cropped': '_T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w',
                         'segm-graymatter': '_T1w_segm-graymatter_space-Ixi549Space_modulated-off_probability', }
        if self.segm_mode == "t1-spm-graymatter":
            image_path = path.join(self.caps_directory, 'subjects', participant, session,
                                   'deeplearning_prepare_data', 'image_based', 't1_spm',
                                   participant + '_' + session
                                   + FILENAME_TYPE['segm-graymatter'] + '.pt')
            if not os.path.exists(image_path):
                origin_nii_path = path.join(self.caps_directory, 'subjects', participant, session,
                                            't1', 'spm', 'segmentation', 'normalized_space', participant + '_' + session
                                            + FILENAME_TYPE['segm-graymatter'] + '.nii.gz')

                image_array = nib.load(origin_nii_path).get_fdata()
                image_tensor = torch.from_numpy(image_array).unsqueeze(0).float()
                save_dir = path.join(self.caps_directory, 'subjects', participant, session,
                                     'deeplearning_prepare_data', 'image_based', 't1_spm')
                if os.path.exists(save_dir):
                    torch.save(image_tensor.clone(), image_path)
                    print('save {}'.format(image_path))
                else:
                    os.makedirs(save_dir)
                    torch.save(image_tensor.clone(), image_path)
                    print('save {}'.format(image_path))
        elif self.segm_mode == "t1_linear":
            image_path = path.join(self.caps_directory, 'subjects', participant, session,
                                   'deeplearning_prepare_data', 'image_based', 't1_linear',
                                   participant + '_' + session
                                   + FILENAME_TYPE['cropped'] + '.pt')
            if not os.path.exists(image_path):
                origin_nii_path = path.join(self.caps_directory, 'subjects', participant, session,
                                            't1', 'spm', 'segmentation', 'normalized_space', participant + '_' + session
                                            + FILENAME_TYPE['cropped'] + '.nii.gz')

                image_array = nib.load(origin_nii_path).get_fdata()
                image_tensor = torch.from_numpy(image_array).unsqueeze(0).float()
                save_dir = path.join(self.caps_directory, 'subjects', participant, session,
                                     'deeplearning_prepare_data', 'image_based', 't1_spm')
                if os.path.exists(save_dir):
                    torch.save(image_tensor.clone(), image_path)
                    print('save {}'.format(image_path))
                else:
                    os.makedirs(save_dir)
                    torch.save(image_tensor.clone(), image_path)
                    print('save {}'.format(image_path))
        else:
            raise NotImplementedError(
                "The path to seg_mode %s is not implemented" % self.seg_mode)
        if os.path.exists(image_path):
            return image_path
        else:
            raise NotImplementedError(
                "The path %s is not find! please make sure your provide tsv have corresponding session data." % self.image_path)
