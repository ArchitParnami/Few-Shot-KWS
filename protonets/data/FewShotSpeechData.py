import os
from functools import partial
import torch
import numpy as np
import glob

from torch.utils.data import Dataset
from torchvision import transforms
from torchnet.transform import compose
from torchnet.dataset import ListDataset, TransformDataset
import torchaudio
import torch.nn.functional as F
from protonets.data.base import convert_dict, CudaTransform


class FewShotSpeechDataset(TransformDataset):

    def __init__(self, data_dir, class_file, n_support, n_query, cuda, args):
        self.sample_rate = args['sample_rate']
        self.clip_duration_ms = args['clip_duration'] 
        self.window_size_ms = args['window_size']
        self.window_stride_ms = args['window_stride']
        self.feature_bin_count = args['num_features']
        self.foreground_volume = args['foreground_volume']
        self.time_shift_ms = args['time_shift']
        self.use_background = args['include_background']
        self.background_volume = args['bg_volume']
        self.background_frequency= args['bg_frequency']
        self.desired_samples = int(self.sample_rate * self.clip_duration_ms / 1000)
        self.silence = args['include_silence']
        self.silence_num_samples = args['num_silence']
        self.unknown = args['include_unknown']
        self.data_cache = {}
        self.data_dir = data_dir
        self.class_file = class_file
        self.n_support = n_support
        self.n_query = n_query
        self.background_data = self.load_background_data()
        self.mfcc = self.build_mfcc_extractor()
        self.transforms = [partial(convert_dict, 'class'), 
                           self.load_class_samples,
                           self.extract_episode]
        if cuda:
            self.transforms.append(CudaTransform())
        self.class_names = self.read()
        transforms = compose(self.transforms)
        super().__init__(ListDataset(self.class_names), transforms)   

    def load_background_data(self):
        background_path = os.path.join(self.data_dir, '..' , '_background_noise_', '*.wav')
        background_data = []
        if self.use_background or self.silence:
            for wav_path in glob.glob(background_path):
                bg_sound, bg_sr = torchaudio.load(wav_path)
                background_data.append(bg_sound.flatten())
        return background_data
    
    def build_mfcc_extractor(self):
        frame_len = self.window_size_ms / 1000
        stride = self.window_stride_ms / 1000
        mfcc = torchaudio.transforms.MFCC(self.sample_rate,
                                        n_mfcc=self.feature_bin_count,
                                        melkwargs={
                                            'hop_length' : int(stride*self.sample_rate),
                                            'n_fft' : int(frame_len*self.sample_rate)})
        return mfcc

    def read(self):
        class_names = []
        with open(self.class_file, 'r') as f:
            class_names = list(map(lambda x: x.rstrip('\n'), f.readlines()))
        if self.silence:
            class_names.append('_silence_')
        if self.unknown:
            class_names.append('_unknown_')
        return class_names
    
    def load_audio(self, key, out_field, d):
        sound, _ = torchaudio.load(filepath=d[key], normalization=True,
                                         num_frames=self.desired_samples)
        d[out_field] = sound
        return d

    def adjust_volume(self, key, d):
        d[key] =  d[key] * self.foreground_volume
        return d
    
    def shift_and_pad(self, key, d):
        audio = d[key]
        time_shift = int((self.time_shift_ms * self.sample_rate) / 1000)
        if time_shift > 0:
            time_shift_amount = np.random.randint(-time_shift, time_shift)
        else:
            time_shift_amount = 0
        
        if time_shift_amount > 0:
            time_shift_padding = (time_shift_amount, 0)
            time_shift_offset = 0
        else:
            time_shift_padding = (0, -time_shift_amount)
            time_shift_offset = -time_shift_amount
            
        padded_foreground = F.pad(audio, time_shift_padding, 'constant', 0)
        sliced_foreground = torch.narrow(padded_foreground, 1, time_shift_offset, self.desired_samples)
        d[key] = sliced_foreground
        return d

    def mix_background(self, use_background, k, d):
        foreground = d[k]
        if use_background:
            background_index = np.random.randint(len(self.background_data))
            background_samples = self.background_data[background_index]
            if len(background_samples) <= self.desired_samples:
                raise ValueError(
                    'Background sample is too short! Need more than %d'
                    ' samples but only %d were found' %
                    (self.desired_samples, len(background_samples)))
            background_offset = np.random.randint(
                0, len(background_samples) - self.desired_samples)
            background_clipped = background_samples[background_offset:(
                background_offset + self.desired_samples)]
            background_reshaped = background_clipped.reshape([1, self.desired_samples])
        
            if np.random.uniform(0, 1) < self.background_frequency:
                bg_vol = np.random.uniform(0, self.background_volume)
            else:
                bg_vol = 0
        else:
            background_reshaped = torch.zeros(1, self.desired_samples)
            bg_vol = 0

        background_mul = background_reshaped * bg_vol
        background_add = background_mul + foreground
        background_clamped = torch.clamp(background_add, -1.0, 1.0)
        d[k] = background_clamped
        return d
    
    def extract_features(self, k, d):
        features = self.mfcc(d[k])[0] # just one channel
        features = features.T # f x t -> t x f
        d[k] = torch.unsqueeze(features,0)
        return d

    def load_class_samples(self, d):
        if d['class'] not in self.data_cache:
            if d['class'] == '_silence_':
                samples = torch.zeros(self.silence_num_samples, 1, self.desired_samples)
                sample_ds = TransformDataset(ListDataset(samples),
                                            compose([
                                                partial(convert_dict, 'data'),
                                                partial(self.mix_background, True, 'data'),
                                                partial(self.extract_features, 'data')
                                            ]))

            else:
                samples = []
                
                if d['class'] == '_unknown_':
                    unknown_dir = os.path.join(self.data_dir, '..', '_unknown_')
                    split = os.path.basename(self.class_file)
                    unknown_wavs = os.path.join(unknown_dir, split)
                    with open(unknown_wavs, 'r') as rf:
                        samples = [os.path.join(unknown_dir, wav_file.strip('\n')) for wav_file in rf.readlines()]
                else:
                    keyword_dir = os.path.join(self.data_dir, d['class'])
                    samples = glob.glob(os.path.join(keyword_dir, '*.wav'))
                
                if len(samples) == 0:
                    raise Exception("No Samples found for GoogleSpeechCommand {} at {}".format(d['class'], keyword_dir))

                sample_ds = TransformDataset(ListDataset(samples),
                                            compose([
                                                partial(convert_dict, 'file_name'),
                                                partial(self.load_audio, 'file_name', 'data'),
                                                partial(self.adjust_volume, 'data'),
                                                partial(self.shift_and_pad, 'data'),
                                                partial(self.mix_background, self.use_background,'data'),
                                                partial(self.extract_features, 'data')
                                            ]))
                                        
            loader = torch.utils.data.DataLoader(sample_ds, batch_size=len(sample_ds), shuffle=False)

            for sample in loader:
                self.data_cache[d['class']] = sample['data']
                break # only need one sample because batch size equal to dataset length

        return { 'class': d['class'], 'data': self.data_cache[d['class']] }
    
    def extract_episode(self, d):
        # data: N x C x H x W
        n_examples = d['data'].size(0)

        if self.n_query == -1:
            self.n_query = n_examples - self.n_support

        if d['class'] == '_unknown_':
            start_index = torch.randint(0, n_examples - (self.n_support + self.n_query), (1,)).data[0]
            example_inds = torch.arange(start_index, start_index + (self.n_support + self.n_query))
        else:
            example_inds = torch.randperm(n_examples)[:(self.n_support + self.n_query)]
        
        support_inds = example_inds[:self.n_support]
        query_inds = example_inds[self.n_support:]

        xs = d['data'][support_inds]
        xq = d['data'][query_inds]

        return {
            'class': d['class'],
            'xs': xs,
            'xq': xq
        }


    
    

    

