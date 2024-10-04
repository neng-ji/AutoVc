import math
from pathlib import Path
from sys import argv

import librosa
import librosa.filters
import librosa.util

import numpy as np
from numpy.random import RandomState

from scipy import signal
from scipy.signal import get_window

import soundfile as sf

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils import data

from submitted import *

# begin hyperparameters
sampling_rate = 16000
mel_shift = 12
mel_scale = 12
class audio_waveglow():
    filter_length = 1024
    hop_length = 256
    win_length = 1024
    n_mel_channels = 80
    sampling_rate = 22050
    mel_fmin = 0
    mel_fmax = 8000.0
    max_wav_value = 32768.0
    min_log_value = -11.52
    max_log_value = 1.2
    silence_threshold_db = -10
# end hyperparameters

def librosa_load_file(file_path, target_sr=None):
    data, sr = librosa.load(str(file_path), sr=target_sr)
    return torch.FloatTensor(data.astype(np.float32)), sr

class MelspecTransform():
    def __init__(self, transform_hparams, normalize=False, resample=None, loader='librosa'):
        args = transform_hparams
        self.stft = TacotronSTFT(
            args.filter_length, args.hop_length, args.win_length,
            args.n_mel_channels, args.sampling_rate, args.mel_fmin,
            args.mel_fmax)
        self.sampling_rate = args.sampling_rate
        self.normalize = normalize

        if resample: self.sampling_rate = int(resample)
        if loader == 'librosa': self.loader = lambda x: librosa_load_file(x, target_sr=self.sampling_rate)
        else: raise NotImplementedError("Loading method not implemented!")
    
    def from_file(self, path):
        data, sr = self.loader(path)
        return self.from_array(data, sr)

    def from_array(self, data, sr=None):
        if self.normalize: data = 0.95*(data / abs(data).max())

        if sr != self.stft.sampling_rate:
            raise ValueError("{} SR doesn't match target {} SR".format( \
                sr, self.stft.sampling_rate))
        audio_norm = data.clamp(-1, 1)
        audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
        melspec = self.stft.mel_spectrogram(audio_norm[None])
        melspec = torch.squeeze(melspec, 0)
        return melspec

class STFT(torch.nn.Module):
    """adapted from Prem Seetharaman's https://github.com/pseeth/pytorch-stft"""
    def __init__(self, filter_length=800, hop_length=200, win_length=800,
                 window='hann'):
        super(STFT, self).__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.forward_transform = None
        scale = self.filter_length / self.hop_length
        fourier_basis = np.fft.fft(np.eye(self.filter_length))

        cutoff = int((self.filter_length / 2 + 1))
        fourier_basis = np.vstack([np.real(fourier_basis[:cutoff, :]),
                                   np.imag(fourier_basis[:cutoff, :])])

        forward_basis = torch.FloatTensor(fourier_basis[:, None, :])
        inverse_basis = torch.FloatTensor(
            np.linalg.pinv(scale * fourier_basis).T[:, None, :].astype(np.float32))

        if window is not None:
            assert(filter_length >= win_length)
            fft_window = get_window(window, win_length, fftbins=True)
            fft_window = librosa.util.pad_center(data=fft_window, size=filter_length)
            fft_window = torch.from_numpy(fft_window).float()

            forward_basis *= fft_window
            inverse_basis *= fft_window

        self.register_buffer('forward_basis', forward_basis.float())
        self.register_buffer('inverse_basis', inverse_basis.float())

    def transform(self, input_data):
        num_batches = input_data.size(0)
        num_samples = input_data.size(1)

        self.num_samples = num_samples

        input_data = input_data.view(num_batches, 1, num_samples)
        input_data = F.pad(
            input_data.unsqueeze(1),
            (int(self.filter_length / 2), int(self.filter_length / 2), 0, 0),
            mode='reflect')
        input_data = input_data.squeeze(1)

        forward_transform = F.conv1d(
            input_data,
            Variable(self.forward_basis, requires_grad=False),
            stride=self.hop_length,
            padding=0)

        cutoff = int((self.filter_length / 2) + 1)
        real_part = forward_transform[:, :cutoff, :]
        imag_part = forward_transform[:, cutoff:, :]

        magnitude = torch.sqrt(real_part**2 + imag_part**2)
        phase = torch.autograd.Variable(
            torch.atan2(imag_part.data, real_part.data))

        return magnitude, phase

    def forward(self, input_data):
        self.magnitude, self.phase = self.transform(input_data)
        reconstruction = self.inverse(self.magnitude, self.phase)
        return reconstruction

class TacotronSTFT(torch.nn.Module):
    def __init__(self, filter_length=1024, hop_length=256, win_length=1024,
                 n_mel_channels=80, sampling_rate=22050, mel_fmin=0.0,
                 mel_fmax=8000.0):
        super(TacotronSTFT, self).__init__()
        self.n_mel_channels = n_mel_channels
        self.sampling_rate = sampling_rate
        self.stft_fn = STFT(filter_length, hop_length, win_length)
        mel_basis = librosa.filters.mel(
            sr=sampling_rate, n_fft=filter_length, n_mels=n_mel_channels,
            fmin=mel_fmin, fmax=mel_fmax)
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer('mel_basis', mel_basis)

    def spectral_normalize(self, magnitudes):
        clip_val=1e-5
        C = 1 # compression factor
        output = torch.log(torch.clamp(magnitudes, min=clip_val) * C)
        return output

    def mel_spectrogram(self, y):
        assert(torch.min(y.data) >= -1)
        assert(torch.max(y.data) <= 1)

        magnitudes, phases = self.stft_fn.transform(y)
        magnitudes = magnitudes.data

        mel_output = torch.matmul(self.mel_basis, magnitudes)
        mel_output = self.spectral_normalize(mel_output)
        return mel_output

class GRUEmbedder(nn.Module):
    def __init__(self, normalize=True, **kwargs):
        super().__init__()
        self.model = SpeakerEmbedder(768, 80, 3, 256, 0.3)
        self.hparams = audio_waveglow
        self.melspec_tfm = MelspecTransform(self.hparams, **kwargs)
        self.should_normalize = normalize
    
    def forward(self, x):
        if self.should_normalize:
            x = self.normalize(x)
        return self.model(x)
    
    def normalize(self, x):
        _normer = -self.hparams.min_log_value/2
        return (x + _normer)/_normer

    def melspec_from_file(self, x):
        return self.melspec_tfm.from_file(x).T
    
    def melspec_from_array(self, x):
        return self.melspec_tfm.from_array(x).T

    def print_hparams(self):
        for key in self.hparams.__dict__():
            if str(key).startswith('__') == True: continue
            print(key, ':', getattr(self.hparams, key))

class Generator(nn.Module):
    def __init__(self, dim_neck, dim_emb, dim_pre, freq):
        super(Generator, self).__init__()
        
        self.encoder = Encoder(dim_neck, dim_emb)
        self.decoder = Decoder(dim_neck, dim_emb, dim_pre)
        self.postnet = Postnet()
        self.freq = freq

    def forward(self, x, c_org, c_trg):
        # start Encoder steps
        x = x.squeeze(1).transpose(2,1)
        c_org = c_org.unsqueeze(-1).expand(-1, -1, x.size(-1))
        x = torch.cat((x, c_org), dim=1)

        outputs = self.encoder(x)
        out_forward = outputs[:,:,:int(outputs.shape[2]/2)]
        out_backward = outputs[:,:,int(outputs.shape[2]/2):]
        
        codes = []
        for i in range(0, out_forward.size(1), self.freq):
            codes.append(torch.cat((out_forward[:,i+self.freq-1,:],out_backward[:,i,:]), dim=-1))

        if c_trg is None:
            return torch.cat(codes, dim=-1)

        x = x.transpose(1, 2)
        tmp = []
        for code in codes:
            tmp.append(code.unsqueeze(1).expand(-1,int(x.size(1)/len(codes)),-1))
        code_exp = torch.cat(tmp, dim=1)
        
        encoder_outputs = torch.cat((code_exp, c_trg.unsqueeze(1).expand(-1,x.size(1),-1)), dim=-1)
        # end Encoder steps
        
        mel_outputs = self.decoder(encoder_outputs)
                
        mel_outputs_postnet = self.postnet(mel_outputs.transpose(2,1))
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet.transpose(2,1)
        
        mel_outputs = mel_outputs.unsqueeze(1)
        mel_outputs_postnet = mel_outputs_postnet.unsqueeze(1)
        
        return mel_outputs, mel_outputs_postnet, torch.cat(codes, dim=-1)

norm_mel = lambda x: (x + mel_shift) / mel_scale
denorm_mel = lambda x: (x*mel_scale) - mel_shift

class AutoVC(Generator):
    def __init__(self, dim_neck=32, dim_emb=256, dim_pre=512, freq=32, normalize=True):
        super().__init__(dim_neck, dim_emb, dim_pre, freq)
        self.normalize = normalize

    def normalize_mel(self, x):
        return norm_mel(x) if self.normalize else x

    def denormalize_mel(self, x):
        return denorm_mel(x) if self.normalize else x
    
    def mspec_from_file(self, pth):
        mspec = get_mspec(pth) # (N, n_mels)
        mspec = self.normalize_mel(mspec)
        return mspec

    def pad_mspec(self, mel):
        mspec_padded, len_pad = pad_seq(mel)
        if not self.normalize:
            raise NotImplementedError("Padding assumes spectrograms scales with min value of 0.")
        return mspec_padded, len_pad

    def mspec_from_numpy(array, sampling_rate):
        mspec = get_mspec_from_array(array, sampling_rate, return_waveform=True) # (N, n_mels)
        mspec = self.normalize_mel(mspec)
        return mspec

def load_hifigan(progress=True, **kwargs):
    svpath = Path('pretrained_models/ece417mp6_hifigan.pt')
    importer = torch.package.PackageImporter(svpath)
    vocoder = importer.load_pickle("models", "hifigan.pkl", map_location=torch.device('cpu'))
    return vocoder

def pad_seq(x, base=32):
    len_out = int(base * math.ceil(float(x.shape[0])/base))
    len_pad = len_out - x.shape[0]
    assert len_pad >= 0
    return torch.nn.functional.pad(x, (0,0,0,len_pad), value=0), len_pad


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def pySTFT(x, fft_length=1024, hop_length=256):
    x = np.pad(x, int(fft_length // 2), mode='reflect')
    noverlap = fft_length - hop_length
    shape = x.shape[:-1] + ((x.shape[-1] - noverlap) // hop_length, fft_length)
    strides = x.strides[:-1] + (hop_length * x.strides[-1], x.strides[-1])
    result = np.lib.stride_tricks.as_strided(x, shape=shape,strides=strides)
    fft_window = get_window('hann', fft_length, fftbins=True)
    result = np.fft.rfft(fft_window * result, n=fft_length).T
    return np.abs(result)

# For wavenet vocoder
mel_basis = librosa.filters.mel(sr=sampling_rate, n_fft=1024, fmin=90, fmax=7600, n_mels=80).T 
mel_basis_hifi = librosa.filters.mel(sr=sampling_rate, n_fft=1024, fmin=0, fmax=8000, n_mels=80).T
min_level = np.exp(-100 / 20 * np.log(10))
b, a = butter_highpass(30, sampling_rate, order=5)

def get_mspec(fn, is_hifigan=True, return_waveform=False):
    x, fs = sf.read(str(fn))
    x = librosa.resample(x, orig_sr=fs, target_sr=sampling_rate)
    y = signal.filtfilt(b, a, x)
    wav = y * 0.96 + (np.random.RandomState().rand(y.shape[0])-0.5)*1e-06
    D = pySTFT(wav).T

    D_mel = np.dot(D, mel_basis_hifi)
    S = np.log(np.clip(D_mel, 1e-5, float('inf'))).astype(np.float32)

    if return_waveform:
        return torch.from_numpy(S), y
    return torch.from_numpy(S)

def get_mspec_from_array(x, input_sr, is_hifigan=True, return_waveform=False):
    x = librosa.resample(x, input_sr, sampling_rate)
    y = signal.filtfilt(b, a, x)
    wav = y * 0.96 + (np.random.RandomState().rand(y.shape[0])-0.5)*1e-06
    D = pySTFT(wav).T

    D_mel = np.dot(D, mel_basis_hifi)
    S = np.log(np.clip(D_mel, 1e-5, float('inf'))).astype(np.float32)

    if return_waveform:
        return torch.from_numpy(S), y
    return torch.from_numpy(S)

old2new_autovc = {x:y for (x,y) in zip(["encoder.recurrent.forward_layers.0.weight_ih", "encoder.recurrent.forward_layers.0.weight_hh", "encoder.recurrent.forward_layers.0.bias_ih", "encoder.recurrent.forward_layers.0.bias_hh", "encoder.recurrent.reverse_layers.0.weight_ih", "encoder.recurrent.reverse_layers.0.weight_hh", "encoder.recurrent.reverse_layers.0.bias_ih", "encoder.recurrent.reverse_layers.0.bias_hh", "encoder.recurrent.forward_layers.1.weight_ih", "encoder.recurrent.forward_layers.1.weight_hh", "encoder.recurrent.forward_layers.1.bias_ih", "encoder.recurrent.forward_layers.1.bias_hh", "encoder.recurrent.reverse_layers.1.weight_ih", "encoder.recurrent.reverse_layers.1.weight_hh", "encoder.recurrent.reverse_layers.1.bias_ih", "encoder.recurrent.reverse_layers.1.bias_hh", "decoder.recurrent1.forward_layers.0.weight_ih", "decoder.recurrent1.forward_layers.0.weight_hh", "decoder.recurrent1.forward_layers.0.bias_ih", "decoder.recurrent1.forward_layers.0.bias_hh", "decoder.recurrent2.forward_layers.0.weight_ih", "decoder.recurrent2.forward_layers.0.weight_hh", "decoder.recurrent2.forward_layers.0.bias_ih", "decoder.recurrent2.forward_layers.0.bias_hh", "decoder.recurrent2.forward_layers.1.weight_ih", "decoder.recurrent2.forward_layers.1.weight_hh", "decoder.recurrent2.forward_layers.1.bias_ih", "decoder.recurrent2.forward_layers.1.bias_hh" ],["encoder.recurrent.weight_ih_l0", "encoder.recurrent.weight_hh_l0", "encoder.recurrent.bias_ih_l0", "encoder.recurrent.bias_hh_l0", "encoder.recurrent.weight_ih_l0_reverse", "encoder.recurrent.weight_hh_l0_reverse", "encoder.recurrent.bias_ih_l0_reverse", "encoder.recurrent.bias_hh_l0_reverse", "encoder.recurrent.weight_ih_l1", "encoder.recurrent.weight_hh_l1", "encoder.recurrent.bias_ih_l1", "encoder.recurrent.bias_hh_l1", "encoder.recurrent.weight_ih_l1_reverse", "encoder.recurrent.weight_hh_l1_reverse", "encoder.recurrent.bias_ih_l1_reverse", "encoder.recurrent.bias_hh_l1_reverse", "decoder.recurrent1.weight_ih_l0", "decoder.recurrent1.weight_hh_l0", "decoder.recurrent1.bias_ih_l0", "decoder.recurrent1.bias_hh_l0", "decoder.recurrent2.weight_ih_l0", "decoder.recurrent2.weight_hh_l0", "decoder.recurrent2.bias_ih_l0", "decoder.recurrent2.bias_hh_l0", "decoder.recurrent2.weight_ih_l1", "decoder.recurrent2.weight_hh_l1", "decoder.recurrent2.bias_ih_l1", "decoder.recurrent2.bias_hh_l1"]) }

old2new_gruembedder = {x:y for (x,y) in zip(["model.rnn_stack.forward_layers.0.weight_ih", "model.rnn_stack.forward_layers.0.weight_hh", "model.rnn_stack.forward_layers.0.bias_ih", "model.rnn_stack.forward_layers.0.bias_hh", "model.rnn_stack.forward_layers.1.weight_ih", "model.rnn_stack.forward_layers.1.weight_hh", "model.rnn_stack.forward_layers.1.bias_ih", "model.rnn_stack.forward_layers.1.bias_hh", "model.rnn_stack.forward_layers.2.weight_ih", "model.rnn_stack.forward_layers.2.weight_hh", "model.rnn_stack.forward_layers.2.bias_ih", "model.rnn_stack.forward_layers.2.bias_hh"],["model.rnn_stack.weight_ih_l0", "model.rnn_stack.weight_hh_l0", "model.rnn_stack.bias_ih_l0", "model.rnn_stack.bias_hh_l0", "model.rnn_stack.weight_ih_l1", "model.rnn_stack.weight_hh_l1", "model.rnn_stack.bias_ih_l1", "model.rnn_stack.bias_hh_l1", "model.rnn_stack.weight_ih_l2", "model.rnn_stack.weight_hh_l2", "model.rnn_stack.bias_ih_l2", "model.rnn_stack.bias_hh_l2"])}

def main(
    source_utterance='source_utterance.flac',
    target_utterance='target_utterance.flac',
    output_utterance='converted_utterance.flac'
):
    device = torch.device('cpu')

    autovc = AutoVC()
    autovc_state_dict = torch.load('pretrained_models/ece417mp6_autovc.pt', map_location=device)
    mapdict_autovc = {}
    for x in autovc_state_dict['model_state_dict'].keys():
        if x in old2new_autovc:
            mapdict_autovc[old2new_autovc[x]]=autovc_state_dict['model_state_dict'][x]
        else:
            mapdict_autovc[x]=autovc_state_dict['model_state_dict'][x]
    #autovc.load_state_dict(autovc_state_dict['model_state_dict'])
    autovc.load_state_dict(mapdict_autovc)
    autovc.to(device)
    autovc.eval()

    hifigan = load_hifigan()
    hifigan.to(device)
    hifigan.eval()

    gruembedder = GRUEmbedder()
    gruembedder_state_dict = torch.load('pretrained_models/ece417mp6_gruembedder.pt', map_location=device)
    mapdict_gruembedder = {}
    for x in gruembedder_state_dict.keys():
        if x in old2new_gruembedder:
            mapdict_gruembedder[old2new_gruembedder[x]]=gruembedder_state_dict[x]
        else:
            mapdict_gruembedder[x]=gruembedder_state_dict[x]
    #gruembedder.load_state_dict(gruembedder_state_dict)
    gruembedder.load_state_dict(mapdict_gruembedder)
    gruembedder.to(device)
    gruembedder.eval()

    mel = autovc.mspec_from_file(source_utterance) 

    sse_src_mel = gruembedder.melspec_from_file(source_utterance)
    with torch.no_grad(): 
        src_embedding = gruembedder(sse_src_mel[None].to(device))

    sse_trg_mel = gruembedder.melspec_from_file(target_utterance)
    with torch.no_grad(): 
        trg_embedding = gruembedder(sse_trg_mel[None].to(device))

    with torch.no_grad():
        spec_padded, len_pad = autovc.pad_mspec(mel)
        x_src = spec_padded.to(device)[None]
        s_src = src_embedding.to(device)
        s_trg = trg_embedding.to(device)
        x_identic, x_identic_psnt, _ = autovc(x_src, s_src, s_trg)
        if len_pad == 0:
            x_trg = x_identic_psnt[0, 0, :, :]
        else:
            x_trg = x_identic_psnt[0, 0, :-len_pad, :]

    @torch.no_grad()
    def vocode(spec):
        spec = autovc.denormalize_mel(spec)
        _m = spec.T[None]
        waveform = hifigan(_m.to(device))[0]
        return waveform.squeeze()

    converted_waveform = vocode(x_trg)
    sf.write(output_utterance, converted_waveform.cpu().numpy(), 16000)

if __name__ == '__main__':
    source_utterance = argv[1] if len(argv) >= 2 else 'source_utterance.flac'
    target_utterance = argv[2] if len(argv) >= 3 else 'target_utterance.flac'
    output_utterance = argv[3] if len(argv) >= 4 else 'converted_utterance.flac'
    main(source_utterance,target_utterance,output_utterance)
