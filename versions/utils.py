from DDNMplus import inference_with_mask as iwm_ddnmplus
from RePaintDDNM import inference_with_mask as iwm_repaintddnm
from RePaint import inference_with_mask as iwm_repaint

import tools.torch_tools as torch_tools
from scipy.io.wavfile import write
import numpy as np
import torch
from torchmetrics.functional.audio import signal_noise_ratio, signal_distortion_ratio
import torchaudio
import os
import librosa
from matplotlib import pyplot as plt
from tqdm import tqdm

def get_version(name_version):
    '''
    CHOOSE WTHER 
    '''    
    assert name_version in ['ddnm+', 'repaint']

    if name_version == 'ddnm+':
        print('DDNM+')
        return iwm_ddnmplus
    elif name_version == 'repaint':
        print("RePaint")
        return iwm_repaint
    
def get_d_audios(file_list, num_to_select, dataset_path, caption_filter=None, max_f=200):
    
    def check_caption(caption_filter, caption):
        if caption_filter is None:
            return True
        for word in caption_filter:
            if word in caption:
                return True
        return False
    
    selected_audios = file_list.head(max_f)

    d_audios = {}
    count = 0
    for i, audio in enumerate(selected_audios.itertuples()):
        if os.path.isfile(f"{dataset_path}/{audio._asdict()['Index']}_{audio._asdict()['start_time']}.wav") \
           and check_caption(caption_filter, audio._asdict()['caption']) :
            d_audios[count] = audio._asdict()
            count += 1
            if count == 50:
                break
                
    return d_audios

def get_psnr(snr):
    
    SNR_DICT = {100: 0.0,
            30: 0.05,
            25: 0.08,
            20: 0.13,
            17.5: 0.175,
            15: 0.22,
            10: 0.36,
            5: 0.6,
            1: 0.9}
    
    return SNR_DICT[snr]

def get_mel_spectrogram(original_audio_path, tango):
    original_mels, _, _ = torch_tools.wav_to_fbank(original_audio_path, 1024, tango.stft)
    return original_mels

def get_original_latents(original_audio_paths, tango):
    original_mels, _, _ = torch_tools.wav_to_fbank(original_audio_paths, 1024, tango.stft)
    original_mels = original_mels.unsqueeze(1)
    original_latents = tango.vae.get_first_stage_encoding(tango.vae.encode_first_stage(original_mels))
    
    return original_mels, original_latents

def apply_noise(original_latents, snr, verbose=False):
    
    lmin, lmax = torch.min(original_latents), torch.max(original_latents)

    noisy_latents = (original_latents - lmin) / (lmax - lmin)

    noise = torch.randn(noisy_latents.shape, device=noisy_latents.device)*get_psnr(snr)
    noisy_latents += noise

    original_psnr = 10*torch.log(torch.max(original_latents)/torch.var(original_latents))
    print("Original range", original_psnr, torch.var(original_latents)) if verbose else None
    noisy_psnr = 10*torch.log(torch.max(noisy_latents)/torch.var(noisy_latents))
    print("Noisy norm", noisy_psnr,torch.var(noisy_latents)) if verbose else None

    #newlmin, newlmax = torch.min(noisy_latents), torch.max(noisy_latents)

    noisy_latents = (noisy_latents * (lmax - lmin)) + lmin

    noisy_psnr = 10*torch.log(torch.max(noisy_latents)/torch.var(noisy_latents))
    print("Noisy change range", noisy_psnr,torch.var(noisy_latents)) if verbose else None


    return noisy_latents


def get_sigma_y(noisy_latents, mask=None):
    
    if mask is not None:
        sigma_y = -1  # As a consequence, sigma_y is derived from the prompt embeddings
    else: 
        sigma_y = (torch.max(noisy_latents)-torch.min(noisy_latents))*torch.std(noisy_latents)
    
    return sigma_y


def get_mask(time_mask_percentage, mask_type='time'):
    
    min_p, max_p = time_mask_percentage
        
    assert mask_type in ['time', 'mel', 'wave']

    if sum(time_mask_percentage)==0:
        return None
    else:
    
        if mask_type == 'time':
            mask = torch.ones(1,1,256,16).to('cuda')
            mask[:, :, int(min_p*256):int(max_p*256), :] = 0
        elif mask_type == 'mel':
            mask = torch.ones(1,1,1024,64).to('cuda')
            mask[:, :, int(min_p*1024):int(max_p*1024), :] = 0
        elif mask_type == 'wave':
            mask = np.ones((1, 163872), dtype=np.int16)
            mask[:, int(min_p*163872):int(max_p*163872)] = 0
        
        return mask



def save_audio(path, waves, id_file=None, descr='generated'):
    
    for i, wave in enumerate(waves):
        write(f"{path}/{id_file if id_file else i}_{descr}.wav", 16000, wave)

    return



def compute_snr(prediction, target, verbose):

    torch.manual_seed(1)

    snr = signal_noise_ratio(prediction, target).item()
    print("SNR: ", "{:.2f}".format(snr)) if verbose else None
    
    return snr



def compute_sdr(prediction, target, verbose):
        
    torch.manual_seed(1)

    sdr = signal_distortion_ratio(prediction, target).item()
    if verbose:
        print("SDR {:.5f}".format(sdr))

    return sdr



def compute_fad(path_to_clean_audio, path_to_generated_audio, verbose):
    from frechet_audio_distance import FrechetAudioDistance

    fad_embeddings = 'vggish'         # either 'vggish' or 'pann'

    if fad_embeddings == 'vggish':
        frechet = FrechetAudioDistance(
            model_name="vggish",
            use_pca=False, 
            use_activation=False,
            verbose=False
        )
    elif fad_embeddings == 'pann':
        frechet = FrechetAudioDistance(
            model_name="pann",
            use_pca=False, 
            use_activation=False,
            verbose=False
        )

    fad_score = frechet.score(path_to_clean_audio, path_to_generated_audio, dtype="float32")
    
    if verbose:
        print("FAD {:.5f}".format(fad_score))
    
    return fad_score


def load_audio_old(prediction_path, target_path):
        prediction, _ = torchaudio.load(prediction_path)
        target, _ = torchaudio.load(target_path)
        
        pmin, pmax = torch.min(prediction), torch.max(prediction)
        tmin, tmax = torch.min(target), torch.max(target)

        prediction = (prediction - pmin) / (pmax - pmin)
        prediction = (prediction * (tmax - tmin)) + tmin
    
        return prediction, target
    
def load_audio(prediction_path, target_path, segment_length):
    prediction = torch_tools.read_wav_file(prediction_path, segment_length)
    target = torch_tools.read_wav_file(target_path, segment_length)
    return prediction, target


def mask_wav(folder_audios, output_dir, segment_length=160000, mask=None):
    assert mask is not None
    
    if not os.path.isdir(f'{folder_audios}/{output_dir}'):
        os.mkdir(f'{folder_audios}/{output_dir}')
    
    for audio in tqdm(os.listdir(f'{folder_audios}/generated')):
        audio_path = f'{folder_audios}/generated/{audio}'
        audio_wav = torch_tools.read_wav_file(audio_path, segment_length)
        audio_wav = audio_wav[:,int(mask[0]*segment_length):int(mask[1]*segment_length)]
        
        save_audio(f'{folder_audios}/{output_dir}', audio_wav.cpu().numpy(), id_file=audio, descr='')
        
    return 


def compute_metrics(prediction_path, target_path, verbose=False, exclude=None):
    
    if os.path.isfile(prediction_path) and os.path.isfile(target_path):
    
        prediction, target = load_audio(prediction_path, target_path, 160000)
        
        snr = compute_snr(prediction, target, verbose)
        sdr = compute_sdr(prediction, target, verbose)
        
        print("\n--------\nBEWARE OF PICKPOCKETS!\n \
              To calculate the FAD, it is necessary to provide the path to two folders")
        
    else:
        
        snr_all = []
        sdr_all = []
        
        # We assume that files have been equally named in both the
        # prediction and target folders.
        for audio_file in os.listdir(prediction_path):
      
            audio_file_target = audio_file.split("_")[0]
            print(audio_file_target) if verbose else None
            if os.path.isfile(f'{target_path}/{audio_file_target}.wav') and (exclude is None or int(audio_file_target) not in exclude):
                prediction, target = load_audio(f'{prediction_path}/{audio_file}',f'{target_path}/{audio_file_target}.wav', 160000)
                
                # print(f'PRED {audio_file_target}   min {torch.min(prediction)}, max {torch.max(prediction)}')
                # print(f'TARGET {audio_file_target} min {torch.min(target)}, max {torch.max(target)}')
                snr = compute_snr(prediction, target, verbose)
                sdr = compute_sdr(prediction, target, verbose)

                snr_all.append(snr)
                sdr_all.append(sdr)
            
        fad = compute_fad(prediction_path, target_path, verbose)
            
        return snr_all, sdr_all, fad
    

def plot_spectrogram(original_mel, noisy_mel, output_mel, mel_mask=None, save_fig=False, id_s=0, output_path=None):
    
    original_mel_signal = original_mel.cpu()[0][0].transpose(0,1)
    original_spectrogram = np.abs(original_mel_signal)
    original_power_to_db = librosa.power_to_db(original_spectrogram, ref=np.max)
    
    if mel_mask is not None:
        noisy_mel = noisy_mel * mel_mask
    noisy_mel_signal = noisy_mel.cpu()[0][0].transpose(0,1)
    noisy_spectrogram = np.abs(noisy_mel_signal)
    noisy_power_to_db = librosa.power_to_db(noisy_spectrogram, ref=np.max)

    output_mel_signal = output_mel.cpu()[0][0].transpose(0,1)
    output_spectrogram = np.abs(output_mel_signal)
    output_power_to_db = librosa.power_to_db(output_spectrogram, ref=np.max)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 3))
    #fig.colorbar(label='dB')

    librosa.display.specshow(original_power_to_db, sr=16000, x_axis='time', y_axis='mel', cmap='magma', hop_length=160, ax=ax1)
    librosa.display.specshow(noisy_power_to_db, sr=16000, x_axis='time', y_axis='mel', cmap='magma', hop_length=160, ax=ax2)
    librosa.display.specshow(output_power_to_db, sr=16000, x_axis='time', y_axis='mel', cmap='magma', hop_length=160, ax=ax3)

    ax1.set_title('Mel-Spectrogram Original', fontdict=dict(size=12))
    ax2.set_title('Mel-Spectrogram Corrupted', fontdict=dict(size=12))
    ax3.set_title('Mel-Spectrogram Output', fontdict=dict(size=12))

    xticks = np.arange(0, 11, 1.0)
    ax1.set_xticks(xticks)
    ax2.set_xticks(xticks)
    ax3.set_xticks(xticks)

    if not save_fig:
        plt.show()
    
    if save_fig:
        fig.savefig(f"{output_path}/{id_s}_mel-spectrograms.png")

    fig.clf()    
    return


def print_metrics_report(snr_all, sdr_all, fad):

    mean_snr = np.mean(snr_all)
    mean_sdr = np.mean(sdr_all)
    min_snr  = np.min(snr_all)
    min_sdr  = np.min(sdr_all)
    max_snr  = np.max(snr_all)
    max_sdr  = np.max(sdr_all)
    std_snr  = np.std(snr_all)
    std_sdr  = np.std(sdr_all)

    # Adjust for a better visualization
    print("\033[1m{:<12} Results \033[0m ".format(" "))
    print("{:<6} {:<9}  {:<10} ".format(' ','  SNR','  SDR'))
    print("{:<6} {:<9.4f} {:<10.4f}".format('mean',mean_snr,mean_sdr))
    print("{:<6}  {:<9.4f}  {:<10.4f}".format('std',std_snr,std_sdr))
    print("{:<6} {:<9.4f} {:<10.4f}".format('min',min_snr,min_sdr))
    print("{:<6} {:<9.4f}  {:<10.4f} ".format('max',max_snr,max_sdr))
    print("")
    print("{:<12} {:<20.4f} ".format('FAD',fad))

    return