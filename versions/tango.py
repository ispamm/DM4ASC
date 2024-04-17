import os

import json
import torch
from tqdm import tqdm
from huggingface_hub import snapshot_download
from models import AudioDiffusion, DDPMScheduler
from audioldm.audio.stft import TacotronSTFT
from audioldm.variational_autoencoder import AutoencoderKL
import IPython
import soundfile as sf

class Tango:
    def __init__(self, path_to_tango, path_to_weights='', device="cuda:0"):

        if path_to_weights=='':
            path_to_weights = os.path.join(path_to_tango, 'weights', 'tango-full-ft-audiocaps')

        vae_config = json.load(open("{}/vae_config.json".format(path_to_weights)))
        stft_config = json.load(open("{}/stft_config.json".format(path_to_weights)))
        main_config = json.load(open("{}/main_config.json".format(path_to_weights)))

        self.vae = AutoencoderKL(**vae_config).to(device)
        self.stft = TacotronSTFT(**stft_config).to(device)
        self.model = AudioDiffusion(**main_config).to(device)

        vae_weights = torch.load("{}/pytorch_model_vae.bin".format(path_to_weights), map_location=device)
        stft_weights = torch.load("{}/pytorch_model_stft.bin".format(path_to_weights), map_location=device)
        main_weights = torch.load("{}/pytorch_model_main.bin".format(path_to_weights), map_location=device)

        self.vae.load_state_dict(vae_weights)
        self.stft.load_state_dict(stft_weights)
        self.model.load_state_dict(main_weights)

        print ("Successfully loaded checkpoint from:", path_to_weights)

        self.vae.eval()
        self.stft.eval()
        self.model.eval()

        self.scheduler = DDPMScheduler.from_pretrained(main_config["scheduler_name"], subfolder="scheduler")

    def chunks(self, lst, n):
        """ Yield successive n-sized chunks from a list. """
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    def generate(self, prompt, steps=100, guidance=3, samples=1, disable_progress=True):
        """ Generate audio for a single prompt string. """
        with torch.no_grad():
            latents = self.model.inference([prompt], self.scheduler, steps, guidance, samples, disable_progress=disable_progress)
            mel = self.vae.decode_first_stage(latents)
            wave = self.vae.decode_to_waveform(mel)
        return wave[0]

    def generate_for_batch(self, prompts, steps=100, guidance=3, samples=1, batch_size=8, disable_progress=True):
        """ Generate audio for a list of prompt strings. """
        outputs = []
        for k in tqdm(range(0, len(prompts), batch_size)):
            batch = prompts[k: k+batch_size]
            with torch.no_grad():
                latents = self.model.inference(batch, self.scheduler, steps, guidance, samples, disable_progress=disable_progress)
                mel = self.vae.decode_first_stage(latents)
                wave = self.vae.decode_to_waveform(mel)
                outputs += [item for item in wave]
        if samples == 1:
            return outputs
        else:
            return list(self.chunks(outputs, samples))