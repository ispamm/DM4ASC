import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

#torch.manual_seed(1999)
#np.random.seed(1999)

@torch.no_grad()
def inference_with_mask(self, prompt, inference_scheduler, num_steps=20, guidance_scale=3, num_samples_per_prompt=1,
                  disable_progress=True, mask=None, original=None, sigma_y=0, travel_length=10, noisy_prompts=False, psnr_prompts=20):

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

    device = self.text_encoder.device

    ### START
    assert mask is not None, "A mask is needed"
    assert original is not None, "The original audio is needed"
    # assert sigma_y >= 0 and sigma_y <= 1, "sigma_y must be between 0 and 1 included"
    mask = mask.to(device)
    original = original.to(device)
    ### END

    classifier_free_guidance = guidance_scale > 1.0
    batch_size = len(prompt) * num_samples_per_prompt

    if classifier_free_guidance:
        prompt_embeds, boolean_prompt_mask = self.encode_text_classifier_free(prompt, num_samples_per_prompt)
    else:
        prompt_embeds, boolean_prompt_mask = self.encode_text(prompt)
        prompt_embeds = prompt_embeds.repeat_interleave(num_samples_per_prompt, 0)
        boolean_prompt_mask = boolean_prompt_mask.repeat_interleave(num_samples_per_prompt, 0)

    if noisy_prompts:
        lmin, lmax = torch.min(prompt_embeds), torch.max(prompt_embeds)
        noise_embeds_prompts = (prompt_embeds - lmin) / (lmax - lmin)
        noise_prompts = torch.randn_like(noise_embeds_prompts) * get_psnr(psnr_prompts)
        prompt_embeds = noise_embeds_prompts + noise_prompts
        prompt_embeds = (prompt_embeds * (lmax - lmin)) + lmin
        
        if sigma_y == -1:
            sigma_y = (torch.max(prompt_embeds)-torch.min(prompt_embeds))*torch.std(prompt_embeds)

    inference_scheduler.set_timesteps(num_steps, device=device)
    timesteps = inference_scheduler.timesteps

    num_channels_latents = self.unet.in_channels
    latents = self.prepare_latents(batch_size, inference_scheduler, num_channels_latents, prompt_embeds.dtype, device)

    num_warmup_steps = len(timesteps) - num_steps * inference_scheduler.order
    progress_bar = tqdm(range(num_steps), disable=disable_progress)

    for i, t in enumerate(timesteps):

        L = min(num_steps - 1 - t, travel_length)

        ns = self.noise_scheduler

        beta_t = ns.betas[t]
        alpha_t = ns.alphas[t]
        alpha_cumprod_t = ns.alphas_cumprod[t]
        alpha_cumprod_tL = ns.alphas_cumprod[t+L]

        alpha_cumprod_L = alpha_cumprod_tL / alpha_cumprod_t
        noise = torch.randn_like(latents)

        latents = (alpha_cumprod_L ** 0.5) * latents + ((1 - alpha_cumprod_L) ** 0.5) * noise

        for j in range(L, -1, -1):

            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if classifier_free_guidance else latents
            latent_model_input = inference_scheduler.scale_model_input(latent_model_input, t+j)

            noise_pred = self.unet(
                latent_model_input, t+j, encoder_hidden_states=prompt_embeds,
                encoder_attention_mask=boolean_prompt_mask
            ).sample

            # perform guidance
            if classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            ### START

            # compute the previous noisy sample x_t -> x_t-1
            # latents = inference_scheduler.step(noise_pred, t, latents).prev_sample

            ns = self.noise_scheduler

            beta_tj = ns.betas[t+j]
            alpha_tj = ns.alphas[t+j]
            alpha_cumprod_tj = ns.alphas_cumprod[t+j]
            alpha_cumprod_tj_prev = ns.alphas_cumprod[t+j-1] if (t + j - 1) >= 0 else ns.one

            x_0tj = (alpha_cumprod_tj ** 0.5) * latents - ((1 - alpha_cumprod_tj) ** 0.5) * noise_pred

            a_tj = ((alpha_cumprod_tj_prev ** 0.5) * beta_tj) / (1. - alpha_cumprod_tj)
            c3 = (((1 - alpha_cumprod_tj_prev) / (1 - alpha_cumprod_tj)) * beta_tj) ** 0.5 # sigma_t
            
            if c3 >= (a_tj * sigma_y):
                lambda_tj = 1.
                gamma_tj = (c3 ** 2) - ((a_tj * sigma_y) ** 2)
            else:
                lambda_tj = c3 / (a_tj * sigma_y)
                gamma_tj = 0.
                
                   

            # formula 13 DDNM
            x_0tj_hat = x_0tj - (lambda_tj * (mask * ((mask * x_0tj) - original)))

            c1 = ((alpha_cumprod_tj_prev ** 0.5) * beta_tj) / (1 - alpha_cumprod_tj)
            c2 = ((alpha_tj ** 0.5) * (1 - alpha_cumprod_tj_prev)) / (1 - alpha_cumprod_tj)

            noise = torch.randn_like(latents) if t + j > 0 else torch.zeros_like(latents)

            # formula 7 DDPM, formulae 11 and 14 DDNM
            latents = c1 * x_0tj_hat + c2 * latents + (gamma_tj ** 0.5) * noise
            ### END

        # call the callback, if provided
        if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % inference_scheduler.order == 0):
            progress_bar.update(1)

    if self.set_from == "pre-trained":
        latents = self.group_out(latents.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()
    return latents
