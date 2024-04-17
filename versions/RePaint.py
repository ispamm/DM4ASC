import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm



@torch.no_grad()
def inference_with_mask(self, prompt, inference_scheduler, guidance_scale=3, num_samples_per_prompt=1,
                  disable_progress=True, mask=None, original=None, t_T=1000, jump_len=10, jump_n_sample=10, noisy_prompts=False, psnr_prompts=20):

    def compute_steps(t_T=1000, jump_len=10, jump_n_sample=10):
        jumps = {}
        for j in range(0, t_T - jump_len, jump_len):
            jumps[j] = jump_n_sample - 1

        t = t_T
        ts = []

        while t >= 1:
            t = t-1
            ts.append(t)

            if jumps.get(t, 0) > 0:
                jumps[t] = jumps[t] - 1
                for _ in range(jump_len):
                    t = t + 1
                    ts.append(t)

        ts.append(-1)
        return ts


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
    
    prompt_embeds2 = prompt_embeds.flatten()
    print(f"Prompts | Min: {torch.min(prompt_embeds2)}, Max: {torch.max(prompt_embeds2)}, std: {torch.std(prompt_embeds)}")

    
    if noisy_prompts:
        lmin, lmax = torch.min(prompt_embeds), torch.max(prompt_embeds)
        noise_embeds_prompts = (prompt_embeds - lmin) / (lmax - lmin)
        noise_prompts = torch.randn_like(noise_embeds_prompts) * get_psnr(psnr_prompts)
        prompt_embeds = noise_embeds_prompts + noise_prompts
        prompt_embeds = (prompt_embeds * (lmax - lmin)) + lmin
    

    inference_scheduler.set_timesteps(t_T, device=device) # serve quando uso inference_scheduler per fare step
    timesteps = compute_steps(t_T=t_T, jump_len=jump_len, jump_n_sample=jump_n_sample)
    num_steps = len(timesteps)

    num_channels_latents = self.unet.in_channels
    latents = self.prepare_latents(batch_size, inference_scheduler, num_channels_latents, prompt_embeds.dtype, device)

    num_warmup_steps = len(timesteps) - num_steps * inference_scheduler.order
    progress_bar = tqdm(range(num_steps), disable=disable_progress)

    for i, (t_last, t_cur) in enumerate(zip(timesteps[:-1], timesteps[1:])):

        if t_cur < t_last:

            t_last = torch.tensor(t_last, dtype=torch.int64).to(device)

            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if classifier_free_guidance else latents
            latent_model_input = inference_scheduler.scale_model_input(latent_model_input, t_last) # returns latent_model_input

            noise_pred = self.unet(
                latent_model_input, t_last, encoder_hidden_states=prompt_embeds,
                encoder_attention_mask=boolean_prompt_mask
            ).sample

            # perform guidance
            if classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = inference_scheduler.step(noise_pred, t_last, latents).prev_sample

            ### START
            noise = torch.randn_like(original) if t_last>0 else torch.zeros_like(latents)
            latent_orig = self.noise_scheduler.add_noise(original, noise, t_last) if t_last>0 else original

            latents = (latent_orig * mask + latents * (1.0 - mask))

        else:

            ns = self.noise_scheduler
            beta_t = ns.betas[t_last+1] # t_last + 1 = t_cur

            noise = torch.randn_like(latents)

            latents = ((1 - beta_t) ** 0.5) * latents + (beta_t ** 0.5) * noise
            ### END


        # call the callback, if provided
        if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % inference_scheduler.order == 0):
            progress_bar.update(1)

    if self.set_from == "pre-trained":
        latents = self.group_out(latents.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()

    return latents