import torch
import numpy as np

class DDPMSampler:
    def __init__(self, generator: torch.Generator, num_training_steps=1000, beta_start: float = 0.00085, beta_end: float = 0.0120):
        self.betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_training_steps, dtype=torch.float32) ** 2
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.one = torch.tensor(1.0)

        self.generator = generator

        self.num_train_timesteps = num_training_steps
        self.timesteps = torch.from_numpy(np.arange(0, num_training_steps)[::-1].copy())

    def remove_noise(self, timestep: int, latents: torch.Tensor, model_output: torch.Tensor):
        self.alphas_cumprod = self.alphas_cumprod.to(latents.device)
        alpha_prod_t = self.alphas_cumprod[timestep]
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (latents - beta_prod_t[:, None, None, None].expand(-1, 4, 64, 64) ** (0.5) * model_output) / alpha_prod_t[:, None, None, None].expand(-1, 4, 64, 64) ** (0.5)
        return pred_original_sample

    
    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
        alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        noise = torch.randn(original_samples.shape, generator=self.generator, device=original_samples.device, dtype=original_samples.dtype)
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples
if __name__ == "__main__":
    generator = torch.Generator(device='cuda')
    image = torch.randn((4, 4, 64, 64), generator=generator, device='cuda')
    timesteps = torch.randint(0, 1000, (4,)).to('cuda')
    SAMPLER = DDPMSampler(generator)
    SAMPLER.set_inference_timesteps(1000)
    noisy_latents = SAMPLER.add_noise(image, timesteps)
    denoised = SAMPLER.step(timesteps, noisy_latents, image)
    print(denoised.shape)
        

    