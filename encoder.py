import torch
from torch import nn
from torch.nn import functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock

class VAE_Encoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            VAE_ResidualBlock(128, 128),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),
            VAE_ResidualBlock(128, 128),
            VAE_ResidualBlock(128, 128),
            VAE_AttentionBlock(128),
            VAE_ResidualBlock(128, 128),
            nn.GroupNorm(32, 128),
            nn.SiLU(),
            nn.Conv2d(128, 8, kernel_size=3, padding=1),
            nn.Conv2d(8, 8, kernel_size=1, padding=0), 
        )

    def forward(self, x, noise):
        for module in self:

            if getattr(module, 'stride', None) == (2, 2):
                x = F.pad(x, (0, 1, 0, 1))
            
            x = module(x)
        mean, log_variance = torch.chunk(x, 2, dim=1)
        log_variance = torch.clamp(log_variance, -30, 20)
        variance = log_variance.exp()
        stdev = variance.sqrt()
        x = mean + stdev * noise
        return x
    
if __name__ == "__main__":
    generator = torch.Generator(device='cpu')
    latents_shape = (2, 4, 64, 64)
    noise = torch.randn(latents_shape, generator=generator, device='cpu')
    image = torch.randn((2, 3, 128, 128), generator=generator, device='cpu')
    model = VAE_Encoder()
    print(model(image, noise).shape)


