from config import *
from dataloader import *
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
from transformers import CLIPTokenizer
from tqdm import tqdm
from encoder import *
from decoder import *
from clip import *
from diffusion import *
from ddpm import *

def get_time_embedding(times):
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32, device=times.device) / 160)
    x = times[:, None] * freqs[None]
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    idle_device = 'cpu'

    dataset = diffusion_dataset(DATA_PATH)

    train, val = random_split(dataset, [int(len(dataset)*SPLIT_RATIO), len(dataset) - int(len(dataset)*SPLIT_RATIO)])
    print("Train: ", len(train))
    print("Val: ", len(val))

    train = DataLoader(train, BATCH_SIZE, num_workers=8, shuffle=True, pin_memory=True)
    val = DataLoader(val, BATCH_SIZE, num_workers=8, shuffle=True, pin_memory=True)

    print("Loading tokenizer ...")
    tokenizer = CLIPTokenizer(vocab_file = VOCAB_FILE, merges_file = MERGES_FILE)

    unconditional_prompts = [UNCONDITIONAL_PROMPT]*BATCH_SIZE
    unconditional_tokens = tokenizer.batch_encode_plus(
                unconditional_prompts, padding = "max_length", truncation = True, max_length = MAX_TOKEN
            ).input_ids
    unconditional_tokens = torch.tensor(unconditional_tokens, dtype=torch.long, device = device)

    generator = torch.Generator(device=device)
    latents_shape = (BATCH_SIZE, 4, 64, 64)

    ENCODER = VAE_Encoder().to(device)
    DECODER = VAE_Decoder().to(device)
    CLIP_MODEL = CLIP().to(device)
    DIFFUSION = Diffusion().to(device)

    SAMPLER = DDPMSampler(generator)
    torch.cuda.empty_cache()

    Params = list(ENCODER.parameters())+list(DECODER.parameters())+list(CLIP_MODEL.parameters())+list(DIFFUSION.parameters())
    OPTIMISER = torch.optim.Adam(Params, lr = LR, weight_decay = WD)

    ENCODER.train()
    DECODER.train()
    CLIP_MODEL.train()
    DIFFUSION.train()

    min_loss = 1000000
    for batch in tqdm(train):
        conditional_prompts, images = batch

        conditional_tokens = tokenizer.batch_encode_plus(
                    conditional_prompts, padding = "max_length", truncation = True, max_length = MAX_TOKEN
                ).input_ids
        conditional_tokens = torch.tensor(conditional_tokens, dtype=torch.long, device = device)

        tokens = torch.cat([conditional_tokens, unconditional_tokens])
        context = CLIP_MODEL(tokens)

        images = images.to(device)
        
        latent_sampling_noise = torch.randn(latents_shape, generator=generator, device=device)
        latent = ENCODER(images, latent_sampling_noise)

        timesteps = torch.randint(0, 1000, (BATCH_SIZE,)).to(device)
        time_embeddings = get_time_embedding(timesteps).repeat(2, 1)
        noisy_latents = SAMPLER.add_noise(latent, timesteps)
        model_input = noisy_latents
        model_input = model_input.repeat(2, 1, 1, 1)

        

        noise_predicted = DIFFUSION(model_input, context, time_embeddings)

        output_cond, output_uncond = noise_predicted.chunk(2)
        cfg_scale = torch.rand(BATCH_SIZE)[:, None, None, None].repeat(1, 4, 64, 64).to(device)
        noise_predicted = cfg_scale * (output_cond - output_uncond) + output_uncond

        denoised_predicted = SAMPLER.remove_noise(timesteps, noisy_latents, noise_predicted)
        image_predicted = DECODER(denoised_predicted)
        
        loss = F.mse_loss(image_predicted, images)
        loss.backward()
        OPTIMISER.step()

        OPTIMISER.zero_grad()

        print('loss: ', loss.item())
        if loss.item() < min_loss:
            min_loss = loss.item()
            print('min loss: ', loss.item())
            torch.save(ENCODER.state_dict(), 'checkpoints/encoder.pth')
            torch.save(DECODER.state_dict(), 'checkpoints/decoder.pth')
            torch.save(CLIP_MODEL.state_dict(), 'checkpoints/clip.pth')
            torch.save(DIFFUSION.state_dict(), 'checkpoints/diffusion.pth')
        # print(out.shape)
        # break