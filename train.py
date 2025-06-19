from config import *
from dataloader import *
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
from transformers import CLIPTokenizer
from tqdm import tqdm
from encoder import *
from decoder import *
from clip import *

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

    generator = torch.Generator(device=device)
    latents_shape = (BATCH_SIZE, 4, 64, 64)

    ENCODER = VAE_Encoder().to(device)
    DECODER = VAE_Decoder().to(device)
    CLIP_MODEL = CLIP().to(device)

    torch.cuda.empty_cache()
    for batch in tqdm(train):
        prompt, images = batch
        tokens = tokenizer.batch_encode_plus(
                    prompt, padding = "max_length", truncation = True, max_length = MAX_TOKEN
                ).input_ids
        tokens = torch.tensor(tokens, dtype=torch.long, device = device)
        images = images.to(device)

        
        noise = torch.randn(latents_shape, generator=generator, device=device)
        image_features = ENCODER(images, noise)
        image = DECODER(image_features)
        text_features = CLIP_MODEL(tokens)
        # print(out.shape)
        # break