import torch.nn as nn
from transformers import ViTModel, ViTMAEModel, CLIPVisionModel, CLIPTokenizer, CLIPTextModel, ResNetModel
from ldm.modules.diffusionmodules.openaimodel import TimestepEmbedSequential, ResBlock, Downsample, Upsample


from ldm.modules.diffusionmodules.util import linear, conv_nd, timestep_embedding


class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError


class IdentityEncoder(AbstractEncoder):
    def encode(self, x):
        return x
    def decode(self, x):
        return x
    
class LayoutEncoder(nn.Module):
    def __init__(
            self,
            in_channels=3,
            base_channels=32,
            vae_num_down=2,
            unet_num_down=3,
            unet_channels=224,
            block_repeat=2, # (The number of features with the same resolution in unet) - 1
            dropout=0,
            dims=2,
            use_checkpoint=True
    ):
        super(LayoutEncoder, self).__init__()
        self.unet_channels = unet_channels
        time_embed_dim = unet_channels*4
        self.time_embed = nn.Sequential(
            linear(unet_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )
        self.vae_down = []
        self.input_block = TimestepEmbedSequential(
            conv_nd(dims, in_channels, base_channels, 3, padding=1),
            nn.SiLU())
        
        for i in range(vae_num_down):
            in_ch = int(base_channels*(2**i))
            out_ch = in_ch*2
            self.vae_down.append(TimestepEmbedSequential(ResBlock(in_ch, time_embed_dim, dropout, out_ch, dims=dims, use_checkpoint=use_checkpoint)))
            self.vae_down.append(TimestepEmbedSequential(Downsample(out_ch, True, dims=dims)))

        self.vae_down.append(TimestepEmbedSequential(ResBlock(out_ch, time_embed_dim, dropout, out_ch*2, dims=dims, use_checkpoint=use_checkpoint)))

        self.unet_down = []
        self.unet_down.append(TimestepEmbedSequential(ResBlock(out_ch*2, time_embed_dim, dropout, unet_channels, dims=dims, use_checkpoint=use_checkpoint)))
        self.unet_down.append(TimestepEmbedSequential(ResBlock(unet_channels, time_embed_dim, dropout, unet_channels, dims=dims, use_checkpoint=use_checkpoint)))

        for i in range(unet_num_down):
            in_ch = int(unet_channels*(2**i))
            for j in range(block_repeat):
                
                if i == unet_num_down-1 or j != block_repeat-1:
                    self.unet_down.append(TimestepEmbedSequential(ResBlock(in_ch, time_embed_dim, dropout, dims=dims, use_checkpoint=use_checkpoint)))
                else:
                    self.unet_down.append(TimestepEmbedSequential(Downsample(in_ch, True, dims=dims)))

            if i != unet_num_down-1:
                self.unet_down.append(TimestepEmbedSequential(ResBlock(in_ch, time_embed_dim, dropout, in_ch*2, dims=dims, use_checkpoint=use_checkpoint)))

        self.vae_down = nn.ModuleList(self.vae_down)
        self.unet_down = nn.ModuleList(self.unet_down)

    def forward(self, x, t):
        t_emb = timestep_embedding(t, self.unet_channels, repeat_only=False)
        emb = self.time_embed(t_emb)
        x = self.input_block(x, emb)
        for module in self.vae_down:
            x = module(x, emb)

        outs = []
        for module in self.unet_down:
            x = module(x, emb)
            outs.append(x)
        return outs
   
class LayoutEnDecoder(nn.Module):
    def __init__(
            self,
            in_channels=3,
            base_channels=32,
            vae_num_down=2,
            unet_num_down=3,
            unet_channels=224,
            block_repeat=2, # (The number of features with the same resolution in unet) - 1
            dropout=0,
            dims=2,
            use_checkpoint=True
    ):
        super(LayoutEnDecoder, self).__init__()
        self.unet_channels = unet_channels
        time_embed_dim = unet_channels*4
        self.time_embed = nn.Sequential(
            linear(unet_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )
        self.vae_down = []
        self.input_block = TimestepEmbedSequential(
            conv_nd(dims, in_channels, base_channels, 3, padding=1),
            nn.SiLU())
        
        for i in range(vae_num_down):
            in_ch = int(base_channels*(2**i))
            out_ch = in_ch*2
            self.vae_down.append(TimestepEmbedSequential(ResBlock(in_ch, time_embed_dim, dropout, out_ch, dims=dims, use_checkpoint=use_checkpoint)))
            self.vae_down.append(TimestepEmbedSequential(Downsample(out_ch, True, dims=dims)))

        self.vae_down.append(TimestepEmbedSequential(ResBlock(out_ch, time_embed_dim, dropout, out_ch*2, dims=dims, use_checkpoint=use_checkpoint)))

        self.unet_down = []
        self.unet_down.append(TimestepEmbedSequential(ResBlock(out_ch*2, time_embed_dim, dropout, unet_channels, dims=dims, use_checkpoint=use_checkpoint)))
        self.unet_down.append(TimestepEmbedSequential(ResBlock(unet_channels, time_embed_dim, dropout, unet_channels, dims=dims, use_checkpoint=use_checkpoint)))

        for i in range(unet_num_down):
            in_ch = int(unet_channels*(2**i))
            for j in range(block_repeat):
                
                if i == unet_num_down-1 or j != block_repeat-1:
                    self.unet_down.append(TimestepEmbedSequential(ResBlock(in_ch, time_embed_dim, dropout, dims=dims, use_checkpoint=use_checkpoint)))
                else:
                    self.unet_down.append(TimestepEmbedSequential(Downsample(in_ch, True, dims=dims)))

            if i != unet_num_down-1:
                self.unet_down.append(TimestepEmbedSequential(ResBlock(in_ch, time_embed_dim, dropout, in_ch*2, dims=dims, use_checkpoint=use_checkpoint)))

        self.unet_up = []

        for i in reversed(range(unet_num_down)):
            in_ch = int(unet_channels*(2**i))
            for j in range(block_repeat):
                
                if i == 0 or j != block_repeat-1:
                    self.unet_up.append(TimestepEmbedSequential(ResBlock(in_ch, time_embed_dim, dropout, dims=dims, use_checkpoint=use_checkpoint)))
                else:
                    self.unet_up.append(TimestepEmbedSequential(ResBlock(in_ch, time_embed_dim, dropout, in_ch//2, dims=dims, use_checkpoint=use_checkpoint)))

            if i != 0:
                self.unet_up.append(TimestepEmbedSequential(Upsample(in_ch//2, True, dims=dims)))

        self.vae_down = nn.ModuleList(self.vae_down)
        self.unet_down = nn.ModuleList(self.unet_down)
        self.unet_up = nn.ModuleList(self.unet_up)

    def forward(self, x, t):
        t_emb = timestep_embedding(t, self.unet_channels, repeat_only=False)
        emb = self.time_embed(t_emb)
        x = self.input_block(x, emb)
        for module in self.vae_down:
            x = module(x, emb)

        outs = []
        for module in self.unet_down:
            x = module(x, emb)
            outs.append(x)
        for module in self.unet_up:
            x = module(x, emb)
            outs.append(x)
        return outs


class FrozenRes101(AbstractEncoder):
    """Uses the Res101 encoder for controlnet (from huggingface)"""
    def __init__(self, version="microsoft/resnet-101", device="cuda", freeze=True): 
        super().__init__()
        self.transformer = ResNetModel.from_pretrained(version)
        self.device = device
        if freeze:
            self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        #self.train = disabled_train
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.transformer(x).last_hidden_state

        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = x.squeeze(-1).squeeze(-1)

        #x = rearrange(x, 'b c h w -> b (h w) c')

        return x.unsqueeze(1)

    def encode(self, x):
        return self(x)
    

    
class FrozenViTMAE(AbstractEncoder):
    """Uses the ViT_MAE encoder for controlnet (from huggingface)"""
    def __init__(self, version="facebook/vit-mae-base", device="cuda", mask_ratio=0., freeze=True): 
        super().__init__()
        self.transformer = ViTMAEModel.from_pretrained(version)
        self.device = device
        self.transformer.config.mask_ratio = mask_ratio
        if freeze:
            self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        #self.train = disabled_train
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.transformer(x).last_hidden_state[:, 0, :]
        return x.unsqueeze(1)

    def encode(self, x):
        return self(x)
    
class FrozenViTMAE_layers(AbstractEncoder):
    """Uses the ViT_MAE encoder for controlnet (from huggingface)"""
    def __init__(self, layer, version="facebook/vit-mae-base", device="cuda", mask_ratio=0., freeze=True): 
        super().__init__()
        self.transformer = ViTMAEModel.from_pretrained(version)
        self.layer = layer
        self.device = device
        self.transformer.config.mask_ratio = mask_ratio
        self.transformer.config.output_hidden_states = True
        if freeze:
            self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        #self.train = disabled_train
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.transformer(x).hidden_states[self.layer][:, 0, :]
        return x.unsqueeze(1)

    def encode(self, x):
        return self(x)

    
class FrozenViTDINO(AbstractEncoder):
    """Uses the ViT_DINO encoder for controlnet (from huggingface)"""
    def __init__(self, version="facebook/dino-vitb16", device="cuda", freeze=True): 
        super().__init__()
        self.transformer = ViTModel.from_pretrained(version)
        self.device = device
        if freeze:
            self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        #self.train = disabled_train
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.transformer(x).last_hidden_state[:, 0, :]
        return x.unsqueeze(1)

    def encode(self, x):
        return self(x)
    
class FrozenViTCLIP(AbstractEncoder):
    """Uses the ViT_CLIP encoder for controlnet (from huggingface)"""
    def __init__(self, version="openai/clip-vit-base-patch16", device="cuda", freeze=True): 
        super().__init__()
        self.transformer = CLIPVisionModel.from_pretrained(version)
        self.device = device
        if freeze:
            self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        #self.train = disabled_train
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.transformer(x).last_hidden_state[:, 0, :]
        return x.unsqueeze(1)

    def encode(self, x):
        return self(x)
    
class FrozenCLIPEmbedder(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from huggingface)"""
    LAYERS = [
        "last",
        "pooled",
        "hidden"
    ]
    def __init__(self, version="openai/clip-vit-large-patch14", device="cuda", max_length=77,
                 freeze=True, layer="last", layer_idx=None):  # clip-vit-base-patch32
        super().__init__()
        assert layer in self.LAYERS
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPTextModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        self.layer_idx = layer_idx
        if layer == "hidden":
            assert layer_idx is not None
            assert 0 <= abs(layer_idx) <= 12

    def freeze(self):
        self.transformer = self.transformer.eval()
        #self.train = disabled_train
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(input_ids=tokens, output_hidden_states=self.layer=="hidden")
        if self.layer == "last":
            z = outputs.last_hidden_state
        elif self.layer == "pooled":
            z = outputs.pooler_output[:, None, :]
        else:
            z = outputs.hidden_states[self.layer_idx]
        return z

    def encode(self, text):
        return self(text)
