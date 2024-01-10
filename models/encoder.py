import torch 
import torch.nn as nn
import torch.nn.functional as F

from decoder import VAE_AttentionBlock, VAE_RedisualBlock 


class VAE_Encoder(nn.Sequential):
    def __init__(self):
        super(VAE_Encoder, self).__init__(
            # Starting from the initial image, keep decreasing the size of the image but increase 
            # the size of the feature size.
            
            # (batch_size, channel, height, width) -> (batch_size, 128, height, width). 
            # add padding to keep the same size
            nn.Conv2d(3, 128, kernel_size=3, padding=1),

            # (batch_size, 128, height, width) -> (batch_size, 128, height, width)
            VAE_RedisualBlock(128, 128),

            # (batch_size, 128, height, width) -> (batch_size, 128, height, width)
            VAE_RedisualBlock(128, 128),

            # (batch_size, 128, height, width) -> (batch_size, 128, height/2, width/2)
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),

            # (batch_size, 128, height/2, width/2) -> (batch_size, 256, height/2, width/2)
            VAE_RedisualBlock(128, 256),

            # (batch_size, 256, height/2, width/2) -> (batch_size, 256, height/2, width/2)
            VAE_RedisualBlock(256, 256),
        
            # (batch_size, 256, height, width) -> (batch_size, 256, height/4, width/4)
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),     

            # (batch_size, 256, height/4, width/4) -> (batch_size, 512, height/4, width/4)
            VAE_RedisualBlock(256, 512),

            # (batch_size, 512, height/4, width/4) -> (batch_size, 512, height/4, width/4)
            VAE_RedisualBlock(512, 512),

            # (batch_size, 512, height/4, width/4) -> (batch_size, 512, height/8, width/8)
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),

            # (batch_size, 512, height/8, width/8) -> (batch_size, 512, height/8, width/8)
            VAE_RedisualBlock(512, 512),       

            # (batch_size, 512, height/8, width/8) -> (batch_size, 512, height/8, width/8)
            VAE_RedisualBlock(512, 512),

            # (batch_size, 512, height/8, width/8) -> (batch_size, 512, height/8, width/8)

            VAE_RedisualBlock(512, 512),

            # (batch_size, 512, height/8, width/8) -> (batch_size, 512, height/8, width/8
            # add attention block which is used to capture global information between pixels
            # (the convolutions above only capture local information)  
            VAE_AttentionBlock(512), 

            # (batch_size, 512, height/8, width/8) -> (batch_size, 512, height/8, width/8)
            VAE_RedisualBlock(512, 512),

            # (batch_size, 512, height/8, width/8) -> (batch_size, 512, height/8, width/8)
            nn.GroupNorm(32, 512), 

            # (batch_size, 512, height/8, width/8) -> (batch_size, 512, height/8, width/8)
            nn.Silu(),

            # (batch_size, 512, height/8, width/8) -> (batch_size, 8, height/8, width/8)
            nn.Conv2d(512, 8, kernel_size=3, padding=1), 

            # (batch_size, 8, height/8, width/8) -> (batch_size, 8, height/8, width/8)
            nn.Conv2d(8, 8, kernel_size=1, padding=0),
        )

    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, channel, height, width)
        # noise: (batch_size, channel, height/8, width/8) 

        for module in self:
            if getattr(module, 'stride', False) == (2,2):
                # (padding_left, padding_right, padding_top, padding_bottom)
                x = F.pad(x, (0, 1, 0, 1))
            x = module(x)

        # (batch_size, 8, height/8, width/8) -> 2 tensors of size (batch_size, 4,  height/8, width/8)
        mean, log_var = torch.chunk(x, 2, dim=1)

        # (batch_size, 4, height/8 * width/8) -> (batch_size, 4, height/8 , width/8)
        log_var = torch.clamp(log_var, min=-30, max=20)

        # (batch_size, 4, height/8 , width/8) -> (batch_size, 4, height/8 , width/8)
        var = torch.exp(log_var)

        # (batch_size, 4, height/8 , width/8) -> (batch_size, 4, height/8 , width/8)
        stdev  = torch.sqrt(var) 

        # sample output using the reparameterization trick
        # Z ~ N(0, 1) -> X ~ N(mean, var) = mean + stdev * Z

        x = mean + stdev * noise

        # scale the output by a constant 
        x = x * 0.18215

        return x 
