import torch 
import torch.nn as nn
from torch.nn import functional as F
from attention import selfAttention 

# NOTE on group normalization 
# similar to layer norm, the statistics are computed over the feature dimension but grouped into groups
# i.e. if we have 4 features: f1, f2, f3, f4 per input data, then we compute 2 groups of statistics: 
# (mu1, mu2), (sigma1, sigma2) where mu1 = (f1 + f2)/2, mu2 = (f3 + f4)/2, sigma1 = sqrt((f1 - mu1)^2 + (f2 - mu1)^2), sigma2 = sqrt((f3 - mu2)^2 + (f4 - mu2)^2)
# and then we normalize the input data similarly to layer norm per group. 
# the reason is that these features come from convolutions which look at local areas of the image
# so two features close to each other are more correlated than two features far away from each other (so make them have the same distribution)

class VAE_AttentionBlock(nn.Module):

    def __init__(self, channels: int):
        super(VAE_AttentionBlock, self).__init__()
        self.group_norm = nn.GroupNorm(32, channels)
        self.attention = selfAttention(1, channels) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, features, height, width)
        residual = x 

        n, c, h, w = x.shape 
        
        # (batch_size, features, height, width) -> (batch_size, features, height*width)
        
        x = x.view(n, c, h*w)

        # (batch_size, features, height*width) -> (batch_size, height*width, features)

        x = x.transpose(-1, -2)
        # (batch_size, height*width, features) -> (batch_size, height*width, features)

        x = self.attention(x)
        
        # (batch_size, height*width, features) -> (batch_size, features, height*width)
        x = x.transpose(-1, -2) 

        x = x.view(n, c, h, w)

        x = x + residual
        
        return x

class VAE_RedisualBlock(nn.Module): 
    def __init__(self, in_channels, out_channels): 
        super(VAE_RedisualBlock, self).__init__()
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels: 
            self.residual_layer = nn.Identity()
        else: 
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, in_channels, height, width)
        residue = x 

        x = self.groupnorm_1(x)

        x = F.silu(x)

        x = self.conv_1(x)

        x = self.groupnorm_2(x) 

        x = F.silu(x) 

        x = self.conv_2(x) 

        return x + self.residual_layer(residue) 
    

class VAE_Decoder(nn.Sequential):
    def __init__(self):
        super(VAE_Decoder, self).__init__(
            nn.Conv2d(4, 4, kernel_size=1, padding=0),
            nn.Conv2d(4, 512, kernel_size=3, padding=1),
            VAE_RedisualBlock(512, 512),
            VAE_AttentionBlock(512),
            VAE_RedisualBlock(512, 512),
            VAE_RedisualBlock(512, 512),
            VAE_RedisualBlock(512, 512),
            # (batch_size, 512, height/8, width/8) -> (batch_size, 512, height/8, width/8)
            VAE_RedisualBlock(512, 512),

            # (batch_size, 512, height/8, width/8) -> (batch_size, 512, height/4, width/4)
            nn.Upsample(scale_factor=2)

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            VAE_RedisualBlock(512, 512),
            VAE_RedisualBlock(512, 512),
            VAE_RedisualBlock(512, 512),

            # (batch_size, 512, height/4, width/4) -> (batch_size, 512, height/2, width/2)
            nn.Upsample(scale_factor=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),

            VAE_RedisualBlock(512, 256),
            VAE_RedisualBlock(256, 256),
            VAE_RedisualBlock(256, 256),

            # (batch_size, 256, height/2, width/2) -> (batch_size, 256, height, width)
            nn.Upsample(scale_factor=2),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),

            VAE_RedisualBlock(256, 128),
            VAE_RedisualBlock(128, 128),
            VAE_RedisualBlock(128, 128),

            nn.GroupNorm(32, 128),

            nn.Silu(),

            # (batch_size, 128, height, width) -> (batch_size, 3, height, width)
            nn.Conv2d(128, 3, kernel_size=3, padding=1)
        )

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # x: (batch_size, 4, height/8, width/8)
        x/=0.18215

        for module in self:
            x = module(x)
        
        # (batch_size, 3, height, width)
        return x        