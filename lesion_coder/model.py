import torch
import torch.nn as nn


class BaseAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # For encoder load pretrained VGG19 model and remove layers upto relu4_1
        torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
        self.vgg = torch.hub.load('pytorch/vision:v0.10.1', 'vgg19', pretrained=True)
        self.vgg = nn.Sequential(*(list(self.vgg.features.children())+[self.vgg.avgpool]))
        # Create AdaIN layer

        # Use Sequential to define decoder [Just reverse of vgg with pooling replaced by nearest neigbour upscaling]
        self.dec = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
        nn.Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect' ),
        nn.ReLU(),
        nn.Upsample(scale_factor=2,mode='nearest'),
        nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
        nn.ReLU(),
        nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
        nn.ReLU(),
        nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
        nn.ReLU(),
        nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
        nn.ReLU(),
        nn.Upsample(scale_factor=2,mode='nearest'),
        nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
        nn.ReLU(),
        nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
        nn.ReLU(),
        nn.Upsample(scale_factor=2,mode='nearest'),
        nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
        nn.ReLU(),
        nn.Conv2d(64, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect')
        #nn.ReLU() #Maybe change to a sigmoid to get into 0,1 range?
        )

    def encode(self,x):
        return self.vgg(x)

    def decode(self,z):
        return self.dec(z)

    def forward(self, x):
        """ x is a image containing content information, y is an image
        containing style information"""
        # Compute content and style embeddings
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return decoded