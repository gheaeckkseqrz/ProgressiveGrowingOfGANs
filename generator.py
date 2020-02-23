import torch
import encoder
import decoder

class Generator(torch.nn.Module):
    def __init__(self, nc):
        super(Generator, self).__init__()
        self.encoder = encoder.Encoder(nc)
        self.decoder = decoder.Decoder(nc)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

t = torch.Tensor(1, 3, 64, 64)
print(t.shape)
e = Generator(8)
#print(e)
print(e(t).shape)
