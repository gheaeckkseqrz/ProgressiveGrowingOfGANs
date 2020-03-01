import torch
import encoder
import decoder

class Generator(torch.nn.Module):
    def __init__(self, nc):
        super(Generator, self).__init__()
        self.encoder = encoder.Encoder(nc)
        self.decoder = decoder.Decoder(nc)
        
    def forward(self, x):
        code = self.encoder(x)
        return self.decoder(code), code

    def next_step(self):
        self.encoder.next_step()
        self.decoder.next_step()
