import torch

class DecoderBlock(torch.nn.Module):
    def __init__(self, nc):
        super(DecoderBlock, self).__init__()
        self.c1 = torch.nn.Conv2d(nc, nc, 3, 1, 1)
        self.c2 = torch.nn.Conv2d(nc, nc, 3, 1, 1)
        self.c3 = torch.nn.Conv2d(nc, nc, 3, 1, 1)
        self.c4 = torch.nn.Conv2d(nc, nc, 3, 1, 1)
        self.b1 = torch.nn.BatchNorm2d(nc)
        self.b2 = torch.nn.BatchNorm2d(nc)
        self.b3 = torch.nn.BatchNorm2d(nc)
        self.b4 = torch.nn.BatchNorm2d(nc)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2)
        x = self.b1(torch.nn.functional.relu(self.c1(x)))
        x = self.b2(torch.nn.functional.relu(self.c2(x)))
        x = self.b3(torch.nn.functional.relu(self.c3(x)))
        x = self.b4(torch.nn.functional.relu(self.c4(x)))
        return x

class Decoder(torch.nn.Module):
    def __init__(self, nc):
        super(Decoder, self).__init__()
        self.b1 = DecoderBlock(nc)
        self.b2 = DecoderBlock(nc)
        self.b3 = DecoderBlock(nc)
        self.b4 = DecoderBlock(nc)
        self.b5 = DecoderBlock(nc)
        self.torgb = torch.nn.Conv2d(nc, 3, 1, 1, 0)
        
    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        x = self.torgb(x)
        return x
