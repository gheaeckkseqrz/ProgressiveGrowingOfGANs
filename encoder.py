import torch

class EncoderBlock(torch.nn.Module):
    def __init__(self, nc):
        super(EncoderBlock, self).__init__()
        self.c1 = torch.nn.Conv2d(nc, nc, 3, 1, 1)
        self.c2 = torch.nn.Conv2d(nc, nc, 3, 2, 1)
        self.c3 = torch.nn.Conv2d(nc, nc, 3, 1, 1)
        self.c4 = torch.nn.Conv2d(nc, nc, 3, 1, 1)
        self.b1 = torch.nn.BatchNorm2d(nc)
        self.b2 = torch.nn.BatchNorm2d(nc)
        self.b3 = torch.nn.BatchNorm2d(nc)
        self.b4 = torch.nn.BatchNorm2d(nc)

    def forward(self, x):
        x = self.b1(torch.nn.functional.relu(self.c1(x)))
        x = self.b2(torch.nn.functional.relu(self.c2(x)))
        x = self.b3(torch.nn.functional.relu(self.c3(x)))
        x = self.b4(torch.nn.functional.relu(self.c4(x)))
        return x

class Encoder(torch.nn.Module):
    def __init__(self, nc):
        super(Encoder, self).__init__()
        self.b1 = EncoderBlock(nc)
        self.b2 = EncoderBlock(nc)
        self.b3 = EncoderBlock(nc)
        self.b4 = EncoderBlock(nc)
        self.b5 = EncoderBlock(nc)
        self.b6 = EncoderBlock(nc)
        self.torgb = torch.nn.Conv2d(3, nc, 1, 1, 0)

    def forward(self, x):
        x = self.torgb(x)
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        return x

