import torch

class DecoderBlock(torch.nn.Module):
    def __init__(self, nc, name):
        super(DecoderBlock, self).__init__()
        self.c1 = torch.nn.Conv2d(nc, nc, 3, 1, 1)
        self.c2 = torch.nn.Conv2d(nc, nc, 3, 1, 1)
        self.c3 = torch.nn.Conv2d(nc, nc, 3, 1, 1)
        self.c4 = torch.nn.Conv2d(nc, nc, 3, 1, 1)
        self.b1 = torch.nn.BatchNorm2d(nc)
        self.b2 = torch.nn.BatchNorm2d(nc)
        self.b3 = torch.nn.BatchNorm2d(nc)
        self.b4 = torch.nn.BatchNorm2d(nc)
        self.name = name

    def forward(self, x):
        # print(self.name, x.shape)
        x = torch.nn.functional.interpolate(x, scale_factor=2)
        x = self.b1(torch.nn.functional.relu(self.c1(x)))
        x = self.b2(torch.nn.functional.relu(self.c2(x)))
        x = self.b3(torch.nn.functional.relu(self.c3(x)))
        x = self.b4(torch.nn.functional.relu(self.c4(x)))
        return x

class Decoder(torch.nn.Module):
    def __init__(self, nc):
        super(Decoder, self).__init__()
        self.b1 = DecoderBlock(nc, "B1")
        self.b2 = DecoderBlock(nc, "B2")
        self.b3 = DecoderBlock(nc, "B3")
        self.b4 = DecoderBlock(nc, "B4")
        self.b5 = DecoderBlock(nc, "B5")
        self.b6 = DecoderBlock(nc, "B6")
        self.layers = [self.b1, self.b2, self.b3, self.b4, self.b5, self.b6]
        self.torgb = torch.nn.Conv2d(nc, 3, 1, 1, 0)
        self.step = 0
        self.alpha = 1

    def next_step(self):
        self.step = self.step + 1
        self.alpha = 0
    
    def forward(self, x):
        self.alpha = min(1, self.alpha + (1/2000))
        assert x.shape[2] == 2 and x.shape[3] == 2
        for i in range(self.step):
            x = self.layers[i](x)
        x1 = self.layers[self.step](x)
        if self.step > 0:
            x = torch.nn.functional.interpolate(x, scale_factor=2, mode='bilinear')
            x = x * (1-self.alpha) + x1 * self.alpha
        else:
            x = x1
        x = self.torgb(x)
        return x

if __name__ == "__main__":
    d = Decoder(4)
    t = torch.Tensor(1, 4, 2, 2)
    for i in range(6):
        print("Step", i)
        print("Input", t.shape)
        output = d(t)
        d.next_step()
        print("Output", output.shape)
