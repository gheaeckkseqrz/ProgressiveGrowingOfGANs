import torch

class EncoderBlock(torch.nn.Module):
    def __init__(self, nc, name):
        super(EncoderBlock, self).__init__()
        self.c1 = torch.nn.Conv2d(nc, nc, 3, 1, 1)
        self.c2 = torch.nn.Conv2d(nc, nc, 3, 2, 1)
        self.c3 = torch.nn.Conv2d(nc, nc, 3, 1, 1)
        self.c4 = torch.nn.Conv2d(nc, nc, 3, 1, 1)
        self.b1 = torch.nn.BatchNorm2d(nc)
        self.b2 = torch.nn.BatchNorm2d(nc)
        self.b3 = torch.nn.BatchNorm2d(nc)
        self.b4 = torch.nn.BatchNorm2d(nc)
        self.name = name

    def forward(self, x):
        # print(self.name, x.shape)
        x = self.b1(torch.nn.functional.relu(self.c1(x)))
        x = self.b2(torch.nn.functional.relu(self.c2(x)))
        x = self.b3(torch.nn.functional.relu(self.c3(x)))
        x = self.b4(torch.nn.functional.relu(self.c4(x)))
        return x

class Encoder(torch.nn.Module):
    def __init__(self, nc):
        super(Encoder, self).__init__()
        self.b1 = EncoderBlock(nc, "B1")
        self.b2 = EncoderBlock(nc, "B2")
        self.b3 = EncoderBlock(nc, "B3")
        self.b4 = EncoderBlock(nc, "B4")
        self.b5 = EncoderBlock(nc, "B5")
        self.b6 = EncoderBlock(nc, "B6")
        self.fromrgb = torch.nn.Conv2d(3, nc, 1, 1, 0)
        self.layers = [self.b1, self.b2, self.b3, self.b4, self.b5, self.b6]
        self.step = 0
        self.alpha = 1

    def next_step(self):
        self.step = self.step + 1
        self.alpha = 0

    def forward(self, x):
        self.alpha = min(1, self.alpha + (1/1000))
        x1 = self.fromrgb(x)
        in_training = self.layers[- 1 - self.step]
        x = in_training(x1)
        if self.step > 0:
            x2 = torch.nn.functional.interpolate(x1, scale_factor=.5)
            x = x * self.alpha + x2 * (1 - self.alpha)
            for b in self.layers[-self.step:]:
                x = b(x)
        assert x.shape[2] == 2 and x.shape[3] == 2
        return x

if __name__ == "__main__":
    e = Encoder(4)
    t = torch.Tensor(1, 3, 4, 4)
    for i in range(6):
        print("Step", i)
        print("Input", t.shape)
        output = e(t)
        t = torch.nn.functional.interpolate(t, scale_factor=2)
        e.next_step()
        print("Output", output.shape)
