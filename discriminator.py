import torch

class Discriminator(torch.nn.Module):
    def __init__(self, ndf, nc):
        super(Discriminator, self).__init__()
        self.main = torch.nn.Sequential(
            # input is (nc) x 64 x 64
            torch.nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
            torch.nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            torch.nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(ndf * 2),
            torch.nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            torch.nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(ndf * 4),
            torch.nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            torch.nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(ndf * 8),
            torch.nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            torch.nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = torch.nn.functional.dropout(x)
        x = self.main(x)
        return x.view(-1, 1).squeeze(1)

    def train(self, positive, negative):
        self.zero_grad()
        prediction_positive = self(positive)
        ground_truth_positive = torch.ones(prediction_positive.shape).cuda()
        loss_positive = torch.nn.functional.binary_cross_entropy(prediction_positive, ground_truth_positive)

        prediction_negative = self(negative)
        ground_truth_negative = torch.zeros(prediction_negative.shape).cuda()
        loss_negative = torch.nn.functional.binary_cross_entropy(prediction_negative, ground_truth_negative)
        
        loss = loss_positive + loss_negative;
        loss.backward()

    def teach(self, generated_samples):
        prediction_generated = self(generated_samples)
        target_generated = torch.ones(prediction_generated.shape).cuda()
        return torch.nn.functional.binary_cross_entropy(prediction_generated, target_generated)
