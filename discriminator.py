import torch
import encoder

class Discriminator(torch.nn.Module):
    def __init__(self, ndf, nc):
        super(Discriminator, self).__init__()
        self.b1 = encoder.EncoderBlock(nc)
        self.b2 = encoder.EncoderBlock(nc)
        self.b3 = encoder.EncoderBlock(nc)
        self.b4 = encoder.EncoderBlock(nc)
        self.b5 = encoder.EncoderBlock(nc)
        self.b6 = encoder.EncoderBlock(nc)
        self.fromrgb = torch.nn.Conv2d(3, nc, 1, 1, 0)
        self.fc = torch.nn.Linear(nc, 1)
        self.nc = nc

    def forward(self, x):
        x = torch.nn.functional.dropout(x)
        x = self.fromrgb(x)
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        x = self.b6(x)
        x = x.view(-1, self.nc)
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x

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
