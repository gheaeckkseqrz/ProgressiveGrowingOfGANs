from __future__ import print_function
import argparse
import random
import torch
import torchvision
import torch.utils.data
import torchvision.transforms as transforms

import generator
import discriminator

import lossManager
import visdom

viz = visdom.Visdom()

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar10 | lsun | mnist |imagenet | folder | lfw | fake')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.000001')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")

opt = parser.parse_args()
print(opt)

# folder dataset
dataset = torchvision.datasets.ImageFolder(root=opt.dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(opt.imageSize),
                                   transforms.CenterCrop(opt.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
nc=16
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=True, num_workers=8)

netG = generator.Generator(nc).cuda()
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

netD = discriminator.Discriminator(nc, nc).cuda()
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

criterion = torch.nn.BCELoss()

fixed_noise = torch.randn(opt.batchSize, nc, 2, 2).cuda()
# setup optimizer
optimizerD = torch.optim.Adam(netD.parameters(), lr=opt.lr, betas=(.5, 0.999))
optimizerG = torch.optim.Adam(netG.parameters(), lr=opt.lr, betas=(.5, 0.999))

l2_loss = lossManager.LossManager(win="L2")
adv_loss = lossManager.LossManager(win="Adversarial")

epoch = 0
while True:
    for i, data in enumerate(dataloader, 0):
        data = data[0].cuda()
        generated_samples = netG(data)
        loss = torch.nn.functional.mse_loss(generated_samples, data)
        l2_loss.registerLoss(loss.item())
        # loss /= loss.item()
        netD.train(data, generated_samples.data)
        optimizerD.step()

        netG.zero_grad()
        errG = netD.teach(generated_samples)
        adv_loss.registerLoss(errG.item())
        errG /= errG.item()
        # errG.backward()

        loss += (errG / 100)
        loss.backward()
        optimizerG.step()

        if i % 100 == 0:
            viz.images(generated_samples.clamp(0, 1), win="FAKE")
            torchvision.utils.save_image(data,
                                         './real_samples.png',
                                         normalize=True)
            fake = netG.decoder(fixed_noise)
            torchvision.utils.save_image(generated_samples.detach(),
                                         './fake_samples_epoch_%03d.png' % (epoch),
                                         normalize=True)

    # do checkpointing
    torch.save(netG.state_dict(), './netG_epoch_%d.pth' % (epoch))
    torch.save(netD.state_dict(), './netD_epoch_%d.pth' % (epoch))
    epoch += 1
