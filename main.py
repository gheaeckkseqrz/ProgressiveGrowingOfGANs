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
import progressbar

viz = visdom.Visdom()

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar10 | lsun | mnist |imagenet | folder | lfw | fake')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.000001')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
opt = parser.parse_args()
print(opt)

nc=64
netG = generator.Generator(nc).cuda()
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

netD = discriminator.Discriminator(nc).cuda()
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

# setup optimizer
optimizerD = torch.optim.Adam(netD.parameters(), lr=opt.lr, betas=(.5, 0.999))
optimizerG = torch.optim.Adam(netG.parameters(), lr=opt.lr, betas=(.5, 0.999))

fixed_noise = torch.randn(opt.batchSize, nc, 2, 2).cuda()

l2_loss = lossManager.LossManager(win="L2")
adv_loss = lossManager.LossManager(win="Adversarial")

epoch = 0
for step in range(6):
    image_size = 4 * pow(2, step)
    print("Step", step, "Size", image_size)
    # folder dataset
    epoch_size = 2048 * 64
    dataset = torchvision.datasets.ImageFolder(root=opt.dataroot,
                                               transform=transforms.Compose([
                                                   transforms.Resize(image_size),
                                                   transforms.CenterCrop(image_size),
                                                   transforms.ToTensor(),
                                               ]))

    for _ in range(pow(2, step)):
        dataset_part, _ = torch.utils.data.random_split(dataset, [epoch_size, len(dataset) - epoch_size])
        dataloader = torch.utils.data.DataLoader(dataset_part, batch_size=opt.batchSize, shuffle=True, num_workers=8)
        for i, data in enumerate(progressbar.progressbar(dataloader), 0):
            data = data[0].cuda()
            generated_samples = netG(data)
            loss = torch.nn.functional.mse_loss(generated_samples, data)
            l2_loss.registerLoss(loss.item())
            netD.train(data, generated_samples.data)
            optimizerD.step()

            netG.zero_grad()
            errG = netD.teach(generated_samples)
            adv_loss.registerLoss(errG.item())

            netG.zero_grad()
            errG.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(netG.parameters(), .4)
            adv_norm = 0
            optimizerG.step()

            netG.zero_grad()
            loss.backward()
            l2_norm = 0
            # for g in netG.parameters():
            #     l2_norm += torch.norm(g.grad).item() if g.grad is not None else 0
            # print("l2_norm", l2_norm)
            optimizerG.step()

            if i % 100 == 0:
                viz.images(torch.cat([generated_samples.clamp(0, 1), data], 3), win="FAKE")
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
    netG.next_step()
    netD.next_step()
