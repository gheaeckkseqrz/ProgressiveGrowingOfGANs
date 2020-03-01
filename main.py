from __future__ import print_function
import argparse
import random
import torch
import torchvision
import torch.utils.data
import torchvision.transforms as transforms

import generator
import discriminator
import encoder

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

perceptual = encoder.Encoder(nc).cuda()
perceptual.requires_grad = False

# setup optimizer
optimizerD = torch.optim.Adam(netD.parameters(), lr=opt.lr, betas=(.5, 0.999))
optimizerG = torch.optim.Adam(netG.parameters(), lr=opt.lr, betas=(.5, 0.999))

fixed_noise = torch.randn(opt.batchSize, nc, 2, 2).cuda()

l2_loss = lossManager.LossManager(win="L2")
adv_loss = lossManager.LossManager(win="Adversarial")

epoch = 0
for step in range(6):
    image_size = 4 * pow(2, step)
    batch_size = min(int(8192 * 4 / pow(4, step)), 2048)
    print("Step", step, "Size", image_size, "Batch", batch_size)
    # folder dataset
    epoch_size = 2048 * 8 * 64
    dataset = torchvision.datasets.ImageFolder(root=opt.dataroot,
                                               transform=transforms.Compose([
                                                   transforms.Resize(image_size),
                                                   transforms.CenterCrop(image_size),
                                                   transforms.ToTensor(),
                                               ]))

    for _ in range(pow(2, step)):
        dataset_part, _ = torch.utils.data.random_split(dataset, [epoch_size, len(dataset) - epoch_size])
        dataloader = torch.utils.data.DataLoader(dataset_part, batch_size=batch_size, shuffle=True)
        for i, data in enumerate(progressbar.progressbar(dataloader), 0):
            data = data[0].cuda()
            generated_samples, code = netG(data)
            code2 = perceptual(generated_samples)

            loss = torch.nn.functional.mse_loss(code2, code.data)
            l2_loss.registerLoss(loss.item())
            prediction_positive, prediction_negative = netD.train(data, generated_samples.data)
            optimizerD.step()

            netG.zero_grad()
            errG, discriminator_prediction = netD.teach(generated_samples)
            adv_loss.registerLoss(errG.item())

            netG.zero_grad()
            errG.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(netG.parameters(), .4)
            adv_norm = 0
            optimizerG.step()

            netG.zero_grad()
            loss.backward()
            optimizerG.step()

            perceptual.load_state_dict(netG.encoder.state_dict())

            if i % 100 == 0:
                viz.bar(code[0].view(1, -1), win="CODE")
                viz.bar(discriminator_prediction.data, win="PRED", opts={'title':"Discimator Prediction for generated samples"})
                viz.bar(prediction_positive.data, win="PREDP", opts={'title':"Discimator Prediction for positive samples"})
                viz.bar(prediction_negative.data, win="PREDN", opts={'title':"Discimator Prediction for negative samples"})
                viz.images(torch.cat([generated_samples.clamp(0, 1).detach(), data.data], 3), win="FAKE")
                torchvision.utils.save_image(data.data,
                                             './real_samples.png',
                                             normalize=True)
                torchvision.utils.save_image(generated_samples.detach(),
                                             './fake_samples_epoch_%03d.png' % (epoch),
                                             normalize=True)

        # do checkpointing
        torch.save(netG.state_dict(), './netG_epoch_%d.pth' % (epoch))
        torch.save(netD.state_dict(), './netD_epoch_%d.pth' % (epoch))
        epoch += 1
    netG.next_step()
    netD.next_step()
    perceptual.next_step()
