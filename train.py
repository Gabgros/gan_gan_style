import yaml
import argparse
import time
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from models.encoder import Encoder
import legacy
import dnnlib
from torch.backends import cudnn
cudnn.benchmark = True
import warnings
# warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser(description='GanGan style encoder training loop')
parser.add_argument('--config', default='./configs/config.yaml')

in_channels = 3
output_size = 512
network_pkl = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl"
dummy_label = None
losses_list = []

class DummyStyleGan(nn.Module):
    def forward(self, x):
        return torch.zeros((x.shape[0], 3, 1024, 1024))

class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.Dropout(0.1), nn.BatchNorm2d(32), nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
    def forward(self, x):
        return x


def init_ref_image():
    img = Image.open('./results/sanity_check_images/reference_image.png')
    transform = Compose([
        Resize((1024, 1024)),
        ToTensor(),
    ])
    ret_val = transform(img).unsqueeze(0)
    if torch.cuda.is_available():
        ret_val = ret_val.cuda()
    return ret_val

def init_style_gan():
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        stylegan = legacy.load_network_pkl(f)['G_ema'].to(device)
    dummy_label = torch.zeros([1, stylegan.c_dim], device=device)
    return stylegan


def save_plot(name, plot_list):
    plt.plot(list(range(len(plot_list))), plot_list)
    plt.savefig(name)
    print("Plot saved")
    plt.clf()


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
'''
def train(epoch, batch_size, num_batches, model, stylegan, optimizer, criterion):

    iter_time = AverageMeter()
    losses = AverageMeter()
    model.train()

    for batch_idx in range(num_batches):
        start = time.time()
        z = torch.randn(batch_size, output_size).cuda()
        images = stylegan(z, dummy_label)

        pred_z = model(images)
        loss = criterion(pred_z, z)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss, pred_z.shape[0])
        iter_time.update(time.time() - start)
        if batch_idx % 10 == 0:
            print(('Train: Epoch: [{0}][{1}/{2}]\t'
                   'Time {iter_time.val:.3f} ({iter_time.avg:.3f})\t'
                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t')
                   .format(epoch, batch_idx, num_batches,
                           iter_time=iter_time, loss=losses))
    losses_list.append([losses.avg.item()])
'''

def train(epoch, batch_size, num_batches, model, stylegan, optimizer, criterion):

    iter_time = AverageMeter()
    losses = AverageMeter()
    model.train()

    for batch_idx in range(num_batches):
        start = time.time()
        z = torch.randn(batch_size, output_size).cuda()
        images = stylegan(z, dummy_label)

        pred_z = model(images)
        loss = criterion(pred_z, z)

        pred_images = stylegan(pred_z, dummy_label) #genetating images using predicted z 
        mse_loss = F.mse_loss(pred_images, images)

        total_loss = loss + mse_loss

        optimizer.zero_grad()
        loss.backward()

        #gradient calculation with respect to StyleGAN params
        stylegan.zero_grad()
        for param in stylegan.parameters():
            param.grad = None  
        fake_images = stylegan(z, dummy_label)
        fake_images.retain_grad()  
        loss.backward(retain_graph=True)
        stylegan_gradient = fake_images.grad 
        
        optimizer.step()

        losses.update(loss, pred_z.shape[0])
        iter_time.update(time.time() - start)
        if batch_idx % 10 == 0:
            print(('Train: Epoch: [{0}][{1}/{2}]\t'
                   'Time {iter_time.val:.3f} ({iter_time.avg:.3f})\t'
                   'Loss {total_loss.val:.4f} ({total_loss.avg:.4f})\t')
                   .format(epoch, batch_idx, num_batches,
                           iter_time=iter_time, loss=losses))
    losses_list.append([losses.avg.item()])


def validate(epoch, batch_size, num_batches, model, stylegan, criterion):
    iter_time = AverageMeter()
    losses = AverageMeter()
    model.eval()
    # evaluation loop
    for batch_idx in range(num_batches):
        start = time.time()
        z = torch.randn(batch_size, output_size).cuda()
        images = stylegan(z, dummy_label)
        with torch.inference_mode():
            pred_z = model(images)
            loss = criterion(pred_z, z)

        losses.update(loss, pred_z.shape[0])
        iter_time.update(time.time() - start)
        if batch_idx % 10 == 0:
            print(('Validation: Epoch: [{0}][{1}/{2}]\t'
                   'Time {iter_time.val:.3f} ({iter_time.avg:.3f})\t'
                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t')
                  .format(epoch, batch_idx, batch_size,
                          iter_time=iter_time, loss=losses))

    losses_list[-1].append(losses.avg.item())

def plot_sanity_check_image(epoch, ref_image, model, stylegan):
    model.eval()
    pred_z = model(ref_image)
    img = stylegan(pred_z, dummy_label)
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    if not os.path.exists('./results/sanity_check_images'):
        os.makedirs('./results/sanity_check_images')
    Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'./results/sanity_check_images/epoch_{epoch}.png')


def main():
    if not os.path.exists('./results'):
        os.makedirs('./results')
    global args
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)

    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)


    if args.model == 'DummyModel':
        model = DummyModel()
    elif args.model == 'Encoder':
        model = Encoder(in_channels, output_size)
    else:
        print("You must select a model! Dying...")
        exit()
    print(model)

    if torch.cuda.is_available():
        model = model.cuda()

    if args.loss_type == "MSE":
        criterion = nn.MSELoss()
    else:
        print("You must select a criterion! Dying...")
        exit()

    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    # TODO: Add a scheduler, done :) 
    scheduler = StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    stylegan = init_style_gan()
    ref_image = init_ref_image()
    try:
        for epoch in range(args.epochs):
            train(epoch, args.batch_size, args.train_num_batches, model, stylegan, optimizer, criterion)
            validate(epoch, args.batch_size, args.val_num_batches, model, stylegan, criterion)
            plot_sanity_check_image(epoch, ref_image, model, stylegan)
            save_plot("./results/training_curve.png", losses_list)

            scheduler.step()

            if epoch % args.save_rate == 0:
                if not os.path.exists('./results/checkpoints'):
                    os.makedirs('./results/checkpoints')
                torch.save(model.state_dict(), './results/checkpoints/' +
                           args.model.lower() + '_' + str(epoch) + '.pth')
                print("Model saved successfully")
    except KeyboardInterrupt:
        if not os.path.exists('./checkpoints'):
            os.makedirs('./checkpoints')
        torch.save(model.state_dict(), './checkpoints/' +
                   args.model.lower() + '_stop.pth')
        print("Model saved successfully")


if __name__ == '__main__':
    main()