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
from torch.optim.lr_scheduler import StepLR
from torchmetrics.image.inception import InceptionScore
# warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser(description='GanGan style encoder training loop')
parser.add_argument('--config', default='./configs/config.yaml')

in_channels = 3
output_size = 512
network_pkl = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl"
dummy_label = None
checkpoints_dir = './results/checkpoints/'
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


def save_plot(name, label, plot_list):
    plt.figure(figsize=(10, 5))
    plt.plot(plot_list, label=label)
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(name)
    plt.close()
    print(f"Plot {label} saved")

def save_all_plots(dir_prefix, all_plot_list):
    plt.figure(figsize=(10, 5))
    if args.reconstruction_loss_weight != -1:
        latent_losses, reconstruction_losses, total_losses = zip(*all_plot_list)
        save_plot(dir_prefix + "/latent_loss_curve.png", "Latent Loss", latent_losses)
        save_plot(dir_prefix + "/reconstruction_loss_curve.png", "Reconstruction Loss", reconstruction_losses)
    else:
        total_losses = all_plot_list

    save_plot(dir_prefix + "/loss_curve.png", "Loss", total_losses)


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


def train(epoch, batch_size, num_batches, model, stylegan, optimizer, criterion):
    iter_time = AverageMeter()
    latent_losses = AverageMeter()
    reconstruction_losses = AverageMeter()
    total_losses = AverageMeter()
    model.train()

    # inception_score = InceptionScore()

    for batch_idx in range(num_batches):
        start = time.time()
        z = torch.randn(batch_size, output_size).cuda()
        images = stylegan(z, dummy_label)

        pred_z = model(images)
        latent_loss = criterion(pred_z, z)
        pred_images = stylegan(pred_z, dummy_label)  # genetating images using predicted z
        # TODO: It seems that the inception score needs images to be uint8 type. But we produce float images. I don't
        #  know if this is the right input, probably we have to transpose the tensor or do other operations.
        #  Can you take a look into it?
        # inception_score.update(pred_images)

        if args.reconstruction_loss_weight != -1:
            reconstruction_loss = criterion(pred_images, images)
            loss = args.latent_loss_weight * latent_loss + args.reconstruction_loss_weight * reconstruction_loss
        else:
            loss = latent_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if args.reconstruction_loss_weight != -1:
            reconstruction_losses.update(reconstruction_loss.item(), pred_z.shape[0])
            latent_losses.update(latent_loss.item(), pred_z.shape[0])

        total_losses.update(loss.item(), pred_z.shape[0])


        iter_time.update(time.time() - start)
        if batch_idx % 10 == 0:
            if args.reconstruction_loss_weight != -1:
                print(('Train: Epoch: [{0}][{1}/{2}]\t'
                       'Time {iter_time.val:.3f} ({iter_time.avg:.3f})\t'
                       'Latent Loss {latent_loss.val:.4f} ({latent_loss.avg:.4f})\t'
                       'Reconstruction Loss {reconstruction_loss.val:.4f} ({reconstruction_loss.avg:.4f})\t'
                       'Total Loss {total_loss.val:.4f} ({total_loss.avg:.4f})\t')
                      .format(epoch, batch_idx, num_batches,
                              iter_time=iter_time, latent_loss=latent_losses,
                              reconstruction_loss=reconstruction_losses, total_loss=total_losses))
            else:
                print(('Train: Epoch: [{0}][{1}/{2}]\t'
                       'Time {iter_time.val:.3f} ({iter_time.avg:.3f})\t'
                       'Total Loss {total_loss.val:.4f} ({total_loss.avg:.4f})\t')
                      .format(epoch, batch_idx, num_batches,
                              iter_time=iter_time, total_loss=total_losses))
    
    # inception_mean, inception_std = inception_score.compute()
    # print(f'Epoch {epoch} - Inception Score: Mean={inception_mean:.4f}, Std={inception_std:.4f}')
    
    if args.reconstruction_loss_weight != -1:
        losses_list.append([float(latent_losses.avg), float(reconstruction_losses.avg), float(total_losses.avg)])
    else:
        losses_list.append([float(total_losses.avg)])


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
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)
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
    elif args.model == 'ResNet':
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, output_size)
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
    # We will add back the scheduler later, once we have better results.
    # scheduler = StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    stylegan = init_style_gan()
    ref_image = init_ref_image()
    try:
        for epoch in range(args.epochs):
            train(epoch, args.batch_size, args.train_num_batches, model, stylegan, optimizer, criterion)
            # scheduler.step()
            if epoch % args.plot_rate == 0:
                plot_sanity_check_image(epoch, ref_image, model, stylegan)
                save_all_plots("./results", losses_list)
                print("Results saved")

            if epoch % args.save_rate == 0:
                torch.save(model.state_dict(), checkpoints_dir + args.model.lower() + '_' + str(epoch) + '.pth')
                print("Model saved successfully")
    except KeyboardInterrupt:
        plot_sanity_check_image("stop", ref_image, model, stylegan)
        save_all_plots("./results.png", losses_list)
        torch.save(model.state_dict(), checkpoints_dir + args.model.lower() + '_stop.pth')
        print("Model saved successfully. Gracefully exiting...")


if __name__ == '__main__':
    main()