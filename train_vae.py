import torch
import random
from torch import optim
from torch.nn import functional as F
import torch.nn 

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image


import numpy as np
import argparse
import os
from utils import *
from model import VAE

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', default='data/tactile', type=str, help='path to the data')
parser.add_argument('--data-tag', default='tactile1', type=str, help='choose domain')
parser.add_argument('--batch-size', default=256, type=int)
parser.add_argument('--num-epochs', default=1000, type=int)
parser.add_argument('--num-workers', default=4, type=int)
parser.add_argument('--learning-rate', default=0.001, type=float)
parser.add_argument('--beta', default=20, type=int)
parser.add_argument('--content-latent-size', default=32, type=int)

args = parser.parse_args()

Model_G = VAE

def set_seed(seed=0):
    # Python random module
    random.seed(seed)
    
    # Numpy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    
    # If you are using CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False



def vae_loss(x, mu, logsigma, recon_x, beta=1):
    recon_loss = F.mse_loss(x, recon_x, reduction='mean')
    kl_loss = -0.5 * torch.sum(1 + logsigma - mu.pow(2) - logsigma.exp())
    kl_loss = kl_loss / torch.numel(x)
    return recon_loss + kl_loss * beta


def forward_loss(x, model, beta):

    mu, logsigma = model.encoder(x)
    contentcode = reparameterize(mu, logsigma)

    recon_x = model.decoder(contentcode)

    return vae_loss(x, mu, logsigma, recon_x, beta)
    
    
def main():
    set_seed(0)
    # create direc
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")

    if not os.path.exists('checkimages'):
        os.makedirs("checkimages")

    if not os.path.exists('checkimages/'+args.data_tag):
        os.makedirs("checkimages/"+args.data_tag)

    # create dataset and loader
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = TactileDataset(args.data_dir,args.data_tag, transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last = True)

    # create model
    model_G = Model_G(content_latent_size = args.content_latent_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_G = model_G.to(device)
    optimizer_G = optim.Adam(model_G.parameters(), lr=args.learning_rate)
    writer = SummaryWriter()
    batch_count = 0

    # do the training
    for i_epoch in range(args.num_epochs):
        for _, imgs in enumerate(loader):
                batch_count += 1
                imgs = imgs.unsqueeze(1).to(device, non_blocking=True).type(torch.cuda.FloatTensor)  

                optimizer_G.zero_grad()
                floss = 0
                floss = forward_loss(imgs, model_G, args.beta)
                floss.backward()
                optimizer_G.step()

                # write log
                writer.add_scalar('floss', floss.item(), batch_count)
                
        #evaluate and save
        if i_epoch % 100 == 0 :
            torch.save(model_G.state_dict(), "./checkpoints/%d_model.pt" % (int(args.data_tag[-1])))
            test_img = imgs[0:8]
            mu,_ = model_G.encoder(test_img)
            recon_imgs = model_G.decoder(mu)

            all_test_img = imgs
            all_mu,_ = model_G.encoder(all_test_img)
            all_recon_imgs = model_G.decoder(all_mu)

            MAE_score = F.mse_loss(all_test_img, all_recon_imgs, reduction='mean')

            saved_imgs = torch.cat([test_img, recon_imgs], dim=0)
            save_image(saved_imgs, "./checkimages/"+args.data_tag+"/%d.png" % (i_epoch), nrow=8)
            print(" epoch: ", i_epoch," MAE: ",'%.6f' % MAE_score.item())
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last = True)   
    writer.close()

if __name__ == '__main__':
    main()
