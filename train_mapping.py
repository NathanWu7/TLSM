import torch
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
from model import VAE,Discriminator,FeatureMapping

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', default='data/tactile_pair/', type=str, help='path to the data')
parser.add_argument('--data-source', default='tactile2', type=str, help='source')
parser.add_argument('--data-goal', default='tactile1', type=str, help='destination ')
parser.add_argument('--batch-size', default=64, type=int)
parser.add_argument('--num-epochs', default=5000, type=int)
parser.add_argument('--num-workers', default=4, type=int)
parser.add_argument('--learning-rate', default=0.0001, type=float)
parser.add_argument('--content-latent-size', default=32, type=int)

args = parser.parse_args()

Model_R = VAE
Model_S = VAE
Model_D = Discriminator
Model_M = FeatureMapping
direction = args.data_source[-1]+args.data_goal[-1]

def producer(x,y, model_R ,model_M,model_S):
    mu, logsigma = model_R.encoder(x)
    contentcode_x = reparameterize(mu, logsigma)

    mu_y, logsigma_y = model_S.encoder(y)
    contentcode_y_gt = reparameterize(mu_y, logsigma_y) #mu

    contentcode_y = model_M( contentcode_x)

    return contentcode_y_gt, contentcode_y

def match_loss(contentcode_y,model_D,device):
    fake_out = model_D(contentcode_y).squeeze()
    loss = torch.nn.BCELoss()(fake_out, torch.ones_like(fake_out, device=device))
    return loss

def content_loss(contentcode_y_gt, contentcode_y): 
    loss = F.l1_loss(contentcode_y_gt,contentcode_y)
    return loss 

def Discriminator_loss(contentcode_y_gt, contentcode_y, model_D, device):
    fake_out = model_D(contentcode_y).squeeze()
    real_out = model_D(contentcode_y_gt).squeeze()
    real_loss = torch.nn.BCELoss()(real_out, torch.ones_like(real_out, device=device))
    fake_loss = torch.nn.BCELoss()(fake_out, torch.zeros_like(real_out, device=device))
    return real_loss + fake_loss
    
def main():
    # create direc
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")

    if not os.path.exists('checkimages'):
        os.makedirs("checkimages")

    # create dataset and loader
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = TactilePairDataset(args.data_dir, args.data_source, args.data_goal, transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last = True)
    
    
    # load and create model

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_R = Model_R()
    model_S = Model_S()
    
    model_R.load_state_dict(torch.load("checkpoints/0_model.pt"))
    model_S.load_state_dict(torch.load("checkpoints/1_model.pt"))

    model_R = model_R.to(device)
    model_S = model_S.to(device)

    model_D = Model_D()
    model_M = Model_M()
    
    model_D = model_D.to(device)
    model_M = model_M.to(device)
    optimizer_D = optim.Adam(model_D.parameters(), lr=args.learning_rate * 0.01)
    optimizer_M = optim.Adam(model_M.parameters(), lr=args.learning_rate)
    
    
    # do the training
    writer = SummaryWriter()
    batch_count = 0
    for i_epoch in range(args.num_epochs):
        for i_batch, imgs in enumerate(loader):
                batch_count += 1
                imgs = imgs.permute(1,0,2,3,4).to(device, non_blocking=True).type(torch.cuda.FloatTensor)  

                contentcode_y_gt, contentcode_y = producer(imgs[0],imgs[1], model_R ,model_M,model_S)

                optimizer_M.zero_grad()
                #mloss = match_loss(contentcode_y,model_D,device)
                closs = content_loss(contentcode_y_gt, contentcode_y) 
                tfloss = closs #+mloss
                tfloss.backward(retain_graph=True)
                optimizer_M.step()               
                
                contentcode_y_gt, contentcode_y = contentcode_y_gt.detach(), contentcode_y.detach()
                dloss = Discriminator_loss(contentcode_y_gt, contentcode_y, model_D, device)
                if dloss > 0.35:
                    optimizer_D.zero_grad()
                    dloss.backward()
                    optimizer_D.step()
                else:
                    pass
                

                # write log
                #writer.add_scalar('mloss', mloss.item(), batch_count)
                writer.add_scalar('closs', closs.item(), batch_count)
                writer.add_scalar('dloss', dloss.item(), batch_count)

        #evaluate and save
        if i_epoch % 200 == 0:
            #torch.save(model_D.state_dict(), "./checkpoints/"+direction+"discriminator_model.pt")
            test_img = imgs[0][0:8]
            mu,_ = model_R.encoder(test_img)
            gt_img = imgs[1][0:8]
            mu_sim = model_M(mu)
            recon_imgs = model_S.decoder(mu_sim)
            saved_imgs = torch.cat([test_img, recon_imgs, gt_img], dim=0)

            all_test_img = imgs[0]
            all_gt_img = imgs[1]
            all_mu,_ = model_R.encoder(all_test_img)
            all_mu_sim = model_M(all_mu)
            all_recon_imgs = model_S.decoder(all_mu_sim)

            torch.save(model_M.state_dict(), "./checkpoints/"+direction+"mapping_model.pt")
            MAE_score = F.mse_loss(all_gt_img, all_recon_imgs, reduction='mean')

            save_image(saved_imgs, "./checkimages/final/"+direction+"_%d.png" % (i_epoch), nrow=8)
            print("epoch: ", i_epoch, " MAE: ",'%.6f' % MAE_score.item())


        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last = True)
    writer.close()

if __name__ == '__main__':
    main()
