from typing import Dict, Tuple
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
from celluloid import Camera

# Intraproject imports
from ddpm import *
from unet import *
from dataset import *

def train_mnist():
    # hardcoding these here
    n_epoch = 1
    batch_size = 32
    n_T = 400 # 500
    device = "cuda:0"
    n_classes = 10
    n_feat = 128 # 128 ok, 256 better (but slower)
    lrate = 1e-4
    save_model = False
    save_dir = './data/diffusion_outputs10/'
    ws_test = [0.0, 0.5, 2.0] # strength of generative guidance

    # Make DDPM model
    ddpm = DDPM(
        nn_model=ContextUnet(
            in_channels=1,
            n_feat=n_feat,
            n_classes=n_classes
        ), 
        betas=(1e-4, 0.02),
        n_T=n_T,
        device=device,
        drop_prob=0.1
    )
    ddpm.to(device)

    # Load data
    dataset=LongTailedMNIST()
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=5
    )

    # Instantiate optimizer
    optim = torch.optim.Adam(
        ddpm.parameters(),
        lr=lrate
    )


    
    # Training loop
    for ep in range(n_epoch):
        print(f'epoch {ep}')
        ddpm.train(),dataloader.dataset.train()

        # linear lrate decay
        optim.param_groups[0]['lr'] = lrate*(1-ep/n_epoch)

        pbar = tqdm(dataloader)
        loss_ema = None
        for x, c in pbar:
            optim.zero_grad()
            x = x.to(device)
            c = c.to(device)
            loss = ddpm(x, c)
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f}")
            optim.step()

    # TEST
    with torch.no_grad():
        ddpm.to(torch.device('cpu'))
        ddpm.eval()

        fig=plt.figure()
        camera=Camera(fig)
        for k_sw in range(0,ddpm.n_T,20):
            print(k_sw)
            x_gen, _ = ddpm.convert(
                dataset[0][0],
                8,
                k_sw,
                1,
                (1, 28, 28),
                torch.device('cpu'),
                guide_w=2.0
            )
            plt.imshow(x_gen.squeeze())
            camera.snap()
        animation=camera.animate()
        animation.save('test.gif',writer='PillowWriter',fps=5)
        print('something')
        
        # for eval, save an image of currently generated samples (top rows)
        # followed by real images (bottom rows)
        # ddpm.eval()
        # with torch.no_grad():
        #     n_sample = 4*n_classes
        #     for w_i, w in enumerate(ws_test):
        #         x_gen, x_gen_store = ddpm.sample(n_sample, (1, 28, 28), device, guide_w=w)

        #         # append some real images at bottom, order by class also
        #         x_real = torch.Tensor(x_gen.shape).to(device)
        #         for k in range(n_classes):
        #             for j in range(int(n_sample/n_classes)):
        #                 try: 
        #                     idx = torch.squeeze((c == k).nonzero())[j]
        #                 except:
        #                     idx = 0
        #                 x_real[k+(j*n_classes)] = x[idx]

        #         x_all = torch.cat([x_gen, x_real])
        #         grid = make_grid(x_all*-1 + 1, nrow=10)
        #         save_image(grid, save_dir + f"image_ep{ep}_w{w}.png")
        #         print('saved image at ' + save_dir + f"image_ep{ep}_w{w}.png")

        #         if ep%5==0 or ep == int(n_epoch-1):
        #             # create gif of images evolving over time, based on x_gen_store
        #             fig, axs = plt.subplots(nrows=int(n_sample/n_classes), ncols=n_classes,sharex=True,sharey=True,figsize=(8,3))
        #             def animate_diff(i, x_gen_store):
        #                 print(f'gif animating frame {i} of {x_gen_store.shape[0]}', end='\r')
        #                 plots = []
        #                 for row in range(int(n_sample/n_classes)):
        #                     for col in range(n_classes):
        #                         axs[row, col].clear()
        #                         axs[row, col].set_xticks([])
        #                         axs[row, col].set_yticks([])
        #                         # plots.append(axs[row, col].imshow(x_gen_store[i,(row*n_classes)+col,0],cmap='gray'))
        #                         plots.append(axs[row, col].imshow(-x_gen_store[i,(row*n_classes)+col,0],cmap='gray',vmin=(-x_gen_store[i]).min(), vmax=(-x_gen_store[i]).max()))
        #                 return plots
        #             ani = FuncAnimation(fig, animate_diff, fargs=[x_gen_store],  interval=200, blit=False, repeat=True, frames=x_gen_store.shape[0])    
        #             ani.save(save_dir + f"gif_ep{ep}_w{w}.gif", dpi=100, writer=PillowWriter(fps=5))
        #             print('saved image at ' + save_dir + f"gif_ep{ep}_w{w}.gif")
        # # optionally save model
        # if save_model and ep == int(n_epoch-1):
        #     torch.save(ddpm.state_dict(), save_dir + f"model_{ep}.pth")
        #     print('saved model at ' + save_dir + f"model_{ep}.pth")

    print('something')

if __name__ == "__main__":
    train_mnist()

