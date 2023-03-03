from torchvision import transforms as TT
import torch.nn.functional as F
import torch

import numpy as np

import matplotlib.pyplot as plt

# def calculate_2dft(input):
#     ft = torch.fft.ifftshift(input)
#     ft = torch.fft.fft2(ft)
#     return torch.fft.fftshift(ft)

# def calculate_2dift(input):
#     ift = torch.fft.ifftshift(input)
#     ift = torch.fft.ifft2(ift)
#     ift = torch.fft.fftshift(ift)
#     return ift.real

def get_amp_phase(input):
    # ft = torch.fft.ifftshift(input)
    ft = torch.fft.fft2(input, dim=(-2, -1))
    # ft = torch.fft.fftshift(ft)
    return torch.fft.fftshift(ft.abs()), ft.angle() 

def calculate_2dift(input):
    # ift = torch.fft.ifftshift(input)
    ift = torch.fft.ifft2(input, dim=(-2, -1))
    # ift = torch.fft.fftshift(ift)
    return ift.real

def low_freq_mutate_np( amp_src, amp_trg, L=0.1 ):

    _, h, w = amp_src.shape
    b = (  np.floor(np.amin((h,w))*L)  ).astype(int)
    c_h = np.floor(h/2.0).astype(int)
    c_w = np.floor(w/2.0).astype(int)

    h1 = c_h-b
    h2 = c_h+b+1
    w1 = c_w-b
    w2 = c_w+b+1

    amp_src[:,h1:h2,w1:w2] = amp_trg[:,h1:h2,w1:w2]
    amp_src = torch.fft.ifftshift( amp_src )
    return amp_src

def beta_transform(img_src, img_tgt, beta, show=False):

    src_amp, src_phase = get_amp_phase(img_src)
    tgt_amp, _ = get_amp_phase(img_tgt)

    # dims = int(beta * img_src.shape[-2]), int(beta * img_src.shape[-1])
    # mask = torch.ones((dims[0], dims[1]), dtype=bool)

    # h_pad = (img_src.shape[-2] - mask.shape[0]) / 2
    # w_pad = (img_src.shape[-1] - mask.shape[1]) / 2

    # t_pad, b_pad = int(np.floor(h_pad)), int(np.ceil(h_pad))
    # l_pad, r_pad = int(np.floor(w_pad)), int(np.ceil(w_pad))

    # mask = F.pad(mask, (l_pad, r_pad, t_pad, b_pad)).unsqueeze(0)

    # src_amp[mask] = tgt_amp[mask]

    src_amp = low_freq_mutate_np(src_amp, tgt_amp, beta)
    
    # src_amp = torch.fft.ifftshift(src_amp)

    res = calculate_2dift((src_amp * torch.exp(1j * src_phase)))

    if show:
        _, axes = plt.subplots(ncols=3, figsize=(10, 5))
        plt.set_cmap('gray')

        axes[0].imshow(TT.ToPILImage()(img_src))
        axes[1].imshow(TT.ToPILImage()(img_tgt))
        axes[2].imshow(res.numpy()[0][:, :, None])

        axes[0].axis('off')
        axes[1].axis('off')
        axes[2].axis('off')

    return res