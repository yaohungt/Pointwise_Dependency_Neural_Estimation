import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

##############################################################
#
# Utility Functions for Gaussian Experiments
#
# #############################################################

def sample_correlated_gaussian(rho=0.5, dim=20, batch_size=128, cubic=None):
    """Generate samples from a correlated Gaussian distribution."""
    x, eps = torch.chunk(torch.randn(batch_size, 2 * dim), 2, dim=1)
    y = rho * x + torch.sqrt(torch.tensor(1. - rho**2).float()) * eps

    if cubic is not None:
        y = y ** 3

    return x, y


def rho_to_mi(dim, rho):
    return -0.5 * np.log(1-rho**2) * dim


def mi_to_rho(dim, mi):
    return np.sqrt(1-np.exp(-2.0 / dim * mi))


def mi_schedule(n_iter):
    """Generate schedule for increasing correlation over time."""
    mis = np.round(np.linspace(0.5, 5.5-1e-9, n_iter)) * 2.0
    return mis.astype(np.float32)

##############################################################
#
# Utility Functions for Scores (batch_sizexbatch_size critic functions)
#
# #############################################################

def logmeanexp_diag(x):
    batch_size = x.size(0)

    logsumexp = torch.logsumexp(x.diag(), dim=(0,))
    num_elem = batch_size

    return logsumexp - torch.log(torch.tensor(num_elem).float()).cuda()


def logmeanexp_nodiag(x, dim=None, device='cuda'):
    batch_size = x.size(0)
    if dim is None:
        dim = (0, 1)

    logsumexp = torch.logsumexp(
        x - torch.diag(np.inf * torch.ones(batch_size).to(device)), dim=dim)

    try:
        if len(dim) == 1:
            num_elem = batch_size - 1.
        else:
            num_elem = batch_size * (batch_size - 1.)
    except:
        num_elem = batch_size - 1
    return logsumexp - torch.log(torch.tensor(num_elem)).to(device)


def renorm_q(f, alpha=1.0, clip=None):
    if clip is not None:
        f = torch.clamp(f * alpha, -clip, clip)
    z = logmeanexp_nodiag(f * alpha, dim=(0, 1))
    return z


def disc_renorm_q(f):
    batch_size = f.size(0)
    z = torch.zeros(1, requires_grad=True, device='cuda')

    opt = optim.SGD([z], lr=0.001)
    for i in range(10):
        opt.zero_grad()

        first_term = -F.softplus(z - f).diag().mean()
        st = -F.softplus(f - z)
        second_term = (st - st.diag().diag()).sum() / \
            (batch_size * (batch_size - 1.))
        total = first_term + second_term

        total.backward(retain_graph=True)
        opt.step()

        if total.item() <= -2 * np.log(2):
            break

    return z


def renorm_p(f, alpha=1.0):
    z = logmeanexp_diag(-f * alpha)
    return z

def estimate_p_norm(f, alpha=1.0):
    z = renorm_q(f, alpha)
    # f = renorm_p(f, alpha)
    # f = renorm_q(f, alpha)
    f = f - z
    f = -f

    return f.diag().exp().mean()
