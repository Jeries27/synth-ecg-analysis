from utils import *
import torch

def training_loss_label(net, loss_fn, inputs, diffusion_hyperparams):

    """
    Compute the training loss of epsilon and epsilon_theta

    Parameters
    ----------
    net (torch network):            the wavenet model
    loss_fn (torch loss function):  the loss function, default is nn.MSELoss()
    inputs (torch.tensor):               training data, shape=(batchsize, 1, length of audio)
    diffusion_hyperparams (dict):   dictionary of diffusion hyperparameters returned by calc_diffusion_hyperparams
                                    note, the tensors need to be cuda tensors

    Returns
    -------
    training loss

    References
    ----------
    [1] Diffusion-based Conditional ECG Generation with Structured State Space Models
    """
    _dh = diffusion_hyperparams
    T, Alpha_bar = _dh["T"], _dh["Alpha_bar"]
    target, label = inputs

    B, C, L = target.shape  # B is batchsize, C=leads, L is audio length

    diffusion_steps = torch.randint(T, size=(B,1,1)).cuda()  # randomly sample diffusion steps from 1~T
    z = std_normal(target.shape)
    transformed_Y = torch.sqrt(Alpha_bar[diffusion_steps]) * target + torch.sqrt(1-Alpha_bar[diffusion_steps]) * z
    epsilon_theta = net((transformed_Y, label, diffusion_steps.view(B,1),))
    return loss_fn(epsilon_theta, z)

def training_loss_conditional(net, loss_fn, inputs, diffusion_hyperparams):

    """
    Compute the conditional training loss of epsilon and epsilon_theta

    Parameters
    ----------
    net (torch network):            the wavenet model
    loss_fn (torch loss function):  the loss function, default is nn.MSELoss()
    inputs (torch.tensor):               training data, shape=(batchsize, 1, length of audio)
    diffusion_hyperparams (dict):   dictionary of diffusion hyperparameters returned by calc_diffusion_hyperparams
                                    note, the tensors need to be cuda tensors

    Returns
    -------
    training loss

    References
    ----------
    [1] Diffusion-based Conditional ECG Generation with Structured State Space Models
    """
    _dh = diffusion_hyperparams
    T, Alpha_bar = _dh["T"], _dh["Alpha_bar"]
    target, label = inputs

    B, C, L = target.shape  # B is batchsize, C=leads, L is audio length

    diffusion_steps = torch.randint(T, size=(B,1,1)).cuda()  # randomly sample diffusion steps from 1~T
    z = std_normal(target.shape)
    transformed_Y = torch.sqrt(Alpha_bar[diffusion_steps]) * target + torch.sqrt(1-Alpha_bar[diffusion_steps]) * z
    epsilon_theta = net((transformed_Y, label, diffusion_steps.view(B,1),))
    epsilon_theta[:,:,:500] = z[:,:,:500] ## we assume that we know the first 500 (observed)
    return loss_fn(epsilon_theta, z) + 0.5*loss_fn(torch.std(epsilon_theta[:,:,500:]), torch.std(z[:,:,500:]))
