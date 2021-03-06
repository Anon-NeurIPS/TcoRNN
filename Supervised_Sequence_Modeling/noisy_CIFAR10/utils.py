import torch
import torchvision
import torchvision.transforms as transforms
import wandb

def get_data(bs_train,bs_test):
    train_dataset = torchvision.datasets.CIFAR10(root='data/',
                                                 train=True,
                                                 transform=transforms.ToTensor(),
                                                 download=True)

    test_dataset = torchvision.datasets.CIFAR10(root='data/',
                                                train=False,
                                                transform=transforms.ToTensor())

    train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [47000,3000])

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=bs_train,
                                               shuffle=True)

    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                              batch_size=bs_test,
                                              shuffle=False)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=bs_test,
                                              shuffle=False)

    return train_loader, valid_loader, test_loader


def normalize_int(x):
  x -= x.min()
  x *= 255 / x.max()
  return x.int()

def Plot_Vid(seq, fps=60, vformat='gif', name='Latents'):
    n_t, n_cin, nh, nw = seq.shape
    # Seq shape should be T,C,H,W

    seq_norm = normalize_int(seq).cpu()
    
    wandb_video = wandb.Video(seq_norm, fps=fps, format=vformat)
    wandb.log({name: wandb_video})
