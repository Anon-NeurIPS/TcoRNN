from torch import nn, optim
import torch
import utils
import uncoupled_model as model_unncoupled
import multiconv1d_model as model_tcornn1d
import conv2d_model as model_tcornn2d
import network as model_cornn
import argparse
import torch.nn.utils
from pathlib import Path
from tqdm import tqdm
import os
from sys import exit

parser = argparse.ArgumentParser(description='training parameters')

parser.add_argument('--run_name', type=str, default=None,
                    help='name of run for wandb')
parser.add_argument('--model_type', type=str, default='cornn',
                    help='type of model, cornn, tcornn1d, tcornn2d')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='')
parser.add_argument('--n_hid', type=int, default=128,
                    help='hidden size of recurrent net')
parser.add_argument('--epochs', type=int, default=120,
                    help='max epochs')
parser.add_argument('--batch', type=int, default=100,
                    help='batch size')
parser.add_argument('--lr', type=float, default=0.0075,
                    help='learning rate')
parser.add_argument('--dt', type=float, default=0.034, 
                    help='step size <dt> of the coRNN')
parser.add_argument('--gamma', type=float, default=1.3,
                    help='y controle parameter <gamma> of the coRNN')
parser.add_argument('--epsilon', type=float, default=12.7,
                    help='z controle parameter <epsilon> of the coRNN')
parser.add_argument('--wandb', dest='wandb', default=False, action='store_true',
                    help='Use weights and biases to log videos of hidden state and training stats')         

args = parser.parse_args()
print(args)

n_inp = 96
n_out = 10

device='cuda'

model_select = {'cornn': model_cornn, 'tcornn1d': model_tcornn1d, 'tcornn2d': model_tcornn2d, 'uncoupled': model_unncoupled}

model = model_select[args.model_type].coRNN(n_inp, args.n_hid, n_out,args.dt, args.gamma, args.epsilon)
train_loader, valid_loader, test_loader = utils.get_data(args.batch,1000)

if args.wandb:
    import wandb
    wandb.init(name=args.run_name,
                project='PROJECT_NAME', 
                entity='ENTITY_NAME', 
                dir='WANDB_DIR',
                config=args)     
    wandb.watch(model)

def log(key, val):
    print(f"{key}: {val}")
    if args.wandb:
        wandb.log({key: val})

fname = f'result/cifar10_log_{args.model_type}_h{args.n_hid}.txt'

## Define the loss
objective = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

rands = torch.randn(1, 1000 - 32, 96)
rand_train = rands.repeat(args.batch,1,1)
rand_test = rands.repeat(1000,1,1)

def test(data_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in data_loader:
            ## Reshape images for sequence learning:
            images = torch.cat((images.permute(0,2,1,3).reshape(1000,32,96),rand_test),dim=1).permute(1,0,2).to(device)
            labels = labels.to(device)
            
            output, _ = model(images, get_seq=False)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(labels.data.view_as(pred)).sum()
    accuracy = 100. * correct / len(data_loader.dataset)

    return accuracy.item()

best_eval = 0.
for epoch in range(args.epochs):
    model.train()
    for i, (images, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
        ## Reshape images for sequence learning:
        images = torch.cat((images.permute(0,2,1,3).reshape(args.batch,32,96),rand_train),dim=1).permute(1,0,2)
        images = images.to(device)
        labels = labels.to(device)

        # Training pass
        optimizer.zero_grad()
        output, _ = model(images, get_seq=False)
        loss = objective(output, labels)
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            log('Train Loss:', loss)
            _, y_seq = model(images, get_seq=True)
            if args.wandb:
                utils.Plot_Vid(y_seq)
            if torch.isnan(loss):
                exit()

    valid_acc = test(valid_loader)
    test_acc = test(test_loader)
    log('Valid Acc:', valid_acc)
    log('Test Acc:', test_acc)

    if (valid_acc > best_eval):
        best_eval = valid_acc
        final_test_acc = test_acc

    Path('result').mkdir(parents=True, exist_ok=True)
    f = open(fname, 'a')
    if (epoch == 0):
        f.write('## learning rate = ' + str(args.lr) + ', dt = ' + str(args.dt) + ', gamma = ' + str(
            args.gamma) + ', epsilon = ' + str(args.epsilon) + '\n')
    f.write('eval accuracy: ' + str(round(valid_acc, 2)) + '\n')
    f.close()

    if (epoch + 1) % 100 == 0:
        args.lr /= 10.
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr

log('Final Test Acc:', test_acc)
f = open(fname, 'a')
f.write('final test accuracy: ' + str(round(final_test_acc, 2)) + '\n')
f.close()