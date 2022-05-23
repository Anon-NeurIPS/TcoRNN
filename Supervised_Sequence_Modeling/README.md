<h1 align='center'> Topographic Coupled Oscillatory Recurrent Neural Network (TcoRNN)<br>
    [Under Review @ NeurIPS 2022] </h1>

This repository is an extension of the original repository [coRNN Github](https://github.com/tk-rusch/coRNN) for the original paper  [Coupled Oscillatory Recurrent Neural Network (coRNN): An accurate and (gradient) stable architecture for learning long time dependencies](https://openreview.net/forum?id=F3s69XzWOia).

## Requirements

```bash
pytorch 1.3+
torchvision 0.4+
torchtext 0.6+
numpy 1.17+
spacy v2.2+
Optional: wandb
```
For each experiment, you can add the flag '--wandb' to enable logging with weights & biases. This will log all of the training metrics as well as videos of the hidden state over sequences for visual inspection. Note you will need to add your project and entity names to the `wandb.init()` call inside each '*_task.py' training script.

## Example Cell Implementation 
```python
from torch import nn
import torch

class TcoRNN_2d_Cell(nn.Module):
    def __init__(self, n_inp, spatial_dim, dt, gamma, epsilon):
        super(TcoRNN_2d_Cell, self).__init__()
        self.dt = dt
        self.gamma = gamma
        self.epsilon = epsilon
        self.spatial_dim = spatial_dim
        self.Wx = nn.Linear(n_inp, spatial_dim * spatial_dim).to(device)
        self.Wz = nn.Conv2d(1, 1, 3, padding=1, padding_mode='circular').to(device)
        self.Wy = nn.Conv2d(1, 1, 3, padding=1, padding_mode='circular').to(device)

    def forward(self,x,hy,hz):
        hz = hz + self.dt * (torch.tanh(self.Wx(x) 
                                         + self.Wz(hz.view(-1, 1, self.spatial_dim, self.spatial_dim)).flatten(start_dim=1)
                                         + self.Wy(hy.view(-1, 1, self.spatial_dim, self.spatial_dim)).flatten(start_dim=1))
                                   - self.gamma * hy - self.epsilon * hz)
        hy = hy + self.dt * hz

        return hy, hz
```