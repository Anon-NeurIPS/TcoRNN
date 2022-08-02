# Topographic Coupled Oscillator Networks Learn Transformations as Traveling Waves
## Under Review @ NeurIPS 2022

This repository contains all code necessary to reproduce the experiments published in the paper. Additionally, we include video visualizations of the depicted traveling waves in the README below. 

## To run the code
- The code to reproduce Tables 1 & 2 can be found in the Supervised_Sequence_Modeling directory
- The code to reproduce Figures 1 & 2 can be found in the MNIST directory
- The code to reproduce Table 5 can be found in the Hamiltonian_Dynamics directory

## Hidden State Wave Visualizations 

#### 2D TcoRNN on Rotating MNIST
Video version of Figure 1:

Before Training:

<img src="https://github.com/Anon-NeurIPS/TcoRNN/blob/master/figures/TcoRNN_2d_mnist/before_training_rot_mnist.gif" width="100" height="100" /> <img src="https://github.com/Anon-NeurIPS/TcoRNN/blob/master/figures/TcoRNN_2d_mnist/before_training_rot_mnist_phase.gif" width="100" height="100" />

After Training: 

<img src="https://github.com/Anon-NeurIPS/TcoRNN/blob/master/figures/TcoRNN_2d_mnist/wave_s4main_pos.gif" width="100" height="100" /> <img src="https://github.com/Anon-NeurIPS/TcoRNN/blob/master/figures/TcoRNN_2d_mnist/wave_s4main_phase.gif" width="100" height="100" />

Hidden state of same model as in Figure 1, but on different data samples (i.e. video version of Figure 4 in Supplementary Material):

<img src="https://github.com/Anon-NeurIPS/TcoRNN/blob/master/figures/TcoRNN_2d_mnist/wave_s1_pos.gif" width="100" height="100" /> <img src="https://github.com/Anon-NeurIPS/TcoRNN/blob/master/figures/TcoRNN_2d_mnist/wave_s1_phase.gif" width="100" height="100" /> 
<img src="https://github.com/Anon-NeurIPS/TcoRNN/blob/master/figures/TcoRNN_2d_mnist/wave_s2_pos.gif" width="100" height="100" /> <img src="https://github.com/Anon-NeurIPS/TcoRNN/blob/master/figures/TcoRNN_2d_mnist/wave_s2_phase.gif" width="100" height="100" />
<img src="https://github.com/Anon-NeurIPS/TcoRNN/blob/master/figures/TcoRNN_2d_mnist/wave_s3_pos.gif" width="100" height="100" /> <img src="https://github.com/Anon-NeurIPS/TcoRNN/blob/master/figures/TcoRNN_2d_mnist/wave_s3_phase.gif" width="100" height="100" />
<img src="https://github.com/Anon-NeurIPS/TcoRNN/blob/master/figures/TcoRNN_2d_mnist/wave_s5_pos.gif" width="100" height="100" /> <img src="https://github.com/Anon-NeurIPS/TcoRNN/blob/master/figures/TcoRNN_2d_mnist/wave_s5_phase.gif" width="100" height="100" />

Hidden state of 2D TcoRNN with different random initalizations:

<img src="https://github.com/Anon-NeurIPS/TcoRNN/blob/master/figures/TcoRNN_2d_mnist/randinit2_waves_pos.gif" width="100" height="100" /> <img src="https://github.com/Anon-NeurIPS/TcoRNN/blob/master/figures/TcoRNN_2d_mnist/randinit2_waves_phase.gif" width="100" height="100" /> 
<img src="https://github.com/Anon-NeurIPS/TcoRNN/blob/master/figures/TcoRNN_2d_mnist/randinit3_waves_pos.gif" width="100" height="100" /> <img src="https://github.com/Anon-NeurIPS/TcoRNN/blob/master/figures/TcoRNN_2d_mnist/randinit3_waves_phase.gif" width="100" height="100" />
<img src="https://github.com/Anon-NeurIPS/TcoRNN/blob/master/figures/TcoRNN_2d_mnist/randinit4_waves_pos.gif" width="100" height="100" /> <img src="https://github.com/Anon-NeurIPS/TcoRNN/blob/master/figures/TcoRNN_2d_mnist/randinit4_waves_phase.gif" width="100" height="100" />

#### 1D TcoRNN on Rotating MNIST 
(each row is a disjoint circular subspace)

<img src="https://github.com/Anon-NeurIPS/TcoRNN/blob/master/figures/TcoRNN_1d_mnist/waves_1d_pos_s1.gif" width="162" height="72" /> <img src="https://github.com/Anon-NeurIPS/TcoRNN/blob/master/figures/TcoRNN_1d_mnist/waves_1d_phase_s1.gif" width="162" height="72" />


#### 2D TcoRNN on Spring Task 
(Ground Truth, Forward Extrapolated Reconstruction, Hidden State, Phase)

<img src="https://github.com/Anon-NeurIPS/TcoRNN/blob/master/figures/TcoRNN_2d_spring/gt.gif" width="100" height="100" /> <img src="https://github.com/Anon-NeurIPS/TcoRNN/blob/master/figures/TcoRNN_2d_spring/recon.gif" width="100" height="100" /> <img src="https://github.com/Anon-NeurIPS/TcoRNN/blob/master/figures/TcoRNN_2d_spring/pos.gif" width="100" height="100" /> <img src="https://github.com/Anon-NeurIPS/TcoRNN/blob/master/figures/TcoRNN_2d_spring/phase.gif" width="100" height="100" /> 

#### 1D TcoRNN on Spring Task 
(Each row is a disjoint circular subspace) </br>
(Ground Truth, Forward Extrapolated Reconstruction, Hidden State, Phase)

<img src="https://github.com/Anon-NeurIPS/TcoRNN/blob/master/figures/TcoRNN_1d_spring/gt.gif" width="100" height="100" /> <img src="https://github.com/Anon-NeurIPS/TcoRNN/blob/master/figures/TcoRNN_1d_spring/recon.gif" width="100" height="100" /> <img src="https://github.com/Anon-NeurIPS/TcoRNN/blob/master/figures/TcoRNN_1d_spring/pos.gif" width="100" height="100" /> <img src="https://github.com/Anon-NeurIPS/TcoRNN/blob/master/figures/TcoRNN_1d_spring/phase.gif" width="100" height="100" /> 

#### Globally Coupled coRNN on Spring Task 
(Ground Truth, Forward Extrapolated Reconstruction, Hidden State, Phase)

<img src="https://github.com/Anon-NeurIPS/TcoRNN/blob/master/figures/coRNN_Spring/gt.gif" width="100" height="100" /> <img src="https://github.com/Anon-NeurIPS/TcoRNN/blob/master/figures/coRNN_Spring/recon.gif" width="100" height="100" /> <img src="https://github.com/Anon-NeurIPS/TcoRNN/blob/master/figures/coRNN_Spring/pos.gif" width="100" height="100" /> <img src="https://github.com/Anon-NeurIPS/TcoRNN/blob/master/figures/coRNN_Spring/phase.gif" width="100" height="100" /> 


#### 2D TcoRNN on Pendulum Task
<img src="https://github.com/Anon-NeurIPS/TcoRNN/blob/master/figures/TcoRNN_2d_pendulum/gt.gif" width="100" height="100" /> <img src="https://github.com/Anon-NeurIPS/TcoRNN/blob/master/figures/TcoRNN_2d_pendulum/recon.gif" width="100" height="100" /> <img src="https://github.com/Anon-NeurIPS/TcoRNN/blob/master/figures/TcoRNN_2d_pendulum/pos.gif" width="100" height="100" /> <img src="https://github.com/Anon-NeurIPS/TcoRNN/blob/master/figures/TcoRNN_2d_pendulum/phase.gif" width="100" height="100" /> 

#### 1D TcoRNN on Pendulum Task
(Each row is a disjoint circular subspace)

<img src="https://github.com/Anon-NeurIPS/TcoRNN/blob/master/figures/TcoRNN_1d_pendulum/gt.gif" width="100" height="100" /> <img src="https://github.com/Anon-NeurIPS/TcoRNN/blob/master/figures/TcoRNN_1d_pendulum/recon.gif" width="100" height="100" /> <img src="https://github.com/Anon-NeurIPS/TcoRNN/blob/master/figures/TcoRNN_1d_pendulum/pos.gif" width="100" height="100" /> <img src="https://github.com/Anon-NeurIPS/TcoRNN/blob/master/figures/TcoRNN_1d_pendulum/phase.gif" width="100" height="100" /> 

#### Globally Coupled coRNN on Pendulum Task
<img src="https://github.com/Anon-NeurIPS/TcoRNN/blob/master/figures/coRNN_pendulum/gt.gif" width="100" height="100" /> <img src="https://github.com/Anon-NeurIPS/TcoRNN/blob/master/figures/coRNN_pendulum/recon.gif" width="100" height="100" /> <img src="https://github.com/Anon-NeurIPS/TcoRNN/blob/master/figures/coRNN_pendulum/pos.gif" width="100" height="100" /> <img src="https://github.com/Anon-NeurIPS/TcoRNN/blob/master/figures/coRNN_pendulum/phase.gif" width="100" height="100" /> 


#### 2D TcoRNN on 2-Body Task
<img src="https://github.com/Anon-NeurIPS/TcoRNN/blob/master/figures/TcoRNN_2d_2body/gt.gif" width="100" height="100" /> <img src="https://github.com/Anon-NeurIPS/TcoRNN/blob/master/figures/TcoRNN_2d_2body/recon.gif" width="100" height="100" /> <img src="https://github.com/Anon-NeurIPS/TcoRNN/blob/master/figures/TcoRNN_2d_2body/pos.gif" width="100" height="100" /> <img src="https://github.com/Anon-NeurIPS/TcoRNN/blob/master/figures/TcoRNN_2d_2body/phase.gif" width="100" height="100" /> 

#### 1D TcoRNN on 2-Body Task
(Each row is a disjoint circular subspace)

<img src="https://github.com/Anon-NeurIPS/TcoRNN/blob/master/figures/TcoRNN_1d_2body/gt.gif" width="100" height="100" /> <img src="https://github.com/Anon-NeurIPS/TcoRNN/blob/master/figures/TcoRNN_1d_2body/recon.gif" width="100" height="100" /> <img src="https://github.com/Anon-NeurIPS/TcoRNN/blob/master/figures/TcoRNN_1d_2body/pos.gif" width="100" height="100" /> <img src="https://github.com/Anon-NeurIPS/TcoRNN/blob/master/figures/TcoRNN_1d_2body/phase.gif" width="100" height="100" /> 

#### Globally Coupled coRNN on 2-Body Task
<img src="https://github.com/Anon-NeurIPS/TcoRNN/blob/master/figures/coRNN_2body/gt.gif" width="100" height="100" /> <img src="https://github.com/Anon-NeurIPS/TcoRNN/blob/master/figures/coRNN_2body/recon.gif" width="100" height="100" /> <img src="https://github.com/Anon-NeurIPS/TcoRNN/blob/master/figures/coRNN_2body/pos.gif" width="100" height="100" /> <img src="https://github.com/Anon-NeurIPS/TcoRNN/blob/master/figures/coRNN_2body/phase.gif" width="100" height="100" /> 



#### 2D TcoRNN on Mujoco Circle Task
<img src="https://github.com/Anon-NeurIPS/TcoRNN/blob/master/figures/TcoRNN_2d_MJcircle/gt.gif" width="100" height="100" /> <img src="https://github.com/Anon-NeurIPS/TcoRNN/blob/master/figures/TcoRNN_2d_MJcircle/recon.gif" width="100" height="100" /> <img src="https://github.com/Anon-NeurIPS/TcoRNN/blob/master/figures/TcoRNN_2d_MJcircle/pos.gif" width="100" height="100" /> <img src="https://github.com/Anon-NeurIPS/TcoRNN/blob/master/figures/TcoRNN_2d_MJcircle/phase.gif" width="100" height="100" /> 

#### 1D TcoRNN on Mujoco Circle Task
(Each row is a disjoint circular subspace)

<img src="https://github.com/Anon-NeurIPS/TcoRNN/blob/master/figures/TcoRNN_1d_MJcircle/gt.gif" width="100" height="100" /> <img src="https://github.com/Anon-NeurIPS/TcoRNN/blob/master/figures/TcoRNN_1d_MJcircle/recon.gif" width="100" height="100" /> <img src="https://github.com/Anon-NeurIPS/TcoRNN/blob/master/figures/TcoRNN_1d_MJcircle/pos.gif" width="100" height="100" /> <img src="https://github.com/Anon-NeurIPS/TcoRNN/blob/master/figures/TcoRNN_1d_MJcircle/phase.gif" width="100" height="100" /> 

#### Globally Coupled coRNN on Mujoco Circle Task
<img src="https://github.com/Anon-NeurIPS/TcoRNN/blob/master/figures/coRNN_MJcircle/gt.gif" width="100" height="100" /> <img src="https://github.com/Anon-NeurIPS/TcoRNN/blob/master/figures/coRNN_MJcircle/recon.gif" width="100" height="100" /> <img src="https://github.com/Anon-NeurIPS/TcoRNN/blob/master/figures/coRNN_MJcircle/pos.gif" width="100" height="100" /> <img src="https://github.com/Anon-NeurIPS/TcoRNN/blob/master/figures/coRNN_MJcircle/phase.gif" width="100" height="100" /> 



#### 2D TcoRNN on Adding Task
<img src="https://github.com/Anon-NeurIPS/TcoRNN/blob/master/figures/TcoRNN_2d_adding/pos.gif" width="100" height="100" />

#### Globally Coupled coRNN on Adding Task
<img src="https://github.com/Anon-NeurIPS/TcoRNN/blob/master/figures/coRNN_adding/pos.gif" width="100" height="100" />


#### 2D TcoRNN (small) on sMNIST Task
<img src="https://github.com/Anon-NeurIPS/TcoRNN/blob/master/figures/TcoRNN_2d_sMNIST/pos.gif" width="100" height="100" />


#### 2D TcoRNN (medium) on sMNIST Task
Before Training:

<img src="https://github.com/Anon-NeurIPS/TcoRNN/blob/master/figures/TcoRNN_2d_sMNIST/sMNIST_medium_before_train.gif" width="100" height="100" />

After Training

<img src="https://github.com/Anon-NeurIPS/TcoRNN/blob/master/figures/TcoRNN_2d_sMNIST/sMNIST_medium_after_train.gif" width="100" height="100" />

#### 2D TcoRNN (medium) on Random Noise
<img src="https://github.com/Anon-NeurIPS/TcoRNN/blob/master/figures/TcoRNN_2d_sMNIST/Tcornn_2d_NoiseTrain.gif" width="100" height="100" />

#### 1D TcoRNN on sMNIST Task
<img src="https://github.com/Anon-NeurIPS/TcoRNN/blob/master/figures/TcoRNN_1d_sMNIST/pos.gif" width="100" height="100" />

#### Globally Coupled coRNN on sMNIST Task
<img src="https://github.com/Anon-NeurIPS/TcoRNN/blob/master/figures/coRNN_sMNIST/pos.gif" width="100" height="100" />

#### Uncoupled coRNN on sMNIST Task
<img src="https://github.com/Anon-NeurIPS/TcoRNN/blob/master/figures/uncoupled_sMNIST/pos.gif" width="100" height="100" />
