# SynGameZero
[![Python 3.8](https://img.shields.io/badge/Python-3.8-3776AB)](https://www.python.org/downloads/release/python-388/)
[![PyTorch 1.7](https://img.shields.io/badge/PyTorch-1.7-FF6F00?logo=pytorch)](https://github.com/pytorch/pytorch/releases/tag/v1.7.0)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A reinforcement learning (RL) based approach for automated flowsheet synthesis for chemical processes. The agent is trained by transforming the task of creating a profitable flowsheet into a turn based two-player game. This transformation allows to employ a similar training procedure as used for AlphaZero.  

To use the code, just copy all files into one folder. All checkpoints and plots will be stored to a destination, which can be specified in the file main.py with the variable temppath (this is done by nn.train_model). The function nn.use_model loads the chosen checkpoint and uses the respective ANN to play the game evalsteps times on random instances.

## Publications
When using this work, please cite our publications:

[1] **Automated Synthesis of Steady-State Continuous Processes using Reinforcement Learning**  
Q. Göttl, D. G. Grimm, J. Burger  
*Front. Chem. Sci. Eng.* 2021. (https://doi.org/10.1007/s11705-021-2055-9)  

[2] **Automated Process Synthesis Using Reinforcement Learning**  
Q. Göttl, D. G. Grimm, J. Burger  
*Proceedings of the 31st European Symposium on Computer Aided Process Engineering (ESCAPE31)* 2021. (https://doi.org/10.1016/B978-0-323-88506-5.50034-6)

[3] **Automated Flowsheet Synthesis Using Hierarchical Reinforcement Learning: proof of concept**  
Q. Göttl, D. G. Grimm, J. Burger  
*Currently under review*
