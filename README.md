# SynGameZero
[![Python 3.8](https://img.shields.io/badge/Python-3.8-3776AB)](https://www.python.org/downloads/release/python-388/)
[![PyTorch 1.8](https://img.shields.io/badge/PyTorch-1.8-FF6F00?logo=pytorch)](https://github.com/pytorch/pytorch/releases/tag/v1.7.0)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A reinforcement learning (RL) based approach for automated flowsheet synthesis for chemical processes. The agent is trained by transforming the task of creating a profitable flowsheet into a turn based two-player game. This transformation allows to employ a similar training procedure as used for AlphaZero.  

## Install & Setup 
First checkout this repo:

```git clone https://github.com/grimmlab/SynGameZero.git```  

We recommend to use virtual environments to install all dependencies. If not already installed, please install `virtualenv`:  

```pip3 install virtualenv```  

Then create a new virtual environment:  

```virtualenv syngamezero```  

Activate your environment:

```source syngamezero/bin/active```  

Install all dependencies using pip:  

```pip3 install -r requirements.txt```

## Run the Code
In the branch ETBE_synthesis, a training procedure and tree search as described in [3] is shown for the example of ETBE synthesis. The branch ETBE_synthesis2.0 shows an improved version of this framework, which is described in [4].

To run the code just execute:

```python3 main.py```  

You can specifiy the output directory in the main.py file by updating the `temppath` variable. 
The function nn.use_model loads a chosen checkpoint and uses the respective ANN to play the game evalsteps times on random instances.

## Publications
When using this work, please cite our publications:

[1] **Automated Synthesis of Steady-State Continuous Processes using Reinforcement Learning**  
Q. Göttl, D. G. Grimm, J. Burger  
*Front. Chem. Sci. Eng.* 2021. (https://doi.org/10.1007/s11705-021-2055-9)  

[2] **Automated Process Synthesis Using Reinforcement Learning**  
Q. Göttl, D. G. Grimm, J. Burger  
*Proceedings of the 31st European Symposium on Computer Aided Process Engineering (ESCAPE31)* 2021. (https://doi.org/10.1016/B978-0-323-88506-5.50034-6)

[3] **Automated Flowsheet Synthesis Using Hierarchical Reinforcement Learning: proof of concept**  
Q. Göttl, Y. Tönges, D. G. Grimm, J. Burger  
*Chemie Ingenieur Technik* 2021. (https://doi.org/10.1002/cite.202100086)

[4] **Using Reinforcement Learning in a Game-like Setup for Automated Process Synthesis without Prior Process Knowledge**  
Q. Göttl, D. G. Grimm, J. Burger  
*Computer Aided Chemical Engineering* 2022. (https://doi.org/10.1016/B978-0-323-85159-6.50259-1)
