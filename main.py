import os
import nn

if __name__ == "__main__":
    steps=10000
    lr=0.0001
    batch_size=64
    #depth guides the depth of the tree
    depth=30
    alpha=-0.9
    #randomqu is the rate of random select steps during tree search
    randomqu=0.1
    stdnodes=64
    agentlayers=2

    #insert working directory and store the other code files there, the results will also be stored there (default: current directory)
    temppath= os.getcwd()

    nn.train_model(temppath,steps,lr,batch_size,depth,alpha,randomqu,stdnodes,agentlayers)

    #choose checkpoint to load
    choice='last'
    evalsteps=1000
    nn.use_model(temppath,depth,alpha,stdnodes,agentlayers,choice,evalsteps)

