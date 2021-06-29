
import nn

if __name__ == "__main__":
    steps=10000
    lr=0.0001
    batch_size=64
    depth=30
    alpha=-0.9
    randomqu=0.1
    stdnodes=48
    agentlayers=2
    

    temppath=r'C:\\' #insert working directory and store the other code files there, the results will also be stored there


    nn.train_model(temppath,steps,lr,batch_size,depth,alpha,randomqu,stdnodes,agentlayers)
    
    choice='last'
    evalsteps=1000
    nn.use_model(temppath,depth,alpha,stdnodes,agentlayers,choice,evalsteps)
    
