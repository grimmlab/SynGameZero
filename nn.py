# -*- coding: utf-8 -*-

from collections import deque
import torch

import numpy as np
import env
import matplotlib.pyplot as plt
import mcts
import copy
import os

#class for neural network
class Net(torch.nn.Module):
    def __init__(self,stdnodes,agentlayers):
        super(Net, self).__init__()

        self.conv=torch.nn.ModuleList([])
        self.conv.append(torch.nn.Conv2d(env.linelength, 16, 1))
        self.conv.append(torch.nn.Conv2d(16, 8, 1))

        self.level1=torch.nn.ModuleList([])
        self.level1.append(torch.nn.Flatten(1,-1))
        self.level1.append(torch.nn.Linear(256,stdnodes))
        for i in range(agentlayers-1):
            self.level1.append(torch.nn.Linear(stdnodes,stdnodes))
        
        self.actor1=torch.nn.ModuleList([])
        self.actor1.append(torch.nn.Linear(stdnodes,env.N_Matrix+1))
        
        self.critic1=torch.nn.ModuleList([])
        self.critic1.append(torch.nn.Linear(stdnodes,1))
        
        self.level2=torch.nn.ModuleList([])
        self.level2.append(torch.nn.Linear(stdnodes+env.N_Matrix,stdnodes))
        for i in range(agentlayers-1):
            self.level2.append(torch.nn.Linear(stdnodes,stdnodes))

        self.actor2=torch.nn.ModuleList([])
        self.actor2.append(torch.nn.Linear(stdnodes,env.N_Unit))
        
        self.critic2=torch.nn.ModuleList([])
        self.critic2.append(torch.nn.Linear(stdnodes,1))
        
        self.level3=torch.nn.ModuleList([])
        self.level3.append(torch.nn.Linear(stdnodes+stdnodes,stdnodes))
        for i in range(agentlayers-1):
            self.level3.append(torch.nn.Linear(stdnodes,stdnodes))

        self.actor3=torch.nn.ModuleList([])
        self.actor3.append(torch.nn.Linear(stdnodes,env.N_Matrix))
        
        self.critic3=torch.nn.ModuleList([])
        self.critic3.append(torch.nn.Linear(stdnodes,1))
        
    def forward(self,x,y,z):
        x=x.astype(np.float32)
        x=torch.tensor(x)
        y=y.astype(np.float32)
        y=torch.tensor(y)
        z=z.astype(np.float32)
        z=torch.tensor(z)
        
        for i in range(len(self.conv)):
            x=self.conv[i](x)
            x=torch.nn.functional.relu(x)
            
        x=self.level1[0](x)
        x=self.run(self.level1[1:],x)

        p=self.run(self.actor1[:-1],x)            
        p=self.actor1[-1](p)
        p=torch.nn.functional.softmax(p,dim=1)
            
        v=self.run(self.critic1[:-1],x)
        v=self.critic1[-1](v)
        v=torch.tanh(v)
        
        le2=torch.cat((x,y),1) 
        le2=self.run(self.level2,le2)

        p2=self.run(self.actor2[:-1],le2)
        p2=self.actor2[-1](p2)
        p2=torch.nn.functional.softmax(p2,dim=1)
        
        v2=self.run(self.critic2[:-1],le2)
        v2=self.critic2[-1](v2)
        v2=torch.tanh(v2)
        
        le3=torch.cat((le2,z),1) 
        le3=self.run(self.level3,le3)
         
        p3=self.run(self.actor3[:-1],le3)
        p3=self.actor3[-1](p3)
        p3=torch.nn.functional.softmax(p3,dim=1)
        
        v3=self.run(self.critic3[:-1],le3)
        v3=self.critic3[-1](v3)
        v3=torch.tanh(v3)
 
        return [p,v,p2,v2,p3,v3]
    
    def run(self,liste,inp):
        for i in range(len(liste)):
            inp=liste[i](inp)
            inp=torch.nn.functional.relu(inp)
            
        return inp
    
#training_steps
def train_step(inp,inp2,inp3,targa,targc,index,b_size,neuralnet,optimizer):
    targa=torch.tensor(targa)
    targc=torch.reshape(torch.tensor(targc),(b_size,1))

    nnret=neuralnet(inp,inp2,inp3)

    lossa = torch.sum(torch.square(nnret[(index*2)]-targa))
    lossa.backward()
    
    torch.nn.utils.clip_grad_norm_(neuralnet.parameters(),1,2)
    optimizer.step() 
    optimizer.zero_grad()
    
    nnret=neuralnet(inp,inp2,inp3)
    
    lossc = torch.sum(torch.square(nnret[(index*2)+1]-targc))
    lossc.backward()
    
    torch.nn.utils.clip_grad_norm_(neuralnet.parameters(),1,2)
    optimizer.step() 
    optimizer.zero_grad()
  
    return ((lossa+lossc)/b_size).cpu().detach().numpy()

#collect states from environments
def statebig(envs):
    m=np.zeros((2,env.N_Matrix,env.linelength))
    m[0]=envs[0].state
    m[1]=envs[1].state

    return m.reshape(1,env.linelength,env.N_Matrix,2)

#sample 3 feed streams
def start():
    if np.random.rand(1)<0.2:
        molar_flowrates=np.random.rand(3)+0.1        
        molar_flowrates[0]=copy.deepcopy(molar_flowrates[1])

    else:
        molar_flowrates=np.random.rand(3)+0.1
        
    return molar_flowrates

#depth guides the depth of the tree, randomqu is the rate of random select steps during tree search
def train_model(generalpath,n_steps,learning_rate,batch_size,
          depth,alpha,randomqu,stdnodes,agentlayers):    
    #checkpoint intervals    
    cp_interval=int(n_steps/10)
    cpcount=0
    
    #three memories for the levels
    replay_memory_size = 10*batch_size
    replay_memory=[]
    for i in range(3):
        replay_memory.append(deque([], maxlen=replay_memory_size))
        
    #function for sampling batches
    minbsize=2
    def sample_memories(l,index):
        l=np.min([l,int(len(replay_memory[index])/4)])
        indices = np.random.permutation(len(replay_memory[index]))[:l]
        cols = [[],[],[],[],[]] # state, s2, s3, action, reward
        for idx in indices:
            memory = replay_memory[index][idx]
            for col, value in zip(cols, memory):
                col.append(value)
        
        cols = [np.array(col) for col in cols]
   
        return cols[0].reshape(l,env.linelength,env.N_Matrix,2), cols[1], cols[2], cols[3],cols[4],l
    
    #variables for the storage of loss over training process
    losssafe=[]
    losssafe2=[]
    rewsafe=[]
    rewsafe2=[]
    winsafe=[]
    winsafe2=[]
    costvar=int(n_steps/100)
    
    #clipvalue for probabilities for actions (to prevent completely sharp distributions)
    cut=0.0001

    #initialize net and optimizer
    neuralnet=Net(stdnodes,agentlayers)
    optimizer=torch.optim.Adam(neuralnet.parameters(), lr=learning_rate)

    step=0
    
    while step<n_steps:
        molar_flowrates=start()
            
        environment1=env.env([np.array([0,0,0,molar_flowrates[0]]),np.array([0,molar_flowrates[1],molar_flowrates[2],0])],1)
        environment2=env.env([np.array([0,0,0,molar_flowrates[0]]),np.array([0,molar_flowrates[1],molar_flowrates[2],0])],-1)
        
        environments=[environment1,environment2]
        
        #generation of nn output for first node to initialize tree
        leg=env.legal(environments[0].state,0,None,None)
        nno=neuralnet(statebig(environments),np.zeros((1,env.N_Matrix)),np.zeros((1,stdnodes)))
        nno[0]=nno[0].cpu().detach().numpy()[0]
        tree=mcts.tree(environments,nno[0],leg)
        
        obs_tot=[]
        while not tree.root.terminal:
            for i in range(depth):
                leaf=tree.searchleaf(alpha,randomqu)
                if not leaf.terminal:
                    if leaf.environments[0].done:
                        tree.expand(leaf,None,None,True)
                        
                    else:
                        if leaf.level==0:
                            nno=neuralnet(statebig(leaf.environments),np.zeros((1,env.N_Matrix)),np.zeros((1,stdnodes)))
                            probs=nno[0].cpu().detach().numpy()[0]
                            value=nno[1].cpu().detach().numpy()[0]
                            leg=env.legal(leaf.environments[0].state,leaf.level,None,None)
                            
                        if leaf.level==1:
                            s2=np.zeros(env.N_Matrix)
                            s2[leaf.anr[-1]]=1
                            leg=env.legal(leaf.environments[0].state,leaf.level,leaf.anr[-1],None)
                            
                            nno=neuralnet(statebig(leaf.environments),s2.reshape(1,env.N_Matrix),np.zeros((1,stdnodes)))
                            probs=nno[2].cpu().detach().numpy()[0]
                            value=nno[3].cpu().detach().numpy()[0]
                            
                        if leaf.level==2:
                            s2=np.zeros(env.N_Matrix)
                            s2[leaf.anr[0]]=1
                            if leaf.anr[-1]==3:
                                s3=np.zeros((1,stdnodes))
                                
                            else:
                                s3=np.ones((1,stdnodes))
                                
                            leg=env.legal(leaf.environments[0].state,leaf.level,leaf.anr[0],leaf.anr[1])
                            nno=neuralnet(statebig(leaf.environments),s2.reshape(1,env.N_Matrix),s3)
                            
                            probs=nno[4].cpu().detach().numpy()[0]
                            value=nno[5].cpu().detach().numpy()[0]

                        probs=np.clip(probs,cut,1)
                        tree.expand(leaf,probs,leg,False)
                        tree.back(leaf,value)
                          
                else:
                    tree.back(leaf,leaf.value)
                
            action,ps = tree.chmove(False)
 
            if tree.backupped:
                del obs_tot[-1]
                tree.backupped=False

            if len(tree.root.actions)>1:
                memo_index=tree.root.level
                s2=np.zeros(env.N_Matrix)
                s3=np.zeros(stdnodes)
                if tree.root.level==1:
                    s2[tree.root.anr[-1]]=1
                    
                if tree.root.level==2:
                    s2[tree.root.anr[0]]=1
                    if tree.root.anr[-1]==env.N_Unit-1:
                        s3=np.ones(stdnodes)

                ps=np.clip(ps,cut,1)
                ps=ps/sum(ps)
                obs_tot.append([statebig(tree.root.environments),s2,s3,ps,tree.root.player,memo_index])
            
            tree.shift(action)
        
        print(tree.root.player,tree.root.environments[0].blueprint,tree.root.environments[1].blueprint,tree.root.winner,tree.root.rewards)

        #save tuples
        for i in range(len(obs_tot)):
            if tree.root.winner==obs_tot[i][-2]:
                replay_memory[obs_tot[i][-1]].append((obs_tot[i][0],obs_tot[i][1],obs_tot[i][2],obs_tot[i][3],1))
            
            else:
                replay_memory[obs_tot[i][-1]].append((obs_tot[i][0],obs_tot[i][1],obs_tot[i][2],obs_tot[i][3],-1))
        
        #wait with training until at least first storage is filled a little bit
        cont=False
        for repmem in range(3):
            if len(replay_memory[repmem])<minbsize*4:
                cont=True
            
        if cont:
            continue

        print('step',step)
        #checkpoint saving
        if step%cp_interval==0:
            torch.save(neuralnet.state_dict(), os.path.join(generalpath,"/checkpoint%s"%str(cpcount)))
            cpcount=cpcount+1
        
        totlosstoplot=0
        denom=0
        for i in range(3):
            denom=denom+1
            X_state_val, X_state2_val, X_state3_val, X_action_val, rewards, b_size=sample_memories(batch_size,i)
            losstoplot=train_step(X_state_val,X_state2_val,X_state3_val,X_action_val, rewards, i, b_size,neuralnet,optimizer)
            totlosstoplot=totlosstoplot+losstoplot                            
            
        totlosstoplot=totlosstoplot/denom
        
        losssafe.append(totlosstoplot)
        rewsafe.append(np.max(tree.root.rewards))
        winsafe.append(tree.root.winner)
        print(totlosstoplot,sum(winsafe)/len(winsafe))
        if len(losssafe)==costvar:
            losssafe2.append(sum(losssafe)/len(losssafe))
            rewsafe2.append(sum(rewsafe)/len(rewsafe))
            winsafe2.append(sum(winsafe)/len(winsafe))
            losssafe=[]
            rewsafe=[]
            winsafe=[]

        #save nno and plot cost, winner and loss
        if step==n_steps-1:
            torch.save(neuralnet.state_dict(), os.path.join(generalpath,"/last"))
            
            plt.plot(losssafe2,linewidth=1,color='black')
            plt.plot(winsafe2,linewidth=1,color='green')
            plt.xlabel(r'Training Steps')
            plt.xticks([0,50,100],('0', str(int(n_steps/2)), str(n_steps)))
            plt.savefig(os.path.join(generalpath,"/loss.pdf"))
            plt.close()
            
            plt.plot(rewsafe2,linewidth=1,color='black')

            plt.xlabel(r'Training Steps')
            plt.xticks([0,50,100],('0', str(int(n_steps/2)), str(n_steps)))
            plt.savefig(os.path.join(generalpath,"/cost.pdf"))
            plt.close()
            
        step=step+1

def use_model(generalpath,depth,alpha,stdnodes,agentlayers,choice,evalsteps):    
    if choice=='last':
        restorepath=os.path.join(generalpath,"/last")
        
    else:
        restorepath=os.path.join(generalpath,"/checkpoint"+str(choice))
        
    neuralnet=Net(stdnodes,agentlayers)
    neuralnet.load_state_dict(torch.load(restorepath))
    neuralnet.eval()

    for stepis in range(evalsteps):
        molar_flowrates=start()
            
        environment1=env.env([np.array([0,0,0,molar_flowrates[0]]),np.array([0,molar_flowrates[1],molar_flowrates[2],0])],1)
        environment2=env.env([np.array([0,0,0,molar_flowrates[0]]),np.array([0,molar_flowrates[1],molar_flowrates[2],0])],-1)
        
        environments=[environment1,environment2]
        
        #generation of nn output for first node to initialize tree
        leg=env.legal(environments[0].state,0,None,None)
        nno=neuralnet(statebig(environments),np.zeros((1,env.N_Matrix)),np.zeros((1,stdnodes)))
        nno[0]=nno[0].cpu().detach().numpy()[0]
        tree=mcts.tree(environments,nno[0],leg)
        
        while not tree.root.terminal:
            for i in range(depth):
                leaf=tree.searchleaf(alpha,0)
                if not leaf.terminal:
                    if leaf.environments[0].done:
                        tree.expand(leaf,None,None,True)
                        
                    else:
                        if leaf.level==0:
                            nno=neuralnet(statebig(leaf.environments),np.zeros((1,env.N_Matrix)),np.zeros((1,stdnodes)))
                            probs=nno[0].cpu().detach().numpy()[0]
                            value=nno[1].cpu().detach().numpy()[0]
                            leg=env.legal(leaf.environments[0].state,leaf.level,None,None)
                            
                        if leaf.level==1:
                            s2=np.zeros(env.N_Matrix)
                            s2[leaf.anr[-1]]=1
                            leg=env.legal(leaf.environments[0].state,leaf.level,leaf.anr[-1],None)
                            
                            nno=neuralnet(statebig(leaf.environments),s2.reshape(1,env.N_Matrix),np.zeros((1,stdnodes)))
                            probs=nno[2].cpu().detach().numpy()[0]
                            value=nno[3].cpu().detach().numpy()[0]
                            
                        if leaf.level==2:
                            s2=np.zeros(env.N_Matrix)
                            s2[leaf.anr[0]]=1
                            if leaf.anr[-1]==3:
                                s3=np.zeros((1,stdnodes))
                                
                            else:
                                s3=np.ones((1,stdnodes))
                                
                            leg=env.legal(leaf.environments[0].state,leaf.level,leaf.anr[0],leaf.anr[1])
                            nno=neuralnet(statebig(leaf.environments),s2.reshape(1,env.N_Matrix),s3)
                            
                            probs=nno[4].cpu().detach().numpy()[0]
                            value=nno[5].cpu().detach().numpy()[0]

                        probs=np.clip(probs,0.0001,1)
                        tree.expand(leaf,probs,leg,False)
                        tree.back(leaf,value)
                          
                else:
                    tree.back(leaf,leaf.value)
                
            action,ps = tree.chmove(False)
 
            if tree.backupped:
                tree.backupped=False

            tree.shift(action)
        
        print(tree.root.player,tree.root.environments[0].blueprint,tree.root.environments[1].blueprint,tree.root.winner,tree.root.rewards)
        print('\n')
        
