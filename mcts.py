
import numpy as np
import copy
import env

class action:
    def __init__(self,p):
        self.p=p
        self.n=0
        self.q=0
        self.w=0
    
class node:
    def __init__(self,parent,anr,player,level):
        self.parent=parent
        self.anr=anr
        self.player=player
        self.level=level
  
        self.leaf=True
        self.actions=[]
        self.childs=[]
        self.legal=None
        self.rewards=[None,None]
        self.terminal=False
        self.environments=None
        self.value=None
        self.winner=None
        
class tree:
    def __init__(self,environs,p,legal):
        #root node is initialized with child nodes and the probabilities
        self.root=node(None,None,1,0)
        self.root.environments=environs
        self.expand(self.root,p,legal,False)
        self.backup=None
        self.backupped=False
        
    #controls the selection policy of the tree search
    def qplusu(self,node,alpha,eps):
        values=np.zeros(len(node.actions))
        if len(values)==1:
            nexta=0
        
        else:
            if np.random.rand()<eps:
                nexta=np.random.choice(len(node.legal),1,p=node.legal/sum(node.legal))[0]
                
            else:
                nsum=0
                for i in range(len(values)):
                    nsum=nsum+node.actions[i].n
                    
                for i in range(len(values)):
                    if node.childs[i]==None:
                        values[i]=-float('inf')
                        
                    else:
                        if node.actions[i].q<=alpha:
                            values[i]=node.actions[i].q
                        
                        else:
                            values[i]=values[i]+node.actions[i].q
                            values[i]=values[i]+(node.actions[i].p*(np.sqrt(nsum)/(1+node.actions[i].n)))
                        
                maxima=np.argwhere(values == np.amax(values))  
                if len(maxima)==1:
                    nexta=np.argmax(values)
                    
                else:            
                    pvs=np.zeros(len(values))
                    for i in maxima:
                        pvs[i[0]]=node.actions[i[0]].p
                        
                    nexta=np.argmax(pvs)
    
        return nexta
    
    #insert costs, so that the first entry belongs to the player, who finished first
    def detwinner(self,costs):
        #if cost of second flowsheet does not exceed the cost of the first flowsheet by at least 0.01%, first fs wins
        epsilon=0.0001
        if costs[0]-costs[1]>=-1*epsilon*np.abs(costs[0]):
            winner=0
            
        else:
            winner=1
            
        return winner
        
    #search leaf node
    def searchleaf(self,alpha,eps):
        node=self.root
        while node.leaf==False and node.terminal==False:
            nr=self.qplusu(node,alpha,eps)
            node=node.childs[nr]

        #if it is a leaf node, the environments from above are copied and the respective actions are evaluated/stored in the leaf node
        if node.leaf:
            if node.parent.player==node.player:
                node.environments=copy.deepcopy(node.parent.environments)
                node.rewards=copy.deepcopy(node.parent.rewards)
                
            else:
                if node.parent.environments[0].done:
                    node.environments=[copy.deepcopy(node.parent.environments[1]),copy.deepcopy(node.parent.environments[0])]
                    node.rewards=[node.parent.rewards[1],node.parent.rewards[0]]
                    
                else:              
                    copy1=copy.deepcopy(node.parent.environments[0])
                    actio=[]
                    for i in range(len(node.anr)):
                        actio.append(node.anr[i])
                        
                    while len(actio)<3:
                        actio.append(None)
  
                    state,rew,done,conv=copy1.anbauen(actio[0],actio[1],actio[2])
                    node.environments=[copy.deepcopy(node.parent.environments[1]),copy1]
                    node.rewards[0]=node.parent.rewards[1]
                    node.rewards[1]=rew
                    
                    if conv:
                        if node.environments[0].done and node.environments[1].done:
                            node.terminal=True
                            if self.detwinner(node.rewards)==0:
                                node.winner=node.player
                                
                            else:
                                node.winner=-1*node.player
                                    
                            if node.winner==node.player:
                                node.value=1
                                
                            else:
                                node.value=-1
                        
                    #if the simulation did not converge due to a rec, the node is cut off
                    #the algorithm goes one step up, to start at the root again
                    #if this is not possible as, the recycle node was already the root,
                    #a backup of the tree is used (which is always created, before a rec is placed)
                    #the whole tree is replaced by the backup
                    else:
                        node.parent.legal[node.anr[-1]]=0
                        if sum(node.parent.legal)==0:
                            if node.parent.parent==None:
                                del self.root
                                self.root=self.backup
                                self.root.legal[-1]=0
                                self.root.actions[-1]=action(-1)
                                self.root.childs[-1]=None
                                self.backup=None
                                self.backupped=True

                            else:
                                node.parent.parent.legal[-1]=0
                                node.parent.parent.actions[-1]=action(-1)
                                node.parent.parent.childs[-1]=None
                                node.parent=None
                            
                        else:
                            node.parent.actions[node.anr[-1]]=action(-1)
                            node.parent.childs[node.anr[-1]]=None
                            node=None
                            
                        node=self.searchleaf(alpha,eps)

        return node 

    #child nodes are placed, with respect to the probabilities p and the feasible decisions lg, 
    #the trivial case is that the current player has already finished, therefore just a sole connection is placed
    def expand(self,nod,p,lg,trivial):       
        nod.legal=lg
        nod.leaf=False
        if trivial:
            nod.actions.append(action(1))
            nod.childs.append(node(nod,[0],nod.player*-1,0))
            
        else:                
            p=p*lg
            p=p/sum(p)                    
            for i in range(len(p)):
                if lg[i]==1:
                    if nod.level==0:
                        if i==len(lg)-1:
                            nod.actions.append(action(p[i]))
                            nod.childs.append(node(nod,[i],nod.player*-1,0))
                            
                        else:
                            nod.actions.append(action(p[i]))
                            nod.childs.append(node(nod,[i],nod.player,1))
                        
                    if nod.level==1:
                        if i<env.N_Unit-1:
                            nod.actions.append(action(p[i]))
                            nod.childs.append(node(nod,[nod.anr[0],i],nod.player*-1,0))
                            
                        else:
                            nod.actions.append(action(p[i]))
                            nod.childs.append(node(nod,[nod.anr[0],i],nod.player,2))
                        
                    if nod.level==2:
                        nod.actions.append(action(p[i]))
                        nod.childs.append(node(nod,nod.anr+[i],nod.player*-1,0))

                else:
                    nod.actions.append(action(-1))
                    nod.childs.append(None)            
            
    #backup up until the root node
    def back(self,node,value):
        player=copy.deepcopy(node.player)
        while node.parent!=None:
            nr=node.anr[-1]
            node=node.parent
            node.actions[nr].n=node.actions[nr].n+1
            if node.player==player:
                node.actions[nr].w=node.actions[nr].w+value
                
            else:
                node.actions[nr].w=node.actions[nr].w-value
                
            node.actions[nr].q=node.actions[nr].w/node.actions[nr].n
            
    #shift of the tree, the tree above is cut off
    def shift(self,actionnr):
        self.root=self.root.childs[actionnr]
        self.root.parent=None
        
    #choose a move in the real game, based on the visit counts
    def chmove(self,deterministic):
        visitcounts=np.zeros(len(self.root.actions))
        if len(visitcounts)==1:
            return 0,visitcounts
        
        else:
            for i in range(len(visitcounts)):
                visitcounts[i]=self.root.actions[i].n

            visitcounts=visitcounts/sum(visitcounts)

            if deterministic:
                maxima=np.argwhere(visitcounts == np.amax(visitcounts)) 
                if len(maxima)==1:
                    return maxima[0][0],visitcounts
                
                #more than one max, then p is relevant for decision
                else:
                    probs=[]

                    for i in range(len(maxima)):
                        probs.append(self.root.actions[maxima[i][0]].p)
                        
                    return maxima[np.argmax(probs)][0],visitcounts
                
            else:
                for i in range(len(visitcounts)):
                    if self.root.actions[i].p!=-1:
                        visitcounts[i]=np.clip(visitcounts[i],0,1)
                        
                    else:
                        visitcounts[i]=0
    
                ac=np.random.choice(len(visitcounts), 1, p=visitcounts)[0]

                #backup, if rec is chosen
                if self.root.level==1 and ac==env.N_Unit-1 and self.root.actions[ac].n==1:
                    self.backup=copy.deepcopy(self.root)

                return ac,visitcounts
       
