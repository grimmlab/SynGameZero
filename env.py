
import numpy as np
import copy
import scipy.optimize as opt
import units

N_Matrix=16
#etbe,ib,nbut,ethanol
N_Comp=4
#r,d_L,d_H,mix/rec
N_Unit=4
linelength=(N_Comp+1)+N_Unit+N_Matrix+1+1

def above0(arr):
    for i in range(len(arr)):
        if arr[i]<0:
            arr[i]=0
            
    return arr

def molfr(arr):
    if sum(arr)>0:  
        fr=arr/sum(arr)
        
    else:
        fr=np.zeros(len(arr))
    
    return fr

#filter feasible actions depending on the state and decisions on previous levels (z1 stores streamchoice)
def legal(state,level,z1):
    if level==0:
        feasible=np.zeros(N_Matrix+1)
        #last action is termination
        feasible[-1]=1
        
        #check for open, existing streams
        for i in range(len(state)):
            if state[i][-1]==1 and state[i][N_Comp]>0 and sum(state[i][N_Comp+1:N_Comp+1+N_Unit])==0:
                feasible[i]=1
        
        #check if matrix full
        if state[-1][-1]==1:
            feasible[0:N_Matrix]=np.zeros(N_Matrix)
            
    if level==1:
        feasible=np.zeros(N_Unit)
        openstreams=0
        closed_withoutrec=0
        for i in range(len(state)):
            if state[i][-1]==1 and state[i][N_Comp]>0 and sum(state[i][N_Comp+1:N_Comp+1+N_Unit])==0:
                openstreams=openstreams+1
                
            #count closed streams, which are no recycles themself (these can be the destination of a possible recycle)
            if state[i][-1]==1 and state[i][N_Comp]>0 and sum(state[i][N_Comp+1:N_Comp+1+N_Unit])==1 and state[i][N_Comp+1+N_Unit-1]==0:
                closed_withoutrec=closed_withoutrec+1
               
        if state[-1][-1]==0:
            feasible[0]=1        

        if state[-2][-1]==0:
            feasible[1:3]=feasible[1:3]+1            
            
        if openstreams>1 and closed_withoutrec>0:
            feasible[-1]=1


    if level==2:
        feasible=np.zeros(N_Matrix)

        #search open streams
        for i in range(len(state)):
            if state[i][-1]==1 and state[i][N_Comp]>0 and sum(state[i][N_Comp+1:N_Comp+1+N_Unit])==0:
                feasible[i]=1
                
        #search closed streams, which are no recycles themself
        for i in range(len(state)):
            if state[i][-1]==1 and state[i][N_Comp]>0 and sum(state[i][N_Comp+1:N_Comp+1+N_Unit])==1 and state[i][N_Comp+1+N_Unit-1]==0:
                feasible[i]=1
                
        feasible[z1]=0
        
    return feasible

#generate the flowsheet matrix
def statetrafo(done,fs):
    state=np.zeros((N_Matrix,linelength))
    for i in range(len(fs.lines)):
        state[i]=fs.giveinf(i)
    
        state[i][N_Comp]=sum(state[i][0:N_Comp])
        state[i][0:N_Comp]=molfr(state[i][0:N_Comp])
            
    for i in range(len(state)):
        state[i][-2]=done
        
    return state
    
#input mol, return kj
def qduty(molar_flowrates):
    #kj/mol
    h_v=np.array([32.995,20.281,20.999,42.563]) 
    
    qtot=0
    for i in range(4):
        qtot=qtot+(h_v[i]*molar_flowrates[i])
        
    return 2*qtot

#return €
def opcost(molar_flowrates):
    cost=8000*0.04*qduty(molar_flowrates)/2257

    return cost
    
#return €
def npv(state):
    r=0
    #euro/kg
    prices=np.array([1.27,0.5,0.5,0.75]) 
    molarmassarray=np.array([102.18,56.106,58.124,46.069])
    
    for i in range(N_Matrix):
        if state[i][-1]==1:
            #values in matrix are in 100kmol
            molar_flowrates=state[i][0:N_Comp]*state[i][N_Comp]*100*1000
            masses=np.zeros(N_Comp)
        
            for j in range(N_Comp):
                masses[j]=molar_flowrates[j]*molarmassarray[j]
                
            #masses in kg
            masses=masses/1000 
    
            #open streams
            if sum(state[i][N_Comp+1:N_Comp+1+N_Unit])==0:
                if sum(masses)>0:
                    massfracs=masses/sum(masses)
                    if np.max(massfracs)>0.99:
                        r=r+(8000*np.max(masses)*prices[np.argmax(masses)])
                
                    else:
                        r=r+(8000*0.5*sum(np.multiply(prices,masses)))
                        
            #r
            if state[i][N_Comp+1]==1:
                basis=64000 #9000 kg/hr
                r=r-(basis*np.power(sum(masses)/9000,0.6))
                
            #d_L,d_H
            if sum(state[i][N_Comp+2:N_Comp+2+2])==1:
                basis=594000 #9000 kg/hr
                r=r-(basis*np.power(sum(masses)/9000,0.6))
                
                #search for line in matrix containing distillate
                for j in range(N_Matrix):
                    if state[i][N_Comp+1+N_Unit+j]==1:
                        break
                    
                distmolar_flowrates=state[j][0:N_Comp]*state[j][N_Comp]*100*1000
                r=r-opcost(distmolar_flowrates)

    return r

#class for stream (this is not necessarily a line in the flowsheet matrix, e.g. recycle)
class stream:
    def __init__(self,inp,nr,start,end,rec=None):
        self.nr=nr
        self.inp=inp
        self.start=start
        self.end=end
        self.rec=rec

#class for line in flowsheet matrix
class line:
    def __init__(self,streamlist,linenr):
        self.streamsin=streamlist
        self.linenr=linenr
        self.app=None
        self.streamsout=[]
                
#class for units and recycles
class apparatus:
    def __init__(self,app,appspec):
        self.app=app
        self.appspec=appspec
        
    def compute_output(self,input_moles):
        #ensure that no negative values are processed (could occur during the use of fsolve)
        inp=above0(copy.deepcopy(np.asarray(input_moles)))
        if sum(inp)>0:
            output=[np.zeros(N_Comp)]
            if self.app==0:#r
                output=[units.reactor(inp)]
                
            if self.app==1:#pure light boiler
                output=units.distillation(inp,1)
                
            if self.app==2:#pure heavy boiler
                output=units.distillation(inp,0)
 
            if self.app==3:#mix/rec
                output=[inp]
            
            for i in output:
                i=above0(i)
                
        else:
            output=[np.zeros(N_Comp)]
            if self.app==1 or self.app==2:
                output.append(np.zeros(N_Comp))
                
        return output

#contains tree structure for one flowsheet
#within the tree, loops for recycles can be constructed
class flowsheet:
    def __init__(self,feeds):
        self.roots=feeds
        self.lines=[]   #list of lines in the flowsheet matrix
        self.streams=[] #multiple streams connect the apparatus and lines
        self.recstreamsnrs=[]   

        for i in range(len(self.roots)):
            self.streams.append(stream(self.roots[i],i,None,i,np.ones(N_Comp)))
            self.lines.append(line([i],i))
            
    #collects all streams which go into a line
    def calcinstream(self,nr):
        out=np.zeros(N_Comp)
        for i in self.lines[nr].streamsin:
            out=out+self.streams[i].inp
        
        return out
    
    #add new apparatus
    def newapp(self,nr,app,appspec):
        conv=True
        self.lines[nr].app=apparatus(app,appspec)

        newlines=self.lines[nr].app.compute_output(self.calcinstream(nr))
        #no rec
        if app!=N_Unit-1:
            for i in range(len(newlines)):
                self.streams.append(stream(newlines[i],len(self.streams),nr,len(self.lines)))
                self.lines[nr].streamsout.append(len(self.streams)-1)
                self.lines.append(line([len(self.streams)-1],len(self.lines)))
                    
        #rec/mix
        else:
            self.streams.append(stream(newlines[0],len(self.streams),nr,appspec,len(self.recstreamsnrs)))
            self.lines[nr].streamsout.append(len(self.streams)-1)
            self.recstreamsnrs.append(len(self.streams)-1)
            self.lines[appspec].streamsin.append(len(self.streams)-1)
            #convergence criteria for root finding problem
            epsilon=0.001
            
            #initial guess for solver
            guess=np.zeros(len(self.recstreamsnrs)*N_Comp)
            for i in range(len(self.recstreamsnrs)):
                guess[i*N_Comp:(i+1)*N_Comp]=self.streams[self.recstreamsnrs[i]].inp

            sol_cand=above0(opt.fsolve(self.rec,guess,full_output=True)[0])
            
            #test if proposed solution is close enough to zero
            if sum(np.abs(self.rec(sol_cand)))<epsilon*len(self.recstreamsnrs)*N_Comp:
                #just redo the loop, to set all streams in the flowsheet to the correct flowrates
                self.rec(sol_cand)
                
            else:
                conv=False
                
        return conv
    
    #function for rec loops
    def rec(self,guess):
        sol=np.zeros(N_Comp*len(self.recstreamsnrs))
        guess=above0(guess)

        #set all streams except feeds to None
        for i in range(len(self.roots),len(self.streams)):
            if self.streams[i].rec==None:
                self.streams[i].inp=None
                
        #set guesses for tears
        for i in range(len(self.recstreamsnrs)):
            self.streams[self.recstreamsnrs[i]].inp=guess[i*N_Comp:(i+1)*N_Comp]
            sol[i*N_Comp:(i+1)*N_Comp]=guess[i*N_Comp:(i+1)*N_Comp]
            
        for nr in range(len(self.lines)):
            if self.lines[nr].app!=None:
                newout=self.lines[nr].app.compute_output(self.calcinstream(nr))
                if self.lines[nr].app.app!=N_Unit-1:
                    for i in range(len(newout)):
                        self.streams[self.lines[nr].streamsout[i]].inp=newout[i]
                        
                else:
                    sol[self.streams[self.lines[nr].streamsout[0]].rec*N_Comp:(self.streams[self.lines[nr].streamsout[0]].rec+1)*N_Comp]=sol[self.streams[self.lines[nr].streamsout[0]].rec*N_Comp:(self.streams[self.lines[nr].streamsout[0]].rec+1)*N_Comp]-newout[0]
                
        return sol
    
    #give information on certain line of flowsheet
    def giveinf(self,nr):
        ret=np.zeros(linelength)
        ret[-1]=1
        for i in self.lines[nr].streamsin:
            ret[0:N_Comp]=ret[0:N_Comp]+self.streams[i].inp
        
        if self.lines[nr].app!=None:
            ret[N_Comp+1+self.lines[nr].app.app]=1
            for i in self.lines[nr].streamsout:
                ret[N_Comp+1+N_Unit+self.streams[i].end]=1
                      
        return ret

#class for simulation
class env:
    def __init__(self,feed,player):
        self.done=False
        self.conv=True
        self.blueprint=[feed]
        self.player=player
        self.fc=flowsheet(feed)
        self.state=statetrafo(self.done,self.fc)

    def anbauen(self,stream,app,appspec):
        if stream==N_Matrix:
            self.blueprint.append(np.asarray([stream]))
            self.done=True

        else:
            self.conv=self.fc.newapp(stream,app,appspec)
            self.blueprint.append(np.asarray([stream,app,appspec]))
            
        self.state=statetrafo(self.done,self.fc)
        if self.done:
            rew=npv(self.state)
            
        else:
            rew=None
             
        return self.state,rew,self.done,self.conv




