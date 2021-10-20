
import numpy as np
import copy

epsilon=0.001

#etbe,ib,nbut,ethanol
def molefracs(array):
    if sum(array)>0:  
        return array/sum(array)
    
    else:
        return np.zeros(len(array))

#count present components
def comp_counter(array):
    count=0
    for i in range(len(array)):
        if array[i]/sum(array)>epsilon:
            count=count+1
            
    return count

#line is a list, which contains first a point and then a vector
#plane is a list, which contains first a point and then the normal vector
def intersection(line,plane,pointtogive):
    if np.abs(np.inner(line[1],plane[1]))<epsilon or np.abs(np.inner((plane[0]-line[0]),plane[1]))<epsilon:
        point=pointtogive
    
    else:
        d=np.inner((plane[0]-line[0]),plane[1])/np.inner(line[1],plane[1])
        point=line[0]+(d*line[1])
        
    return point

def projection_quat_r3(molefractions):
    x=(molefractions[1]+1-molefractions[0])/2
    y=((np.sqrt(3)/2)*molefractions[2])+((np.sqrt(3)/6)*molefractions[3])
    z=(np.sqrt(6)/3)*molefractions[3]
    
    return np.array([x,y,z])
    
def projection_r3_quat(xyz):
    molefractions=np.zeros(4)
    molefractions[3]=(3/np.sqrt(6))*xyz[2]
    molefractions[2]=((2/np.sqrt(3))*xyz[1])-(molefractions[3]/3)
    molefractions[1]=((2*xyz[0])-molefractions[2]-molefractions[3])/2
    molefractions[0]=1-sum(molefractions)
    
    return molefractions

def distance(p1,p2):
    return np.sqrt(sum(np.square(p1-p2)))

def reactor(molar_flowrates):
    ccount=comp_counter(molar_flowrates)
    tocalc=True
    
    #in some cases there is no reaction
    if ccount<=1 and molar_flowrates[0]/sum(molar_flowrates)<epsilon:
        tocalc=False
        
    if ccount==2 and molar_flowrates[2]/sum(molar_flowrates)>epsilon and molar_flowrates[0]/sum(molar_flowrates)<epsilon:
        tocalc=False
    
    if tocalc:
        K=111.1
        
        n0ges=sum(molar_flowrates)
        n0etoh=molar_flowrates[3]
        n0ib=molar_flowrates[1]
        n0etbe=molar_flowrates[0]
        
        coeff=np.zeros(3)
        coeff[0]=1+K
        coeff[1]=(-1*K*n0etoh)+(-1*K*n0ib)+(-1*n0ges)+n0etbe
        coeff[2]=(K*n0ib*n0etoh)-(n0etbe*n0ges)
        
        candidate_sol=np.roots(coeff)
        
        i=0
        conv=False
        
        #check candidate solutions so that conversion does not exceed the available components
        while not conv:
            if (molar_flowrates[0]+candidate_sol[i]<-epsilon*np.max(np.abs(molar_flowrates)) 
                or molar_flowrates[1]-candidate_sol[i]<-epsilon*np.max(np.abs(molar_flowrates)) 
                or molar_flowrates[3]-candidate_sol[i]<-epsilon*np.max(np.abs(molar_flowrates))):
                i=i+1
                
            else:
                conv=True
                break
            
        if conv:
            solution=candidate_sol[i]
        
            new_molarflowrates=np.zeros(4)
            new_molarflowrates[0]=molar_flowrates[0]+solution
            new_molarflowrates[1]=molar_flowrates[1]-solution
            new_molarflowrates[2]=molar_flowrates[2]
            new_molarflowrates[3]=molar_flowrates[3]-solution
        
    else:
        new_molarflowrates=molar_flowrates
        
    return new_molarflowrates

#separation 1 for light boiler, separation 0 for heavy boiler
def distillation(molar_flowrates,separation):
    #azeotropic data, singular points
    azeo_etbe_ethanol=np.array([0.37,0,0,0.63])
    azeo_nbut_ethanol=np.array([0,0,0.96,0.04])
    singular_points=[np.array([0,0,0]),np.array([1,0,0]),np.array([0.5,np.sqrt(3)/2,0]),np.array([0.5,np.sqrt(3)/6,np.sqrt(6)/3]),projection_quat_r3(azeo_etbe_ethanol),projection_quat_r3(azeo_nbut_ethanol)]
   
    #etbe,ib,nbut,ethanol,azeo etbe etha,azeo etha nbut
    boilingpoints=np.array([156.93,61.44,69.45,142.23,133.42,68.86])
    
    point=projection_quat_r3(molefracs(molar_flowrates))
    
    #defining points for separatrix
    #etbe,ib,nbut,ethanol
    p1=projection_quat_r3(azeo_etbe_ethanol)
    p2=projection_quat_r3(azeo_nbut_ethanol)
    p3=projection_quat_r3(np.array([0,1,0,0]))
    
    #parametric form
    v1=p2-p1
    v2=p3-p1
    
    #normal vector points "down"
    separatrix=[p1,np.cross(v1,v2)]
    
    outsing=False
    
    #if feed is to close to singular point, separation is neglected
    for i in range(len(singular_points)):
        if np.sqrt(np.sum(np.square(point-singular_points[i])))<epsilon:
            outsing=True
            break
        
    if comp_counter(molar_flowrates)<=1 or outsing:
        if separation==0:
            ret=[np.zeros(4),molar_flowrates]
            
        else:
            ret=[molar_flowrates,np.zeros(4)]
    
    else:
        if distance(point,intersection([point,separatrix[1]], separatrix, point))<epsilon and separation==0:
            ret=[np.zeros(4),molar_flowrates]
            
        else:
            #check for the relevant boiling points and therefore the products of the distillation
            relevantdata=np.zeros(6)
            for i in range(4):
                if molar_flowrates[i]/sum(molar_flowrates)>epsilon:
                    relevantdata[i]=1
                    
            if relevantdata[0]==1 and relevantdata[3]==1:
                relevantdata[4]=1 #azeo etbe etha
                
            if relevantdata[2]==1 and relevantdata[3]==1:
                relevantdata[5]=1 # azeo etha nbut
                
            pointfound=False
            boilingpoints2=copy.deepcopy(boilingpoints)
            while not pointfound:
                if separation==0:
                    for i in range(6):
                        if relevantdata[i]==0:
                            boilingpoints2[i]=-1*float('inf')
                    
                    indexwewant=np.argmax(boilingpoints2)
                    
                else:
                    for i in range(6):
                        if relevantdata[i]==0:
                            boilingpoints2[i]=float('inf')
                    
                    indexwewant=np.argmin(boilingpoints2)
                
                if indexwewant==1 or indexwewant>3 or np.sign(np.inner(p1-point,separatrix[1]))==np.sign(np.inner(p1-singular_points[indexwewant],separatrix[1])):
                    pointfound=True
                    break
                    
                else:
                    relevantdata[indexwewant]=0
                    
            #no ethanol means no azeotropes and just one distillation region
            if molar_flowrates[3]/sum(molar_flowrates)<=epsilon:
                pure_product=np.zeros(4)
                pure_product[indexwewant]=molar_flowrates[indexwewant]
                
            else:            
                pure=singular_points[indexwewant]
                
                #all relevant planes collected, to find a intersection
                all_planes=[]
                all_planes.append([singular_points[1],np.cross(singular_points[2]-singular_points[1],singular_points[3]-singular_points[1])])
                all_planes.append([singular_points[0],np.cross(singular_points[2]-singular_points[0],singular_points[3]-singular_points[0])])
                all_planes.append([singular_points[3],np.cross(singular_points[0]-singular_points[3],singular_points[1]-singular_points[3])])
                all_planes.append([singular_points[2],np.cross(singular_points[0]-singular_points[2],singular_points[1]-singular_points[2])])
                all_planes.append(separatrix)
                
                #corresponding to singular points, which of the above planes are not connected and therefore relevant for the intersection
                not_connected_to=[[0,4],[1],[2,4],[3,4],[1,2],[0,1]]
        
                lineforlater=[point,pure-point]
        
                candidate_list=[]
                candidate_list2=[]
                for i in range(len(not_connected_to[indexwewant])):
                    candidate_list.append(intersection(lineforlater,all_planes[not_connected_to[indexwewant][i]],point))
            
                for i in candidate_list:
                    candidate_list2.append(distance(point, i))
                        
                other=candidate_list[np.argmin(candidate_list2)]
    
                if np.sqrt(np.sum(np.square(other-pure)))<epsilon:
                    n_pure=sum(molar_flowrates)
                
                else:
                    n_pure=(np.sqrt(np.sum(np.square(point-other)))/np.sqrt(np.sum(np.square(other-pure))))*sum(molar_flowrates)
                
                pure_product=n_pure*projection_r3_quat(pure)
                for i in range(4):
                    if molar_flowrates[i]-pure_product[i]<0:
                        pure_product[i]=molar_flowrates[i]
                        
            other_product=molar_flowrates-pure_product
            
            if separation==0:
                ret=[other_product,pure_product]
                
            else:
                ret=[pure_product,other_product]

    return ret
