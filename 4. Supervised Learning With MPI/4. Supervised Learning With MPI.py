#CUP DATASET
#################################
from mpi4py import MPI
import mpi4py
import numpy as np
import pandas as pd
import random 
import matplotlib.pyplot as plt

  

#################################################################################################
def train_test(dataset,ratio_of_train_samples):
    total_number_of_samples = dataset.shape[0]
    number_of_train_samples = int(total_number_of_samples * ratio_of_train_samples)
    
    train_set = []
    test_set =[] 
    
    #choosing random numbers for train sets
    indexes_of_training_samples = random.sample(range(0,total_number_of_samples),number_of_train_samples)
    
    for i in range(0,total_number_of_samples):
        sample =dataset.iloc[i,].values
        if i in indexes_of_training_samples :
            train_set.append(sample)
        else:
            test_set.append(sample)
    
    return (pd.DataFrame(train_set),pd.DataFrame(test_set) )                 

#################################################################################################
    

def LR_predict(X,beta):
    return np.matmul(X,beta)



#################################################################################################
def rmse(x,y,beta):
    rows = x.shape[0]
    prediction = np.matmul(x,beta)
    total = 0
    for i in range(0,rows):
        total = total + ((y[i]-prediction[i])**2)
    return np.sqrt(total)



#################################################################################################

#here f the feature(column) which wanted to be predicted
#here f the feature(column) which wanted to be predicted
def sgd(dataset,beta,mu, iterations,f):
    x =(dataset.values)[:,:]
    y =(dataset.values)[:,f] 
    
    beta_old = beta
    loss =[]
    

    number_of_epoch  =  x.shape[0]
    
    indexes_of_epoch = random.sample(range(0,number_of_epoch),number_of_epoch)
    
    for j in range(0,iterations):
        for i in indexes_of_epoch:
            beta_new = beta_old -mu * np.matmul(x[i],beta_old)
            new_loss = rmse(x,y,beta_new)
            loss.append(new_loss)
            beta_old=beta_new

 
    return beta,loss

#################################################################################################


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
name = MPI. Get_processor_name()


################################################################################################




n=size

if rank == 1:
    cup_dataset_non_numerical= pd.read_csv('cup98VAL.txt', low_memory = False)
    
    #Converting the all non-numerical values to numerical values
    cup_dataset=pd.get_dummies(cup_dataset_non_numerical, dummy_na=True)

    #converting nan values to 0 values
    cup_dataset=cup_dataset.fillna(0).iloc[:10,]
    
    #splitting the data 
    shared_data=[]

    for item in np.split(cup_dataset, n):
        splitted_train_test = train_test(item,0.7)
        shared_data.append(splitted_train_test)
        

else:
    shared_data = None
    
    
received = comm.scatter(shared_data, root=1)


############################################################################################



#8. OPERATION
start = MPI.Wtime() #starting time
for i in range(0,size):
    if rank==i:
        (train,test)=received
        local_beta= sgd(train,np.ones((train.shape[1],1)),0.0000001,2,0)[0]
        
        
        
                            
end = MPI.Wtime() #ending time
time= end-start #for calculationg the timing



#############################################################################################


if rank!=None:
    Gathered_one = comm.gather(local_beta,root=1)
    Gathered_two=comm.gather(time,root=1) 
    
    global_beta=Gathered_one[0]
    for i in range(1,n):
        global_beta=global_beta + i
    global_beta =1/n*(global_beta)
        
        

    