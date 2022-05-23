#HOMEWORK 3 
#Gökce sucu



######################################################
#0. DATA READING

import numpy as np
import random
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

newsgroups_train = fetch_20newsgroups()







#########################################
# 1. VECTOR REPRESENTATION OF TEXTS 

title_of_the_twenty_news = list(newsgroups_train.target_names)
newsgroups_train = fetch_20newsgroups(subset='train',categories=title_of_the_twenty_news)
vectorizer = TfidfVectorizer()
vectors_of_the_texts = vectorizer.fit_transform(newsgroups_train.data)


#creating list of all text vectors
list_of_vectors= []
for i in vectors_of_the_texts:
    vector = i.toarray() #for matrix representation 
    list_of_vectors.append(vector)
    






#################################################################################################
#2 EUCLID DISTANCE

def euclid(x_one,x_two):  
    dimension = int(x_one.shape[1])

    distance = 0
    
    for i in range(0,dimension):
            distance = distance + (x_one[0][i]-x_two[0][i])**2
    
    return  np.sqrt(distance)   








###########################################################################################
#3 K- MEANS INITIAL CENTERS

def k_means_initial_centers(dataset,k):
    
    number_of_samples = len(dataset)
    
    #to initalize first center of the data set
    N= random.randint(0,number_of_samples-1)
    
    indexes_of_centers = [N]
    #our first center of the first cluster (we choose randomly)
    first_center = dataset[N]
    
    
    initial_centers = [first_center]
    
    
    #last (k-1) elements would b our first iteration centers 
    K=1
    
    while K<k:
        maximum_distance = 0
        
        for n in range(0,number_of_samples):
            if n not in indexes_of_centers:
                distance_of_a_sample_to_all_centers= []
                for i in range(0,K):
                    distance= euclid(dataset[n],initial_centers[i] )
                    distance_of_a_sample_to_all_centers.append(distance)
                
                total_distance_to_all_centers = sum(distance_of_a_sample_to_all_centers)
                
                if total_distance_to_all_centers> maximum_distance:
                    maximum_distance = total_distance_to_all_centers
                    maximum_index = n
           
        initial_centers.append(dataset[maximum_index])
        indexes_of_centers.append(maximum_index)
        K = K+1
        
    return initial_centers







#########################################################################################
#4. K-MEANS ALGORTIHM

def k_means(dataset,k,initial_centers,iterations):
    number_of_sample = len(dataset)
    centers = initial_centers
    
    partitions = []
    for x in range(0,k):
        partitions.append([])
        
        
    for i in range(0,iterations):
        for n in range(0,number_of_sample):
            min_distance = euclid(dataset[n],centers[0])
            partition_class = 0            
            
            for K in range(1,k):
                distance = euclid(dataset[n],centers[K])
                if distance<min_distance:
                    min_distance = distance
                    partition_class = K
            (partitions[partition_class]).append(dataset[n])
        
        for index in range(0,k):
            all_samples_in_partition = len(partitions[index])
            total = sum(partitions[index])
            centers[index] = (1/all_samples_in_partition)*total

    return centers 









########################################################################################
#5 TEXTS SPLITTING FUNCTION

def split_texts_into_n_piece(dataset,n):
    splitted_texts= []
    number_of_texts = len(dataset) 
    number_of_texts_in_piece= int(number_of_texts /n)
    
    for z in range(0,n-1):
        chunk = dataset[z*number_of_texts_in_piece:(z+1)*number_of_texts_in_piece]
        splitted_texts.append(chunk)
    
    splitted_texts.append( dataset[(n-1)*number_of_texts_in_piece: ]  )
    
    return splitted_texts


#########################################################################################
#6. HYPERPARAMETERS
k = 10
 
iteration = 10






##########################################################################################
# 6. INITIALIZING THE CENTERS

initial_centers = k_means_initial_centers(list_of_vectors,k)



######################################################################################
#5. CALLINGTHE COMMUNICATORS

from mpi4py import MPI


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
name = MPI. Get_processor_name()


################################################################################################

n=size







########################################################################################
#6. SCATTERING THE DATA

if rank == 1:
    shared_data=split_texts_into_n_piece(list_of_vectors, n)
         
else:
    shared_data = None
    
    
received = comm.scatter(shared_data, root=1)










############################################################################################
#7. OPERATION
start = MPI.Wtime() #starting time
for i in range(0,size):
    if rank==i:
        x=k_means(received,k,initial_centers,iteration) 
end = MPI.Wtime() #ending time
y = end-start #for calculationg the time








#############################################################################################

#9. GATHERING

local_centers = comm.gather(x,root=1)
time =comm.gather(y,root=1)    



#finging the global centers values
if local_centers!=None:
    if time !=None:
        total  = np.array(local_centers[0])
        for i in range(1,k):
            total=total + np.array(local_centers[i])
        total = total/k
        global_centers = total/k
        
        print(global_centers)
        print(time)
        
       


