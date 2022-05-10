
from mpi4py import MPI
import numpy as np


def pieces_of_vector(v,process):
    list_of_pieces=list(range(process))
    vector_size= int(v.shape[0])
    numbers_in_pieces= int(vector_size/process)
    for i in range(0,process-1):
        list_of_pieces[i]=v[i*numbers_in_pieces:(i+1)*numbers_in_pieces,:]
    list_of_pieces[process-1]=v[(process-1)*numbers_in_pieces:,:] 
    return list_of_pieces

def pieces_of_matrix(A,process):
    list_of_pieces=list(range(process))
    vector_size= int(A.shape[0])
    numbers_in_pieces= int(vector_size/process)
    for i in range(0,process-1):
        list_of_pieces[i]=A[i*numbers_in_pieces:(i+1)*numbers_in_pieces,:]
    list_of_pieces[process-1]=A[(process-1)*numbers_in_pieces:,:] 
    return list_of_pieces

def concat_gathered(gathered_list):
    length  = len(gathered_list)
    concated_last= gathered_list[0]
    for i in range(1,length):
        concated_last = np.concatenate((concated_last,gathered_list[i]))
    return concated_last
    
def n_dim_matrix(n):
    A= np.random.randint(0,10,size=(10**n,10**n))
    return A 
def n_dim_vector(n):
    v = np.random.randint(0,10,size=(10**n,1))
    return v




comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
name = MPI. Get_processor_name()

N=4

v_one= (n_dim_vector(N)).reshape(10**N,1)

A_matrix  = n_dim_matrix(N)

#i splitted the matrix a by rows so there are still 10**N columns.
pieces_of_A=pieces_of_matrix(A_matrix,size)



data = []
for i in range(0,size):
    data.append(np.concatenate((pieces_of_A[i],v_one.T),axis=0))



if rank == 1:
    shared_data=data
         
else:
    shared_data = None
    
    
received = comm.scatter(shared_data, root=1)




for i in range(0,size):
    if rank==i:
        start = MPI.Wtime()
        x = np.matmul(received[:-1,:] ,received[-1,:].T)
        end = MPI.Wtime()
        y = end-start
        
Gathered_one = comm.gather(x,root=1)
Gathered_two=comm.gather(y,root=1)

print("Gathered List",Gathered_one)
print("Result of Multiplication",concat_gathered(Gathered_one))
print("Timing",sum(Gathered_two),"\n")
print("\n")
    