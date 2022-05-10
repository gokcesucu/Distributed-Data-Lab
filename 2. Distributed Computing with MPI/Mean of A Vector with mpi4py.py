# EXERCISE 1b) MEAN

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


def n_dim_vector(n):
    v = np.random.randint(0,10,size=(10**n,1))
    return v

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
name = MPI. Get_processor_name()

N=8

v_one= (n_dim_vector(N)).reshape(10**N,1)
data=v_one

if rank == 1:
    shared_data=pieces_of_vector(data,size)
         
else:
    shared_data = None
    
    
received = comm.scatter(shared_data, root=1)





for i in range(0,size):
    if rank==i:
        start = MPI.Wtime()
        x = np.mean(received)
        end = MPI.Wtime()
        y = end-start
        
Gathered_one = comm.gather(x,root=1)
Gathered_two=comm.gather(y,root=1)

print("Mean",np.mean(Gathered_one))
print("Timing",sum(Gathered_two),"\n")
print("\n")
    