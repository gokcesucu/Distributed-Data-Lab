

from mpi4py import MPI
import numpy as np



def n_dim_matrix(n):
    A= np.random.randint(0,10,size=(10**n,10**n))
    return A 


def pieces_of_matrix_columns(A,process):
    list_of_pieces=list(range(process))
    matrix_size= int(A.shape[0])
    amount_of_column_piece= int(matrix_size/process)
    for i in range(0,process-1):
        list_of_pieces[i]=A[:,i*amount_of_column_piece:(i+1)*amount_of_column_piece]
    list_of_pieces[process-1]=A[:,(process-1)*amount_of_column_piece:] 
    return list_of_pieces

def concat_gather(gathered):
    length = len(gathered)
    solution= gathered[0]
    for i in range(1,length):
        solution = np.concatenate((solution,gathered[i]),axis=1)
    return solution

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
name = MPI. Get_processor_name()

N=4


A_matrix  = n_dim_matrix(N)
B_matrix  = n_dim_matrix(N)

#i splitted the matrix a by columns so there are still 10**N columns.
set_of_columns_B=pieces_of_matrix_columns(B_matrix,size)



data = []
for i in range(0,size):
    data.append([A_matrix,set_of_columns_B[i]])



if rank == 1:
    shared_data=data
         
else:
    shared_data = None
    
    
received = comm.scatter(shared_data, root=1)




for i in range(0,size):
    if rank==i:
        start = MPI.Wtime()
        x = np.matmul(received[0] ,received[1])
        end = MPI.Wtime()
        y = end-start
        
Gathered_one = comm.gather(x,root=1)
Gathered_two=comm.gather(y,root=1)

print("Gathered List",Gathered_one)
print("Solution",concat_gather(Gathered_one))
print("Timing",sum(Gathered_two),"\n")
print("\n")
    