#EXERCISE 1:CLEANING AND TOKENIZING


#################################
#1. DATA READING

#libraries
import os 


titles_of_20_folders = os.listdir("/Users/gokcesucu/hw3/20_newsgroups")

#removing the DS_Store Folder
titles_of_20_folders.remove('.DS_Store')


#list of th titles of each folders
titles_of_files_in_folders = []
for i in titles_of_20_folders:
    path = "/Users/gokcesucu/hw3/20_newsgroups/"+i
    titles= os.listdir(path)
    titles_of_files_in_folders.append(titles)



##############################################################################################
#2. CREARING RAW DATA OF EACH CORPUS
raw_corpuses = []
for i in range(0,20):
    name_of_the_folder = titles_of_20_folders[i]
    for j in titles_of_files_in_folders[i]:
        name_of_the_file = j
        path = "/Users/gokcesucu/hw3/20_newsgroups/" + name_of_the_folder +"/"+name_of_the_file
        text = open(path, encoding = 'latin1')
        raw_corpuses.append(text.read())  
#there are 19997 corpuses




#################################################################################################
#3. DATA CLEANING

#libraries
import nltk
from nltk.corpus import stopwords
import string


#for downloading the stop words in english
nltk.download('stopwords')

#list of english stop words
stop_words = stopwords.words('english')

#list of punctuations
punctuations = list(string.punctuation)

#list of numbers
numbers = list(range(10))



#corpus cleaning function
def data_cleaning_one(corpus):
    splitted_data = corpus.split()
    new_list = []
    for i in splitted_data:
        
        i = i.lower()
        
        #removing punctuations
        for x in punctuations:
            i=i.replace(x," ")
    
        #removing numbers
        for y in numbers:
            i = i.replace(str(y)," ")
        new_list= new_list+i.split()
        
        #removing stop words
        for j in stop_words:
            if j in new_list:
                new_list.remove(j)
    return new_list




#########################
def cleaned_data_of_a_list(liste):
    cleaned = []
    for i in liste:
        cleaned.append(data_cleaning_one(i))
    return cleaned
        
        

#################################################################################################
#4. CALLING THE COMMUNICATOR
from mpi4py import MPI


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
name = MPI. Get_processor_name()


################################################################################################

#5 CHUNKING THE DATA
n= size

def split_data(raw_corpuses,n):
    splitted_corpuses= []
    number_of_corpuses = len(raw_corpuses) 
    number_of_corpuses_in_piece= int(number_of_corpuses /n)
    
    for z in range(0,n-1):
        chunk = raw_corpuses[z*number_of_corpuses_in_piece:(z+1)*number_of_corpuses_in_piece]
        splitted_corpuses.append(chunk)
    
    splitted_corpuses.append( raw_corpuses[(n-1)*number_of_corpuses_in_piece: ]  )
    
    return splitted_corpuses





########################################################################################
#6. SPLITTING

if rank == 1:
    shared_data=split_data(raw_corpuses,n)
         
else:
    shared_data = None
    
    
received = comm.scatter(shared_data, root=1)


############################################################################################


#7. OPERATION
start = MPI.Wtime()
for i in range(0,size):
    if rank==i:
        x = cleaned_data_of_a_list(received)
        print(x[0])
        
end = MPI.Wtime()
y = end-start



#############################################################################################

#8. GATHERING

Gathered_one = comm.gather(x,root=1)
Gathered_two=comm.gather(y,root=1)    

print(Gathered_one)
print(Gathered_two)
print(max(Gathered_two))


