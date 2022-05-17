#EXERCISE 3:IDF SCORE


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
for i in range(0,2):
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
import math # necessary for log calculations

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





        
###########################################################################################
# CREATING A WHOLE WORDS THAT MENTONED IN THE CORPUSES
whole_tokens = []
for a in raw_corpuses:
    cleaned_tokened = data_cleaning_one(a)
    whole_tokens = whole_tokens + cleaned_tokened

whole_list_of_tokens= list(set(whole_tokens))

 ########################################################################################### 
#CREATING IDE FUNCTION
def inverse_document_freq(token,raw_corpuses):
    number_of_token_existed_corpus = 0
    number_of_totel_corpuses = len(raw_corpuses)
    for i in raw_corpuses:
        cleaned_tokened = data_cleaning_one(i)
        if token in cleaned_tokened:
            number_of_token_existed_corpus =number_of_token_existed_corpus+1
    ide = math.log(number_of_totel_corpuses/number_of_token_existed_corpus) 
    return (token,ide)
                  

#################################################################################################

#5. CALLING THE COMMUNICATOR
from mpi4py import MPI


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
name = MPI. Get_processor_name()


################################################################################################

#6 CHUNKING THE DATA
n= size

def split_tokens_list(whole_tokens_set,n):
    splitted_corpuses= []
    number_of_corpuses = len(whole_tokens_set) 
    number_of_corpuses_in_piece= int(number_of_corpuses /n)
    
    for z in range(0,n-1):
        chunk = whole_tokens_set[z*number_of_corpuses_in_piece:(z+1)*number_of_corpuses_in_piece]
        splitted_corpuses.append(chunk)
    
    splitted_corpuses.append( whole_tokens_set[(n-1)*number_of_corpuses_in_piece: ]  )
    
    return splitted_corpuses





########################################################################################
#7. SPLITTING

if rank == 1:
    shared_data=split_tokens_list(whole_list_of_tokens, n)
         
else:
    shared_data = None
    
    
received = comm.scatter(shared_data, root=1)


############################################################################################


#8. OPERATION
start = MPI.Wtime() #starting time
for i in range(0,size):
    if rank==i:
        x = []
        for i in received: # received is a list of tokens
            x.append(inverse_document_freq(i, raw_corpuses))
end = MPI.Wtime() #ending time
y = end-start #for calculationg the time




#############################################################################################

#9. GATHERING

Gathered_one = comm.gather(x,root=1)
Gathered_two=comm.gather(y,root=1)    

print("IDF= ",Gathered_one)
print("Time for Each Process=",Gathered_two)
print("Timing",max(Gathered_two))
    


