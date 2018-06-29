""" Spam filtering using 
hidden markob model"""
import re
import string
from decimal import Decimal
import numpy as np
import pandas as pd
# use natural language toolkit
import nltk
#import stop word
from nltk.corpus import stopwords
#stemmer library import
from nltk.stem.lancaster import LancasterStemmer
stop_words = set(stopwords.words('english'))
# word stemmer
stemmer = LancasterStemmer()

#encoding the csv file
import chardet
with open('data_for_spam.csv', 'rb') as f:
    result = chardet.detect(f.read())  # or readline if the file is large
dataset=pd.read_csv('data_for_spam.csv', encoding=result['encoding'])
#splitting thr csv file into A and y
x=dataset.iloc[:,0]
y=dataset.iloc[:,1]
X=x.to_dict()

#train test split 
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)

#index acending
s=y_train.reset_index()
y_train=s.iloc[:,1]

#class wise data entry in training_data
training_data = []
for d in range(len(X_train)):
    #spam class training_data
    if y_train[d]=="spam":
        training_data.append({ "class":"spam","sentence":X_train[d]})
    #ham class training_data 
    else:
        training_data.append({ "class":"ham","sentence":X_train[d]})
#length of thr trainig_data  
print ("%s sentences of training data" % len(training_data))

#class dictinery
class_words = {}

# turn a list into a set (of unique items) and then a list again (this removes duplicates)
classes = list(set([a['class'] for a in training_data]))
for c in classes:
    # prepare a list of words within each class
    class_words[c] = []
    
#spam and ham word list creat
sp_word=[]
hm_word=[]
cr_words=[]    
for data in training_data:  
    sen=data['sentence']
    #temporarry list for appending to word list
    cr_words=[]
    #removing digit and punctuation
    sentence= re.sub(r'\d+','', sen)
    sentence= re.sub('['+string.punctuation+']', '', sentence) 
    #spam word lists
    if data['class']=='spam':
        for word in nltk.word_tokenize(sentence):
            if word not in stop_words:
                stemmed_word = stemmer.stem(word.lower())
                cr_words.append(stemmed_word)
        #sentence word count   
        from collections import Counter
        count=Counter(cr_words)
        #decending by value
        w=[(l,k) for k,l in sorted([(j,i) for i,j in count.items()], reverse=True)]
        w=np.array(w)
        #just word list taken in a decinding way
        B=w[:,0]
        #forming a spam word list 
        sp_word.append(B)
    else: 
        #spam word lists
        for word in nltk.word_tokenize(sentence):
            if word not in stop_words:
                stemmed_word = stemmer.stem(word.lower())
                cr_words.append(stemmed_word)
            
        #sentence word count   
        from collections import Counter
        count=Counter(cr_words)
        w=[(l,k) for k,l in sorted([(j,i) for i,j in count.items()], reverse=True)]
        w=np.array(w)
        #just word list taken in a decinding wa
        B=w[:,0]
        #forming a spam word list 
        hm_word.append(B)

#forming a ham matrix
for i in range(1):
    #total list of ham_row in a column then find the max length a which will be the row dimention
    a=[]
    cl_word=hm_word
    hm_word=[]
    for j in range(len(cl_word)):
        a.append(len(cl_word[j]))
    for j in range(len(cl_word)):
        n=cl_word[j].tolist()
        #finding highest row length
        w=max(a)
        for k in range(w):
            if k==len(cl_word[j]):
                b=len(cl_word[j])+1
                #append rest of by appending to
                for l in range(b,w+1):
                    #to added cause its  in stop word 
                    n.append('None')
        hm_word.append(n)        
hm_word_matrix=np.array(hm_word)
###end forming the ham word matrix

#forming a spam matrix
for i in range(1):
    #total list of spam_row in a column then find the max length a which will be the row dimention
    a=[]
    cl_word=sp_word
    sp_word=[]
    for j in range(len(cl_word)):
        a.append(len(cl_word[j]))
    for j in range(len(cl_word)):
        n=cl_word[j].tolist()
        #finding highest row length
        w=max(a)
        for k in range(w):
            if k==len(cl_word[j]):
                b=len(cl_word[j])+1
                #append rest of by appending to
                for l in range(b,w+1):
                    n.append('None')
        sp_word.append(n) 
#convert list to array
sp_word_matrix=np.array(sp_word)
###end forming the spam word matrix

#smoothing factor
f=.15   

#states calculaton for ham_word_matrix 
hm_obs=0
for i in range(len(hm_word)):
    a=len(hm_word[i])
    for j in range(a):
        hm_obs+=1
hm_states=int((hm_obs/len(hm_word)))

#states calculaton for spam_word_matrix 
sp_obs=0
for i in range(len(sp_word)):
    a=len(sp_word[i])
    for j in range(a):
        sp_obs+=1
sp_states=int((sp_obs/len(sp_word)))

#ith sate probabilty for any classes
def state_ith_prob(wd,cr_word,cr_word_matrix,state):
    c_R_d=[]
    count_R_d=0
    for i in range(1):
        a=len(cr_word)
        for k in range(a):
            if wd==cr_word_matrix[k][state]:
                c_R_d.append(1)
            count_R_d+=1 
    return len(c_R_d),count_R_d
      
#n_states probability
def state_tot_prob(wd,cr_word,cr_word_matrix):
    c_A_d=[]
    count_A_d=0
    for i in range(len(cr_word)):
        a=len(cr_word[i])
        for k in range(a):
            if wd==cr_word_matrix[i][k]:
                c_A_d.append(1)
            count_A_d+=1
    return len(c_A_d),count_A_d

#hmm model
def hmm(w,cl,state,cr_word,cr_word_matrix):
    if cl=='spam':
        R=state_ith_prob(w,cr_word,cr_word_matrix,state)
        A=state_tot_prob(w,cr_word,cr_word_matrix)
        R_d,R_ln=R 
        A_d,A_ln=A
        score=(f*(R_d/R_ln)+(1-f)*(A_d/A_ln))
    else:
        R=state_ith_prob(w,cr_word,cr_word_matrix,state)
        A=state_tot_prob(w,cr_word,cr_word_matrix)
        R_d,R_ln=R 
        A_d,A_ln=A
        score=(f*(R_d/R_ln)+(1-f)*(A_d/A_ln))
    return score

#transition matrix for ham
# y is transition list
y=[]
for i in range (hm_states):
    #row list appended in y
    a=[]
    for j in range (hm_states):
        #i=j+1 emitted 
        if j==i+1:
            a.append(1)
        else:
            a.append(0)
    y.append(a)
#y_list to array
hm_transition_matrix=np.array(y)

#transition matrix for spam 
# y is transition list
y=[]
for i in range (sp_states):
    #row list appended in y
    a=[]
    for j in range (sp_states):
        #i=j+1 emitted 
        if j==i+1:
            a.append(1)
        else:
            a.append(0)
    y.append(a)
#y_list to array
sp_transition_matrix=np.array(y)


#learning_Rate
n=10000

#classify the test sentences
def classify(sentence):
    sentence= re.sub(r'\d+','', sentence)
    sentence= re.sub('['+string.punctuation+']', '', sentence)
    sentence=sentence.lower()
    tokens=[]
    for word in nltk.word_tokenize(sentence):
        if word not in stop_words:
            stemmed_word = stemmer.stem(word.lower())
            tokens.append(stemmed_word)

    high_class = None
    high_score =0
    # loop through our classes
    for c in classes:
        # calculate score of sentence for each class
        if c=='ham':
            alpha=0
            sum=0
            #header loop for calculating alpha_T
            for i in  range(hm_states):
                #individual word in tokens and alpha_T count
                for word in tokens:
                    #word probabilty
                    a=hmm(word,c,i,hm_word,hm_word_matrix)              
#                    print(a)
                    #sates probabilty
                    for j in range(hm_states):
                        #at first state initial probability 1
                        if 0==i:
                            r=1
                        #r after state one states probability
                        else:
                            for j in range(hm_states):
                                #emitted only i=j+1
                                sum=sum+b*(hm_transition_matrix[j][i])
                            r=sum
                    #o1,o2,......oT probability
                    b=r*a
                    #i_th probability of a sentence
                    #alpha_T at the end
                    alpha=alpha+b
            score=alpha
            score=n*(score)
        else:
            c=='spam'
            alpha=0
            sum=0
            for i in  range(sp_states):
                for word in tokens:
                    
                    a=hmm(word,c,i,sp_word,sp_word_matrix)
                    for j in range(hm_states):
                        if 0==i:
                            r=1
                        else:
                            for j in range(sp_states):
                                sum=sum+b*sp_transition_matrix[j][i]
                            r=sum
                    b=r*a
                    alpha=alpha+b
            #score global variable for comparision
            score=alpha
            #n smoothing factor
            score=n*(score)  
        #determine class according to score
        if score > high_score:
            high_class = c
            high_score = score
    return high_class, high_score
#y_test calculation
z=[]
for j in range(len(X_test)):
    z.append(classify(X_test[j]))
#list to series
Z= pd.Series( (v[0] for v in z))  
y_pred=Z
#confusion metrix  
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score
cm = confusion_matrix(y_test, y_pred)
Accuracy_Score = accuracy_score(y_test, y_pred)
Precision=precision_score(y_test, y_pred, average='weighted')

