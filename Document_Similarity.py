#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Haridut
"""

import numpy as np
import sys
filename=open(sys.argv[1],'r',encoding='UTF-8') 
filename=filename.readlines()
#Creating Vocabulary
vocabulary=[]
for line in filename:
    for word in line.split():
        if word not in vocabulary:
                vocabulary.append(word) 
print (len(vocabulary)) #Prints the number of Unique Words in the Corpus = |V|
#Word Frequency in each document and simultaneous building of idf
a={}
dicti={}
i=1
for line in filename:
    vocab='vocab%d' %i
    a[vocab]={}
    i=i+1
    for word in line.split():
            if word in a[vocab]:
                a[vocab][word]+=1
            else:
                a[vocab][word]=1
                try:
                  dicti[word]+=1
                except:
                    dicti[word]=1
#Calculating tf vector for each document
tf_t={}
i=1
j=1
for line in filename:
    length=len(line.split())
    tf_t[i]=[]
    vocab='vocab%d'%j
    j+=1
    for word in vocabulary:
        try:
            tf_t[i].append((a[vocab][word]/length))
        except:
            b=0
            tf_t[i].append(b)
    i=i+1 
#Calculating idf
idf=[]
for word in vocabulary:
    val=(np.log(len(filename)/dicti[word]))
    idf.append(val)
#Calculating tf-idf
d={}
i=1
for line in filename:
    tf_t[i]=np.array(tf_t[i])
    idf=np.array(idf)
    d[i]=np.multiply(tf_t[i],idf)
    i=i+1
#Manhattan (Manhattan)
Manhattan_dist=[]
for i in range(1,501):
       dist=0
       for j in range(0,len(d[i])):
           dist+=abs(d[i][j]-d[500][j])
       ans=dist
       if ((ans-int(ans))==0):
              Manhattan_dist.append(ans)
       else:
           ans=round(ans,3)
           Manhattan_dist.append(ans)
#Euclidean distance       
Euclidean_dist=[]
for i in range(1,501):
       euc_dist=0
       for j in range(0,len(d[i])):
           euc_dist+=((d[i][j]-d[500][j])**2)
       ans=(np.sqrt(euc_dist))
       if ((ans-int(ans))==0):
          Euclidean_dist.append(ans)
       else:
          ans=round(ans,3)
          Euclidean_dist.append(ans)
#Supremum Distance
Supremum_dist=[]
for i in range(1,501):
       sup_dist=[]
       for j in range(0,len(d[i])):
           sup_dist.append(abs(d[i][j]-d[500][j]))
       ans=max(sup_dist)
       if ((ans-int(ans))==0):
         Supremum_dist.append(ans)
       else:
         ans=round(ans,3)
         Supremum_dist.append(ans)

#CosineSimilarity
cosine_dist=[]
for i in range(1,501):
      mag1=0
      magQ=0
      run_sum=0
      for j in range(0,len(d[i])):
          mag1+=((d[i][j])**2)
          magQ+=((d[500][j])**2)
          run_sum+=d[i][j]*d[500][j]
      mag1=np.sqrt(mag1) #magnitude of vectors
      magQ=np.sqrt(magQ) #magnitude of the 500th vector 'Q'
      cosined=(run_sum/mag1*magQ)
      if ((cosined-int(cosined))==0):
           cosine_dist.append(cosined)
      else:
         cosined=round(cosined,3)
         cosine_dist.append(cosined)
#Finding Top 5 Most Similar Documents in different measure of distances    
least_Manhattan=np.argsort(Manhattan_dist) 
print(str((least_Manhattan[0])+1)+' '+ str((least_Manhattan[1])+1) +' '+ str((least_Manhattan[2])+1)+' '+ str((least_Manhattan[3])+1)+' '+ str((least_Manhattan[4])+1))
least_Euclidean=np.argsort(Euclidean_dist) 
print(str((least_Euclidean[0])+1)+' '+ str((least_Euclidean[1])+1) +' '+ str((least_Euclidean[2])+1)+' '+ str((least_Euclidean[3])+1)+' '+ str((least_Euclidean[4])+1))
least_Supremum=np.argsort(Supremum_dist) 
print(str((least_Supremum[0])+1)+' '+ str((least_Supremum[1])+1) +' '+ str((least_Supremum[2])+1)+' '+ str((least_Supremum[3])+1)+' '+ str((least_Supremum[4])+1))
least_Cosine=np.argsort(cosine_dist) 
print(str((least_Cosine[-1])+1)+' '+ str((least_Cosine[-2])+1) +' '+ str((least_Cosine[-3])+1)+' '+ str((least_Cosine[-4])+1)+' '+ str((least_Cosine[-5])+1))            
#Rearranging the tf-idf vector for PCA Transformation: (Cleaning)
trans_list=[]
for i in range(len(vocabulary)):
       sublist=[]
       trans_list.append(sublist)
t=0
j=1
i=0
for word in vocabulary:
    for j in range(1,501):
        trans_list[i].append(d[j][t])
    t=t+1
    i=i+1
trans_matrix=np.asarray(trans_list)#Produces a 7325X500 matrix from list of lists
trans_matrix=trans_matrix.transpose() #Produces a 500X7325 matrix for PCA
#PCA-Dimension reduction to 2-D
from sklearn.decomposition import PCA as sklearnPCA 
sklearn_pca = sklearnPCA(n_components=2) #Reducing into two dimensions
sklearn_transf = sklearn_pca.fit_transform(trans_matrix) 
#  PCA Based Euclidean distance calculation
PCA_eucdistance=[]
for i in range(500):
    pca_eucdist=0
    for j in range(2):
        pca_eucdist+=((sklearn_transf[i][j]-sklearn_transf[499][j])**2)
    ans=(np.sqrt(pca_eucdist))
    if (ans-int(ans)==0):
        PCA_eucdistance.append(ans)
    else:
        ans=round(ans,3)
        PCA_eucdistance.append(ans)
#Top 5 according to PCA
least_PCAeucdist=np.argsort(PCA_eucdistance)
print(str((least_PCAeucdist[0])+1)+' '+ str((least_PCAeucdist[1])+1) +' '+ str((least_PCAeucdist[2])+1)+' '+ str((least_PCAeucdist[3])+1)+' '+ str((least_PCAeucdist[4])+1))
