from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn as nn
import math
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

tokenizer = AutoTokenizer.from_pretrained('./sentence_transformers')
model = AutoModel.from_pretrained('./sentence_transformers')

Embedded_sentence_data = []
sentences = []
question = input("please input a query: ")
sentences.append(question)
max_sim = []
for i in df['Contents']:
    sentences.append(i)
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    sentence_embedding = mean_pooling(model_output, encoded_input['attention_mask']).detach().numpy()
    Embedded_sentence_data.append(sentence_embedding[1])
    a = cosine_similarity([sentence_embedding[0]], sentence_embedding[1:])
    max_sim.append(a)
    sentences.remove(sentences[1])
    
#PCA
def normalize(X):
    means = X.mean(axis=0)
    stds = X.std(axis=0)
    X_normalized = (X - means) / stds
    return X_normalized, means, stds

def pca(X):
    sigma = (X.T @ X) / len(X)
    U, S, V = np.linalg.svd(sigma)
    return U, S, V

Embedded_sentence_data = np.array(Embedded_sentence_data)
X_normalized, means, stds = normalize(Embedded_sentence_data)
U, S, V = pca(X_normalized)    

#Project data to low dimention
def projectData(X, U, K):
    Z = X @ U[:, :K]
    return Z

three_d_data = projectData(X_normalized, U, 3)

#plot 3D picture
plt.rcParams['axes.unicode_minus']=False
fig = plt.figure(figsize=(10,10))
ax = Axes3D(fig)
#ax = fig.add_subplot(111, projection='3d')
x = three_d_data[:,0]
y = three_d_data[:,1]
z = three_d_data[:,2]
x_q = sentence_embedding[0][0]
y_q = sentence_embedding[0][1]
z_q = sentence_embedding[0][2]

ax.scatter(xs=x, ys=y, zs=z, c='#4d3333', s=10, label='all sentence', marker='o')
ax.scatter(xs=x_q, ys=y_q, zs=z_q, c='red', s=10, label='question sentence', marker='x')
ax.set_zlabel('third dimention', fontdict={'size': 15, 'color': 'black'}) 
ax.set_ylabel('second dimention', fontdict={'size': 15, 'color': 'black'})
ax.set_xlabel('first dimention', fontdict={'size': 15, 'color': 'black'})

plt.show()
if max(max_sim) < 0.5:
    print("No answer match.")
else:
    print(max(max_sim))
    print(max_sim.index(max(max_sim)))