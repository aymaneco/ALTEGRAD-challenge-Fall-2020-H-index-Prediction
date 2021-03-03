
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np

def remove_embedding(df_train, df_auth_emb):
    """
    Remove authors that have the embedding vector equals to 0 from the training data and the embedding matrix
    Inputs :
        df_train : The training data ;
        df_auth_emb : Author embedding
    Output :
        df_train : The training data after removing the author embedding that equals to 0 ;
        df_auth_emb : Updated Author embedding withou 0 vectors
    """
    auth_emb_nul = []
    index_ = df_auth_emb.T.sum().index
    sum_ = df_auth_emb.T.sum().values
    auth_emb_nul = []
    for i in range(len(index_)):
      if sum_[i]==0:
        auth_emb_nul.append(index_[i])
    ############################################
    auth_emb_nul_train = []
    for _,row in df_train.iterrows():
        node = row['authorID']
        if int(node) in auth_emb_nul:
          auth_emb_nul_train.append(node)
    ############################################
    df_auth_emb = df_auth_emb.drop(auth_emb_nul_train)
    df_train = df_train[~df_train["authorID"].isin(auth_emb_nul_train)]
    df_train = df_train.reset_index()
    return df_train, df_auth_emb

def pca_node_embedding(n_1, df_node_emb):
    """
    Inputs :
        n_1 : Number of n_components ;
        df_node_emb : Node embedding
    Output :
        df_node_emb_pca : PCA applied on node embedding
    """   
    pca_node_emb = PCA(n_components=n_1,svd_solver='arpack')
    df_node_emb_pca = pca_node_emb.fit_transform(df_node_emb)
    df_node_emb_pca = pd.DataFrame(df_node_emb_pca).set_index(df_node_emb.index)
    print(np.sum(pca_node_emb.explained_variance_ratio_))
    return df_node_emb_pca

def pca_author_embedding(n_2, df_auth_emb):
    """
    Inputs :
        n_2 : Number of n_components ;
        df_auth_emb : Author embedding
    Output :
        auth_doc2vec_pca : PCA applied on author embedding
    """   
    pca_auth_doc2vec = PCA(n_components=n_2,svd_solver='arpack') 
    auth_doc2vec_pca = pca_auth_doc2vec.fit_transform(df_auth_emb)
    auth_doc2vec_pca = pd.DataFrame(auth_doc2vec_pca).set_index(df_auth_emb.index)
    print(np.sum(pca_auth_doc2vec.explained_variance_ratio_))
    return auth_doc2vec_pca