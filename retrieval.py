import time
import numpy as np
import pandas as pd
import faiss


def single_emb_search(process_file_addr, query_emb, num_results):
    # process_file_addr: complete address of process file
    # query_emb: search vector oof shape [1, vector_length]
    process_file = pd.read_pickle('./process_files/' + process_file_addr + ".pkl")
    #face_embeddings = process_file['embeddings']
    #Expand Dims of Query to [1, dims]
    query_emb = np.expand_dims(query_emb, axis=0)
    
    # Reshape process_file['embeddings'] to [n, dims]
    embs = np.stack(process_file['embeddings'], axis=0)
    #embs = []
    #for vals in process_file['embeddings']:
    #    embs +=[vals]
    #embs = np.array(embs)
    #print(embs.shape)
    #print(embs.shape, query_emb.shape)
    if num_results > len(embs):
        num_results = len(embs)

    dims = query_emb.shape[1]
    maxM = 16
    index = faiss.IndexHNSWFlat(dims, maxM)
    # training is not needed
    # this is the default, higher is more accurate and slower to
    # construct
    index.hnsw.efConstruction = 40
    # to see progress
    index.verbose = True
    index.add(embs)

    t0 = time.time()
    #query = np.expand_dims(embs[50],1).T
    D, I = index.search(query_emb, num_results)
    t1 = time.time()
    #print(t1-t0)
    # Remove -1 from I
    I = [i for i in I[0] if i!=-1]
    #process_file[I[0]]
    # Filter out data
    #output = process_file.loc[I[0]]
    output = process_file.loc[I]
    #output = process_file[process_file.index.isin(I[0])]

    return output







