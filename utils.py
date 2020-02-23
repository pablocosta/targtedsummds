import math
import scipy
import pandas as pd
import numpy as np
from tqdm import tqdm as tqdm
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from numba import jit, cuda


def get_sentece_vectors(sentence, model, doc_freqs=False):
    sent = sentence.split(" ")
    tokfreqs = Counter(sent)
    embeddings_map = {}
    embeddings = model.embed_sentence(sentence)
    for i in range(0, len(sent)):
        embeddings_map[sent[i]] = np.array(embeddings[2][i])
    weights = [tokfreqs[token] * math.log(N / (doc_freqs.get(token, 0) + 1))
               for token in tokfreqs] if doc_freqs else None
    embedding1 = np.average([embeddings_map[token]
                             for token in tokfreqs], axis=0, weights=weights).reshape(1, -1)
    return embedding1




@jit(nopython=True)
def cos_sim(a, b):
	"""Takes 2 vectors a, b and returns the cosine similarity according 
	to the definition of the dot product
	"""
	return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


@jit(nopython=False)
def do_similarities(x, y):
    result =  np.zeros(y.shape[0])

    for ele in range(y.shape[0]):
        result[ele] = cos_sim(x[0], np.array(y[ele])[0])
    return result

def get_context_avg(row, y, y_name, age, gender, state, cut_off):
    df_return = pd.DataFrame()
    
    df_return["y"] = y_name
    df_return["age"] = age
    df_return["genero"] = gender
    df_return["estado"] = state

    similarities = do_similarities(np.array(row["x_vecors"]), y)

    df_return["similaridade"] = similarities
    df_return["x"] = pd.Series([row["x"]]).repeat(df_return.shape[0]).to_list()
    df_return = df_return[df_return["similaridade"] >= cut_off]

    return df_return

def alligned(df, model=None, cut_off=0.89):
    list_df = []
    for _, row in tqdm(df.iterrows()):
        list_df.append(get_context_avg(row, df["y_vecors"].to_numpy(),  df["y"], df["age"], df["genero"], df["estado"], cut_off=cut_off))
    return list_df
