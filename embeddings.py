import pandas as pd
import numpy as np
from utils import alligned, get_sentece_vectors
from allennlp.commands.elmo import ElmoEmbedder
from tqdm import tqdm as tqdm
"""
elmo = ElmoEmbedder(
    options_file='./elmo/elmo_pt_options.json',
    weight_file='./elmo/elmo_pt_weights.hdf5',
    cuda_device=0
)

tqdm.pandas()
df = pd.read_csv("./corpus/final_.csv", sep="\t")

df["x_vecors"] = df["x"].progress_apply(
    lambda x: get_sentece_vectors(x, elmo))

df["y_vecors"] = df["y"].progress_apply(
    lambda x: get_sentece_vectors(x, elmo))

df.to_json("./distance_vectors.json")
"""

df = pd.read_json("./distance_vectors.json").sample(frac=0.2)
df.to_json("./sample_20.json")
list_df = alligned(df, cut_off=0.89)
pd.concat(list_df).to_csv("./new_allignments.csv")


