import pandas as pd
from scipy.sparse import csr_matrix

def load_interactions(path, id_map, n_users, n_items):
    df = pd.read_csv(path)
    df['item_idx'] = df['opportunity_id'].map(id_map)
    df = df.dropna()

    rows = df['user_id'].astype(int)
    cols = df['item_idx'].astype(int)
    data = df['interaction'].astype(float)

    return csr_matrix((data, (rows, cols)), shape=(n_users, n_items))
