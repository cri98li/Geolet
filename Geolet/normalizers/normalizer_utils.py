"""
ritorna un dataset nella forma class, columns_n... con tid come index
"""
import numpy as np
import pandas as pd
from tqdm.auto import tqdm


def dataframe_pivot(df, maxLen, verbose, fillna_value, columns):
    df["pos"] = df.groupby(['tid', 'partId']).cumcount()

    if maxLen is not None:
        if maxLen >= 1:
            if verbose: print(F"Cutting sub-trajectories length at {maxLen} over {df.max().pos}", flush=True)
            df = df[df.pos < maxLen]
        else:
            if verbose: print(F"Cutting sub-trajectories length at {df.quantile(.95).pos} over {df.max().pos}", flush=True)
            df = df[df.pos < df.quantile(.95).pos]

    if verbose: print("Pivoting tables", flush=True)

    i = -1
    prec_partID = -1
    j=0

    #righe: tid; colonne: partId, class, altre
    max = df.pos.max()+1
    new_matrix = np.ones((len(df.partId.unique()), int(max*len(columns))))*np.NAN
    classes = []
    partIds = []

    for row in tqdm(df.values, disable=not verbose, position=0, leave=True):
        curr_partID = row[df.columns.tolist().index("partId")]
        classe = row[df.columns.tolist().index("class")]
        if prec_partID != curr_partID:
            prec_partID = curr_partID
            j = 0
            classes.append(classe)
            partIds.append(curr_partID)
            i += 1

        for k, colName in enumerate(columns):
            new_matrix[i, j+max*k] = row[df.columns.tolist().index(colName)]

        j+=1

    df_pivot = pd.DataFrame(new_matrix, columns=[str(x) for x in range(max*len(columns))])
    df_pivot["class"] = classes
    df_pivot["partId"] = partIds

    if fillna_value is not None:
        df_pivot.fillna(fillna_value, inplace=True)

    return df_pivot.set_index("partId")[["class"]+[x for x in df_pivot.columns if x not in ["class", "partId"]]]