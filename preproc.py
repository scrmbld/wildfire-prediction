import pandas as pd
import numpy as np

# define some contests
WINDOW_SIZE = 64
SLIDE_FACTOR = 0.3
BATCH_SIZE = 64

# get some min/max values
def get_bounds(df, cols = None):
    if cols is None:
        cols = df.columns
    bounds = {}
    for col in cols:
        bounds[col] = {
            "max": float(df[col].max()),
            "min": float(df[col].min())
        }

    return bounds

# create a function to scale values into [0, 1]
def scale(x, min, max):
    return (x-min) / (max-min)

def unscale(x, min, max):
    rng = max - min
    return (x*rng) + min

def create_map(df, col):
    unique_set = set(df[col].unique())
    return {key: i for i,key in enumerate(unique_set)}

def preprocess(df, bounds):
    res = df.copy()
    # # create the daynight mapping
    # daynight_map = create_map(res, "daynight")
    # inv_daynight_map = {i: key for key, i in daynight_map.items()}
    # # apply the mapping
    # res['daynight'] = res['daynight'].map(daynight_map).fillna(res['daynight'])
    # res.head(10)

    # scale every column in the dataset
    for col in res.columns:
        b = bounds.get(col)
        if b is not None:
            res[col] = res[col].apply(scale, args=(b['min'], b['max']))

    return res

def unprocess(df, bounds):
    res = df.copy()
    for col in res.columns:
        b = bounds.get(col)
        if b is not None:
            res[col] = res[col].apply(unscale, args=(b['min'], b['max']))

    # turn the day/night column back into letters
    # res['daynight'] = res['daynight'].map(inv_daynight_map).fillna(res['daynight'])

    return res

# split into inputs and labels
def xy_split(df):
    dset_x = []
    dset_y = []
    for i in range(df.shape[0]-1):
        dset_x.append(df.loc[i])
        dset_y.append(df.loc[i+1])

    dset_x = np.array(dset_x)
    dset_y = np.array(dset_y)

    return dset_x, dset_y


# arrange into sequences
# we need to do this because we cannot randomly pick readings from the dataset and still have valid inputs
# | their relative location is *very* important
def sequencify(xs, ys):
    sequence_x = []
    sequence_y = []
    for i in range(0, xs.shape[0]-WINDOW_SIZE, int(WINDOW_SIZE*SLIDE_FACTOR)):
        sequence_x.append(xs[i:i+WINDOW_SIZE])
        sequence_y.append(ys[i+WINDOW_SIZE])

    sequence_x = np.array(sequence_x)
    sequence_y = np.array(sequence_y)

    print(f'{sequence_x.shape=}, {sequence_y.shape=}')
    return np.concatenate([sequence_x[:,:,:], sequence_y[:,None,:]], axis=1)

