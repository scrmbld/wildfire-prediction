import pandas as pd
import numpy as np

# define some constants
WINDOW_SIZE = 64 # also the sequence length
SLIDE_FACTOR = 0.3 # how much we slide the window by when making sequences
BATCH_SIZE = 64 # I don't think this is used by I also don't trust my LSP with jupyter notebooks

def get_bounds(df, cols = None):
    """
    Determine the maximum and minimum value for every column listed in `cols`. If `cols` is not specified, do every column.
    """
    if cols is None:
        cols = df.columns
    bounds = {}
    for col in cols:
        bounds[col] = {
            "max": float(df[col].max()),
            "min": float(df[col].min())
        }

    return bounds

def scale(x, min, max):
    """
    Scale a value into [0, 1] given the min and max for that measurement.
    """
    return (x-min) / (max-min)

def unscale(x, min, max):
    """
    Undo the scaling for a value given the original min and max for that measurement.
    """
    rng = max - min
    return (x*rng) + min

def create_map(df, col):
    """
    Create a dictionary to map tokens to integers.
    """
    unique_set = set(df[col].unique())
    return {key: i for i,key in enumerate(unique_set)}

def preprocess(df, bounds):
    """
    Preprocess the dataframe *specifically* for our model.
    """
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
    """
    Undo the effectos of the function `preprocess` for the given data frame.
    """
    res = df.copy()
    for col in res.columns:
        b = bounds.get(col)
        if b is not None:
            res[col] = res[col].apply(unscale, args=(b['min'], b['max']))

    # turn the day/night column back into letters
    # res['daynight'] = res['daynight'].map(inv_daynight_map).fillna(res['daynight'])

    return res

# split into inputs and labels
def xy_split(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Split into inputs and labels.
    """
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
    """
    Arrange the given x and y values into sequences using a sliding window algorithm.
    """
    sequence_x = []
    sequence_y = []
    for i in range(0, xs.shape[0]-WINDOW_SIZE, int(WINDOW_SIZE*SLIDE_FACTOR)):
        sequence_x.append(xs[i:i+WINDOW_SIZE])
        sequence_y.append(ys[i+WINDOW_SIZE])

    sequence_x = np.array(sequence_x)
    sequence_y = np.array(sequence_y)

    print(f'{sequence_x.shape=}, {sequence_y.shape=}')
    return np.concatenate([sequence_x[:,:,:], sequence_y[:,None,:]], axis=1)

