# Predicting Bushfire Propagation from NASA FIRMS Data

This repository contains an implementation of an LSTM model designed to predict the propagation of the 2019 Australian bushfires.

The dataset:  
[https://www.kaggle.com/datasets/carlosparadis/fires-from-space-australia-and-new-zeland?select=fire_archive_M6_96619.csv](https://www.kaggle.com/datasets/carlosparadis/fires-from-space-australia-and-new-zeland?select=fire_archive_M6_96619.csv)

## Project Structure

- `graph_modis.ipynb` contains an initial exploration of the dataset. This includes generating a folium HeatMapOverTime of fire pixels from the data.
- `prepare_modis.ipynb` does data cleaning and computes timestamps from the acq\_date and acq\_time fields.
- `preproc.py` contains functions for organizing the data into sequences and scaling the values for the models.
- `preprocess.ipynb` uses these functions on the dataset and saves the results.
- `lstm_model.py` defines the model class (`FirePredictor`) so that it can be used in multiple notebooks
- `train.ipynb` trains `FirePredictor` model and writes the wieghts to a JSON file
- `eval_model.ipynb` graphs the model's predictions against the ground trush values and computes some performance statistics

- `graphs` contains both folium graphs (html files) and learning curve graphs (png files)
    - `fire_archive_M6_96619.html` is the heatmap over time visualization for the `raw_data` csv file of the same name
    - `model_comparison.html` compares the output of the baseline model to the ground truth
    - `reduced_comparison.html` compares the output of the baseline reduced area model to the ground truth
    - `optimized_reduced_comparison.html` compares the output of the optimized reduced area model to the ground truth
    - `with_rect.html` shows where the reduced geographical area is
    - `original_curve.png` shows the learning curve of the baseline model
    - `reduced_curve.png` shows the learning curve of the baseline reduced area model
    - `optimized_reduced_curve.png` shows the learning curve of the optimized reduced area model

- `dataset` contains both the original data, and all of the cleaned and preprocessed data
    - `raw_data` contains the original dataset
    - `modis_reduced` contains the reduced geographical area dataset

    - `preprocessed` contains preprocessed MODIS data, ready to be fed into the baseline model
    - `reduced_preprocessed` contains the preprocessed data for the reduced area dataset
    - `optimized_reduced_preprocessed` contains the preprocessed data with a sequence length of 32

    - each of the preprocessed folders are created& named for a particular version of the model that I tested

