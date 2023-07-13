##

Steps to get the model up and running

Run everything from from the folder `bayesian_model/` as root.

Make the conda env (replace conda with mamba if needed)
```
conda env create -f ../envs/heatrl/env-linux.yaml
conda activate heatrl
```

Generate the required preprocessing
```
python prepare_data.py
```

This will create a bunch of files under the folder `bayesian_model/data/preprocessed`. 

The files will look like

```
data/processed/
├── states.parquet  # time-varying features
├── actions.parquet  # alert or not
├── spatial_feats.parquet  # spatial features, num rows = num fips
├── fips2idx.json  # mapping from fips to row index in spatial_feats.parquet
├── location_indices.json  # location (fips index) f each row of states.parquet
├── offset.parquet  # offset for the poisson regresison, it is the mean of 
                    # corresponds to
```
The data is broken in several files. The only advantage is that it is mostly ML/RL ready
and that parquet files can be opened from both Python and R efficiently.

TODO: move the output of this processed files to the original `data` symlinked folder.

There are three pyro models currently.
* fit_data_pyro_base.py
* fit_data_pyro_simple.py
* fit_data_pyro_nn.py

I recommend only looking at the first one, the other two are a bit experimental. The model used by this file is esentially the same as the one in `bayesian_model/model_not_spatial.stan`, which might be easier to read if more used to Stan notation.


To run the models simply do
```
python fit_data_pyro_base.py
```

It might take around 4-5 hours. No need to use large memory or many CPUs (e.g. 16gb of memory and 8 cores should be fine).

Currently the results are saved as the posterior samples of the underlying parameters only. These will be saved as `fit_data_pyro_base.json`. The json is a dictionary with the name of a parameter of the model and the samples. They can be used to reconstruct posterior samples of `tau`, the effectiveness of the heat alert.

TODO: save the outputs on a more convenient way. Two options:
- Directly save the results of tau
- Save the model and guide, which together create an finite number of posterior samples using the `pyro.infer.Predictive` class.