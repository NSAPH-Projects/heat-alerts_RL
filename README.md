# heat-alerts_mortality_RL

This is code for investigating applicability of RL to environmental health, specifically issuance of heat alerts.

### Data Processing:
1. Merging mortality data and heat alerts data: Explore_merged_data.R
2. Processing county-level covariates: Extract_land_area.R, Get_county_Census_data.R, Prep_DoE_zones.R
3. Merge together: Merge-finalize_county_data.R
4. Include hospitalizations: Get_hosp_data.R
5. Add more covariates to address confounding: More_confounders.R
6. Finalize and select only counties with population > 65,000: Single_county_prep_data.R

### Installing the conda environment and getting the data ready for modeling:
```
conda env create -f ../envs/heatrl/env-linux.yaml
conda activate heatrl
```
Then run the script bayesian_model/prepare_data.py to get everything in the right format for the Bayesian model and gym environment.

### Bayesian rewards modeling:
*From within the bayesian_model directory:*
1. Run train_nn.py using Hydra arguments. Configurations are in the conf directory. For example:
```
python train_nn.py training=full model.name="Full_8-1"
```
See [here](https://hydra.cc/docs/intro/) for an introduction to Hydra. <br>
2. Evaluate the accuracy of these predictions with the script Evaluate_pyro_R.R

### Online RL:

