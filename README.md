# heat-alerts_mortality_RL

This is code for investigating applicability of RL to environmental health, specifically issuance of heat alerts.

### Data Processing:
1. Merging mortality data and heat alerts data: Explore_merged_data.R
2. Processing county-level covariates: Extract_land_area.R, Get_county_Census_data.R, Prep_DoE_zones.R
3. Merge together: Merge-finalize_county_data.R
4. Include hospitalizations: Get_hosp_data.R
5. Add more covariates to address confounding: More_confounders.R
6. Separate and finish cleaning the train and test sets: Make_train_test_sets.R
7. Export to csv (for Python) and create standardized dataset for further analyses in R: Data_for_Python.R

### Installing necessary packages:
Terminal commands to install the conda environment:
```bash
conda env create -f ../envs/heatrl/env-linux.yaml
conda activate heatrl
```
*Note: you have to be on a gpu-enabled partition for pytorch to install the cuda requirements.*

### Analysis:
