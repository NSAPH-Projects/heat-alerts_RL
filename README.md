# Heat Alerts Sequential Decision Making 

This is code for investigating applicability of reinforcement learning (RL) to environmental health, specifically issuance of heat alerts in the United States. Additional information on the observational dataset we use can be found at the end of this document.

### Installing the conda environment:
```
conda env create -f envs/rl/env-linux.yaml
conda activate heatrl
```

Versions of primary software used: Python version 3.10.9; cuda version 12.0.1; R version 4.2.2

<ins>**\*\*\*Start here if you have access to the health data\*\*\***</ins>

### Data processing:
*In the directory heat_alerts/Data_Processing:*
1. Merging mortality data and heat alerts data: Explore_merged_data.R
2. Processing county-level covariates: Extract_land_area.R, Get_county_Census_data.R, Prep_DoE_zones.R, Process_election_data.R
3. Merge together: Merge-finalize_county_data.R
4. Include hospitalizations: Get_hosp_data.R
5. Add more covariates to address confounding: More_confounders.R
6. Finalize and select only counties with population > 65,000: Single_county_prep_data.R

Then run the script heat_alerts/scripts/prepare_bayesian_model_data.py to get everything in the right format for the Bayesian model and gym environment. This will create a bunch of files under the folder `data/processed`. 

The files will look like:

```
data/processed/
├── states.parquet  # time-varying features, num rows = num fips x num days
├── actions.parquet  # alert or not
├── spatial_feats.parquet  # spatial features, num rows = num fips 
├── fips2idx.json  # mapping from fips to row index in spatial_feats.parquet
├── location_indices.json  # location (fips index) f each row of states.parquet
├── offset.parquet  # offset for the poisson regresison, it is the mean number of hospitalizations for that summer
```
These files are formatted to be ready for ML/RL; parquet files can be opened from both Python and R efficiently.

### Bayesian rewards modeling:

The bulk of the code for this model is in heat_alerts/bayesian_model/pyro_heat_alert.py

1. Run train_bayesian_model.py using Hydra arguments. Configurations are in the conf directory. For example, to get the model used in our paper, which we determine to be the most robust without too many constraints:
```
python train_bayesian_model.py training=full_fast constrain=mixed model.name="FF_mixed"
```
*Note: whatever name the bayesian model is saved under should be pasted into the corresponding file in the conf/online_rl/sb3/r_model/ directory, for instance in the case above, in the mixed_constraints.yaml file we would write "guide_ckpt: ckpts/FF_mixed_guide.pt"*
See [here](https://hydra.cc/docs/intro/) for an introduction to Hydra. <br>

2. Evaluate the accuracy of these predictions (i.e. $R^2$) and identify counties with (a) high number of alerts and (b) high estimated (variance of) effectiveness of heat alerts with the script heat_alerts/bayesian_model/Evaluate_pyro_R.R

<ins>**\*\*\*Start here if you just have access to the simulator, not the health data\*\*\***</ins>

We can *validate* the bayesian rewards model using the following scripts in the heat_alerts/bayesian_model/ directory:
1. Run Validate_model.py with --type="initial" and --model_name="FF_mixed"
2. Train a model on these sampled outcomes using the flag sample_Y=true (again using the script train_bayesian_model.py)
3. Run Validate_model.py again with --type="validation" and the name of the new model trained on the fake data
4. Calculate coverage using Validate_model_intervals.R

### Gym environment (simulator):

The gym environment is detailed in several scripts within the directory heat_alerts/online_rl:
 * env.py contains the overall mechanics of stepping through and resetting the gym environment.
 * datautils.py contains several functions for data formatting, which is performed before calling the environment instantiation.
 * callbacks.py contains the calculation of custom metrics that we wish to save from each episode through the environment and can view using tensorboard -- used primarily during RL training.

**Note: in the following sections, if you're using the provided shell (.sh) scripts, you will need to adjust the size of the job array depending on how many you run at once.**

### Evaluate benchmark policies:

1. Run heat_alerts/online_rl/Online_print_evals.R then Run_jobs/Online_evals.sh -- if you're using slurm, the command to run the latter is "sbatch"
2. Process these results using heat_alerts/scripts/Benchmark_evals.R

### Online RL:
To train an RL model, use the script train_online_rl_sb3.py -- note that there are many possible arguments, passed using Hydra / config files. 

To reproduce the analyses in the paper: 

1. Tune hyperparameters for TRPO for each county (with and without forecasts / future information) by running heat_alerts/online_rl/Online_print_RLs.R followed by Run_jobs/Online_RL_short.sh and/or Run_jobs/Online_tuning.sh (if splitting up the job array is needed). Process these results using heat_alerts/scripts/Final_tuning_evals.R
2. Train comparison algos (DQN and PPO) and process the evaluation results using the same scripts ^^^

### Generate figures and tables for the paper:
1. Table of descriptive statistics: heat_alerts/scripts/Summary_stats_table.R
2. Plot of coefficients sampled from the Bayesian rewards model posterior: heat_alerts/bayesian_model/Make_plots.py
3. Supplementary "time series" plot of observed quantile of heat index, modeled baseline NOHR hospitalizations, and modeled alert effectiveness (assuming no past alerts) across days of summer: (i) get data from heat_alerts/scripts/Time_series_plots.py, then (ii) make plots with heat_alerts/scripts/Make_TS_plots.R
4. Main RL results table, supplementary table of county characteristics, and supplementary table of RL results using the per-alert (compare_to_zero) metric: heat_alerts/scripts/Make_tables_for_paper.R
   - To obtain an approximate confidence interval for the absolute number of NOHR hospitalizations saved, use heat_alerts/scripts/Approx_CI.R
6. Supplemental plots of different heat alert policies for individual counties: heat_alerts/scripts/Mini-case-studies.R
7. Boxplot and histograms of day-of-summer and alert streak lengths: heat_alerts/scripts/Make_plots_for_paper_FINAL.R
8. CART analysis / plots: heat_alerts/scripts/Investigate_systematic_diffs.R

****

### Additional information on the data:

We start with a US county-level dataset spanning 2006-2016 (warm months) which has been used in past studies of heat alert effectiveness. Main variables in this dataset are daily values of ambient heat index, heat alerts issued by the National Weather Service, and the number of in-patient fee-for-service Medicare hospitalizations for causes associated with extreme heat in past studies. We additionally compile other datasets to help characterize variability in the health impacts of extreme heat and heat alerts, such as sociodemographics and regional climate zone classifications. References for these datasets and past studies are in the main text.

The main analysis in this paper uses a gym environment (simulator of environmental variables and health outcomes) that we created based on the observed data. This simulator is publicly available. However, the health data that were used to create the model of the health outcomes are highly sensitive, and are only available to researchers with qualifying private servers. Access can be requested via application to the Centers for Medicare and Medicaid Services (see https://www.resdac.org/research-identifiable-files-rif-requests). 

