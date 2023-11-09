# Heat Alerts Sequential Decision Making 

This is code for investigating applicability of reinforcement learning (RL) to environmental health, specifically issuance of heat alerts in the United States.

### Data Processing:
1. Merging mortality data and heat alerts data: Explore_merged_data.R
2. Processing county-level covariates: Extract_land_area.R, Get_county_Census_data.R, Prep_DoE_zones.R
3. Merge together: Merge-finalize_county_data.R
4. Include hospitalizations: Get_hosp_data.R
5. Add more covariates to address confounding: More_confounders.R
6. Finalize and select only counties with population > 65,000: Single_county_prep_data.R

### Installing the conda environment and getting the data ready for modeling:
```
conda env create -f envs/rl/env-linux.yaml
conda activate heatrl
```
Then run the script heat_alerts/scripts/prepare_bayesian_model_data.py to get everything in the right format for the Bayesian model and gym environment.

### Bayesian rewards modeling:
1. Run train_bayesian_model.py using Hydra arguments. Configurations are in the conf directory. For example:
```
python train_nn.py training=full_fast constrain=mixed model.name="FF_mixed"
```
See [here](https://hydra.cc/docs/intro/) for an introduction to Hydra. <br>
2. Evaluate the accuracy of these predictions with the script heat_alerts/bayesian_model/Evaluate_pyro_R.R

*Note: in the following sections, if you're using the provided shell (.sh) scripts, you will need to adjust the size of the job array depending on how many you run at once.*

### Evaluate benchmark policies:

1. Run heat_alerts/online_rl/Online_print_evals.R then Run_jobs/Online_evals.sh -- if you're using slurm, the command to run the latter is "sbatch"
2. Process these results using heat_alerts/scripts/Benchmark_evals.R

### Online RL:
To train an RL model, use the script train_online_rl_sb3.py -- note that there are many possible arguments, passed using Hydra / config files.

To reproduce the analyses in the paper: 

1. Tune hyperparameters for TRPO for each county (with and without forecasts / future information) by running heat_alerts/online_rl/Online_print_RLs.R followed by Run_jobs/Online_RL_short.sh and/or Run_jobs/Online_tuning.sh (if splitting up the job array is needed). Process these results using heat_alerts/scripts/Final_tuning_evals.R
2. Train comparison algos (DQN and PPO) using the same scripts ^^^
3. 
   
2. The gym environment is detailed in several scripts within the directory heat_alerts/online_rl
 * env.py contains the overall mechanics of stepping through and resetting the gym environment.
 * datautils.py contains several functions for data formatting, which is performed before calling the environment instantiation.
 * callbacks.py contains the calculation of custom metrics that we wish to save from each episode through the environment.
3. 


