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

### Online RL:
1. Run the script train_online_rl_sb3.py OR train_online_rl_d3rlpy.py -- note that there are many possible arguments, passed using Hydra / config files. For the base configurations:
```
python train_online_rl_sb3.py --algo=ppo,qrdqn,trpo,dqn --multirun
```
 * The script heat_alerts/Online_print_jobs.R can be used to write out the terminal commands for many experiments, and the script Run_jobs/Online_RL.sh can be used to start a job array to run these using SLURM.
2. The gym environment is detailed in several scripts within the directory heat_alerts/online_rl
 * env.py contains the overall mechanics of stepping through and resetting the gym environment.
 * datautils.py contains several functions for data formatting, which is performed before calling the environment instantiation.
 * callbacks.py contains the calculation of custom metrics that we wish to save from each episode through the environment.
3. 


