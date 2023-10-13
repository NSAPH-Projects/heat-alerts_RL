
source("heat_alerts/scripts/Evaluation_functions.R")

## Change manually:
eval_func_name<- "compare_to_zero" 
# eval_func_name<- "avg_return"

#### Run evaluations:

if(eval_func_name == "compare_to_zero"){ # if using compare_to_zero, need additional argument below
  eval_func<- compare_to_zero 
}else if(eval_func_name == "avg_return"){
  eval_func<- avg_return
}


#### Expand grid:
tune_counties<- c(32003, 29019, 45015, 19153, 41053)
tune_HI<- c(0.55, 0.7, 0.85)
tune_forecasts<- c("none", "all") # , "quantiles"
NHU<- c(16, 32, 64)
NHL<- c(1, 2, 3)
LR<- c(0.001, 0.0001, 0.01)
gamma<- c(1, 0.999, 0.99)
n_steps<- c(2048, 4096, 512)

## First time:
filling<- FALSE
DF<- expand.grid(tune_counties, tune_HI, tune_forecasts, 
                 NHU, NHL, LR, gamma, n_steps)
names(DF)<- c("tune_counties", "tune_HI", "tune_forecasts", 
              "NHU", "NHL", "LR", "gamma", "n_steps")

DF$Eval<- 0

## Filling in:
filling<- TRUE
DF<- read.csv(paste0("Fall_results/Tuning_evals_", eval_func_name, ".csv"))[,-1]


## If eval_func != compare_to_zero:

start<- Sys.time()

for(i in 1:nrow(DF)){
  m<- paste0("tune_F-", DF$tune_forecasts[i], "_Rstr-HI-", DF$tune_HI[i], 
             "_arch-", DF$NHL[i], "-", DF$NHU[i], 
             "_lr-", DF$LR[i], "_g-", DF$gamma[i], "_ns-", DF$n_steps[i],
             "_fips-", DF$tune_counties[i])
  if(filling & is.na(DF$Eval[i])){
    DF$Eval[i]<- eval_func(paste0("Summer_results/ORL_RL_train_samp-R_samp-W_", m, "_fips_", DF$tune_counties[i], ".csv"))
    if(i %% 50 == 0){
      print(paste(i, "=", Sys.time() - start))
    }
  }
}

write.csv(DF, paste0("Fall_results/Tuning_evals_", eval_func_name, ".csv"))


## If eval_func == compare_to_zero:

start<- Sys.time()

for(i in 1:nrow(DF)){
  m<- paste0("tune_F-", DF$tune_forecasts[i], "_Rstr-HI-", DF$tune_HI[i], 
             "_arch-", DF$NHL[i], "-", DF$NHU[i], 
             "_lr-", DF$LR[i], "_g-", DF$gamma[i], "_ns-", DF$n_steps[i],
             "_fips-", DF$tune_counties[i])
  
  filename_zero_train<- paste0("Summer_results/ORL_NA_train_samp-R_samp-W_mixed_constraints_fips_", DF$tune_counties[i], ".csv")
  
  if(filling & is.na(DF$Eval[i])){
    DF$Eval[i]<- eval_func(paste0("Summer_results/ORL_RL_train_samp-R_samp-W_", m, "_fips_", DF$tune_counties[i], ".csv"),
                           filename_zero_train)
    if(i %% 50 == 0){
      print(paste(i, "=", Sys.time() - start))
    }
  }
}

write.csv(DF, paste0("Fall_results/Tuning_evals_", eval_func_name, ".csv"))


#### Inspect results: regress Eval on c(tune_forecasts, NHU, NHL, LR, gamma, n_steps),
  # after selecting the best HI threshold for each?

tune_counties<- c(32003, 29019, 45015, 19153, 41053)

## Change manually:
eval_func_name<- "compare_to_zero" 
# eval_func_name<- "avg_return"

DF<- read.csv(paste0("Fall_results/Tuning_evals_", eval_func_name, ".csv"))[,-1]

vars<- c("tune_counties", "tune_forecasts", "NHU", "NHL", "LR", "gamma", "n_steps")
opt_DF<- distinct(DF[,vars])

opt_DF$Eval<- 0
opt_DF$OT<- 0

for(j in 1:nrow(opt_DF)){
  pos<- which(DF[,vars] == opt_DF[j, vars])
}

