
source("heat_alerts/scripts/Evaluation_functions.R")

## Change manually:
# eval_func_name<- "compare_to_zero" 
eval_func_name<- "avg_return"
# eval_func_name<- "per_alert"

#### Run evaluations:

if(eval_func_name == "per_alert"){
  eval_func<- per_alert
}else if(eval_func_name == "compare_to_zero"){ # if using compare_to_zero, need additional argument below
  eval_func<- compare_to_zero 
}else if(eval_func_name == "avg_return"){
  eval_func<- avg_return
}

HI_thresholds<- seq(0.5, 0.9, 0.05)

## If eval_func != compare_to_zero:

for(r_model in c("mixed_constraints" 
                 # , "alert_constraints", "all_constraints", "no_constraints", "hi_constraints"
)){
  results<- matrix(
    0, nrow=length(counties), 
    # ncol=6 # [trpo]*[none, all]*[none, qhi, qhi_ot]
    # ncol=18 # [trpo]*[6 forecast options]*[none, qhi, qhi_ot]
    ncol=4 # [ppo, dqn]*[eval_qhi, qhi_ot]
  )
  my_names<- rep("", ncol(results))
  
  for(k in 1:length(counties)){
    county<- counties[k]
    
    i<- 1 # column tracker
    for(algo in c( # "trpo",
                   "dqn",
                   "ppo"
    )){
      if(algo == "dqn"){
        forecast_list<- c("none", "all")
      }else{
        forecast_list<- c(
          # "none", "all",  
          "num_elig", "quarters", "three_day", "ten_day", "quantiles", "N_Av4_D3"
        )
      }
      for(forecasts in forecast_list){
        m<- paste0(r_model, "_", algo, "_F-", forecasts, "_fips-", county)
        results[k,i]<- eval_func(paste0("Summer_results/ORL_RL_eval_samp-R_obs-W_", m, "_fips_", county, ".csv"))
        my_names[i]<- paste0(algo, "_F-", forecasts)
        i<- i+1
        Models<- paste0(r_model, "_", algo, "_F-", forecasts, "_Rstr-HI-", HI_thresholds, "_fips-", county)
        for(h in 1:length(Models)){
          m<- Models[h]
          train_samp<- eval_func(paste0("Summer_results/ORL_RL_train_samp-R_samp-W_", m, "_fips_", county, ".csv"))
          eval<- eval_func(paste0("Summer_results/ORL_RL_eval_samp-R_obs-W_", m, "_fips_", county, ".csv"))
          if(h == 1){
            Eval_samp<- train_samp
            Eval<- eval
            j<- 1
          }else{
            if(train_samp > Eval_samp){
              Eval_samp<- train_samp
              Eval<- eval
              j<- h
            }
          }
        }
        results[k,i]<- Eval
        my_names[i]<- paste0(algo, "_qhi_F-", forecasts)
        i<- i+1
        my_names[i]<- paste0("ot_", algo, "_qhi_F-", forecasts)
        results[k,i]<- HI_thresholds[j]
        i<- i+1
        print(forecasts)
      }
    }
    print(county)
  }
  DF<- round(data.frame(results),3)
  names(DF)<- my_names
  
  f<- paste0("Fall_results/RL_evals_", r_model, "_", eval_func_name, ".csv")
  if(file.exists(f)){
    old<- read.csv(f)[,-1]
    DF<- cbind(old, DF)
  }
  write.csv(DF, f)
  print(paste("Finished with", r_model))
}


## If eval_func == compare_to_zero:

for(r_model in c("mixed_constraints" 
                 # , "alert_constraints", "all_constraints", "no_constraints", "hi_constraints"
)){
  results<- matrix(
    0, nrow=length(counties), 
    # ncol=6 # [trpo]*[none, all]*[none, qhi, qhi_ot]
    ncol=18 # [trpo]*[6 forecast options]*[none, qhi, qhi_ot]
  )
  my_names<- rep("", ncol(results))
  
  for(k in 1:length(counties)){
    county<- counties[k]
    filename_zero_eval<- paste0("Summer_results/ORL_NA_eval_samp-R_obs-W_", r_model, "_fips_", county, ".csv")
    filename_zero_train<- paste0("Summer_results/ORL_NA_train_samp-R_samp-W_", r_model, "_fips_", county, ".csv")
    
    i<- 1 # column tracker
    for(algo in c( "trpo" #,
                   #  "dqn",
                   # "ppo"
    )){
      if(algo == "dqn"){
        forecast_list<- c("none", "all")
      }else{
        forecast_list<- c(
          # "none", "all",  
          "num_elig", "quarters", "three_day", "ten_day", "quantiles", "N_Av4_D3"
        )
      }
      for(forecasts in forecast_list){
        m<- paste0(r_model, "_", algo, "_F-", forecasts, "_fips-", county)
        results[k,i]<- eval_func(paste0("Summer_results/ORL_RL_eval_samp-R_obs-W_", m, "_fips_", county, ".csv"),
                                 filename_zero_eval)
        my_names[i]<- paste0(algo, "_F-", forecasts)
        i<- i+1
        Models<- paste0(r_model, "_", algo, "_F-", forecasts, "_Rstr-HI-", HI_thresholds, "_fips-", county)
        for(h in 1:length(Models)){
          m<- Models[h]
          train_samp<- eval_func(paste0("Summer_results/ORL_RL_train_samp-R_samp-W_", m, "_fips_", county, ".csv"),
                                 filename_zero_train)
          eval<- eval_func(paste0("Summer_results/ORL_RL_eval_samp-R_obs-W_", m, "_fips_", county, ".csv"),
                           filename_zero_eval)
          if(h == 1){
            Eval_samp<- train_samp
            Eval<- eval
            j<- 1
          }else{
            if(train_samp > Eval_samp){
              Eval_samp<- train_samp
              Eval<- eval
              j<- h
            }
          }
        }
        results[k,i]<- Eval
        my_names[i]<- paste0(algo, "_qhi_F-", forecasts)
        i<- i+1
        my_names[i]<- paste0("ot_", algo, "_qhi_F-", forecasts)
        results[k,i]<- HI_thresholds[j]
        i<- i+1
        print(forecasts)
      }
    }
    print(county)
  }
  DF<- round(data.frame(results),3)
  names(DF)<- my_names
  
  f<- paste0("Fall_results/RL_evals_", r_model, "_", eval_func_name, ".csv")
  if(file.exists(f)){
    old<- read.csv(f)[,-1]
    DF<- cbind(old, DF)
  }
  write.csv(DF, f)
  print(paste("Finished with", r_model))
}



### Inspect results:

for(r_model in c("mixed_constraints"
  # , "alert_constraints", "all_constraints", "no_constraints", "hi_constraints"
)){
  print(r_model)
  DF<- read.csv(paste0("Fall_results/RL_evals_", r_model, "_", eval_func_name, ".csv"))[,-1]
  # print(DF)
  bench_df<- read.csv(paste0("Fall_results/Benchmarks_", r_model, "_", eval_func_name, ".csv"))
  for(j in seq(1, ncol(DF))[-seq(3,ncol(DF),3)]){
    print(paste0(names(DF)[j], ": median_diff = ", round(median(DF[,j] - bench_df$NWS),3), ", WMW = ",
                wilcox.test(DF[,j], bench_df$NWS, paired = TRUE, alternative = "greater", exact=FALSE)$statistic))
    # print(paste(names(DF)[j], " =", round(mean(DF[,j] - bench_df$NWS),4)))
    # print(paste(names(DF)[j], " =", wilcox.test(DF[,j], bench_df$NWS, paired = TRUE, alternative = "greater", exact=FALSE)$statistic))
    # print(paste(names(DF)[j], " =", round(wilcox.test(DF[,j], bench_df$NWS, paired = TRUE, alternative = "greater", exact=FALSE)$p.value,8)))
  }
}


