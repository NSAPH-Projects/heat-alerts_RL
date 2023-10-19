
source("heat_alerts/scripts/Evaluation_functions.R")

library(stringr)

## Change manually:
eval_func_name<- "avg_return"
eval_func<- avg_return

#### Run evaluations:

r_model<- "mixed_constraints" 
HI_thresholds<- seq(0.5, 0.9, 0.05)
algo<- "trpo"

## First round:

results<- matrix(
  0, nrow=length(counties), 
  ncol=3 # [Eval][OT][Eval_samp]
)

for(k in 1:length(counties)){
  county<- counties[k]
  # Models<- paste0("Tune_F-", "Q_D10", "_Rstr-HI-", HI_thresholds, "_arch-2-32", "_ns-2048", "_fips-", county)
  Models<- list.files("Summer_results/", paste0("_arch-2-32", "_ns-2048", "_fips-", county, "_fips_", county, ".csv"))
  m_val<- Models[str_detect(Models, "samp-W")]
  m_test<- Models[str_detect(Models, "obs-W")]
  for(h in 1:length(m_val)){
    eval_samp<- eval_func(paste0("Summer_results/", m_val[h]))
    eval<- eval_func(paste0("Summer_results/", m_test[h]))
    if(h == 1){
      Eval_samp<- eval_samp
      Eval<- eval
      j<- 1
    }else{
      if(eval_samp > Eval_samp){
        Eval_samp<- eval_samp
        Eval<- eval
        j<- h
      }
    }
  }
  results[k,1]<- Eval
  results[k,2]<- HI_thresholds[j]
  results[k,3]<- Eval_samp
  print(county)
}

DF<- data.frame(results)
colnames(DF)<- c("Eval", "OT", "Eval_samp")
DF$County<- counties

write.csv(DF, "Fall_results/Main_analysis_batch1.csv", row.names=FALSE)


## All:

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


### Inspect results:

DF<- read.csv("Fall_results/Main_analysis_batch1.csv")
bench_df<- read.csv(paste0("Fall_results/Benchmarks_", r_model, "_", eval_func_name, ".csv"))

wmw<- wilcox.test(DF$Eval, bench_df$NWS, paired = TRUE, alternative = "greater", exact=FALSE)
print(paste0("Median_diff = ", round(median(DF$Eval - bench_df$NWS),3),
             ", WMW = ", wmw$statistic, ", p_value = ", round(wmw$p.value,5)))


