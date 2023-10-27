
source("heat_alerts/scripts/Evaluation_functions.R")

library(stringr)

r_model<- "mixed_constraints" 

## Change manually:
eval_func_name<- "avg_return"
eval_func<- avg_return

#### Run evaluations:

algo<- "trpo"
HI_thresholds<- seq(0.5, 0.9, 0.05)
# forecasts<- c("Q_D10")
forecasts<- c("none")

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

NHU<- c(16, 32, 64)
NHL<- c(1, 2, 3)
n_steps<- c(1024, 2048, 4096)

results<- matrix(
  0, nrow=length(counties), 
  ncol=6 # [Eval][Eval_samp][OT][NHL][NHU][n_steps]
)
results<- data.frame(results)
names(results)<- c("Eval", "Eval_samp", "OT", "NHL", "NHU", "n_steps")

param_df<- expand.grid(NHL, NHU, n_steps, HI_thresholds, forecasts)
names(param_df)<- c("NHL", "NHU", "n_steps", "HI", "forecast")

for(k in 1:length(counties)){ 
  county<- counties[k]
  i<- 1 # keeping track of row in param_df
  j<- 1 # keeping track of best model
  Eval_samp<- eval_func(paste0("Summer_results/ORL_RL_eval_samp-R_samp-W_Tune_F-", 
                          param_df$forecast[i], "_Rstr-HI-", param_df$HI[i],
                          "_arch-", param_df$NHL[i], "-", param_df$NHU[i],
                          "_ns-", param_df$n_steps[i], "_fips-", county, "_fips_", county, ".csv"))
  for(i in 2:nrow(param_df)){
    eval_samp<- eval_func(paste0("Summer_results/ORL_RL_eval_samp-R_samp-W_Tune_F-", 
                                 param_df$forecast[i], "_Rstr-HI-", param_df$HI[i],
                                 "_arch-", param_df$NHL[i], "-", param_df$NHU[i],
                                 "_ns-", param_df$n_steps[i], "_fips-", county, "_fips_", county, ".csv"))
    if(eval_samp > Eval_samp){
      j<- i
      Eval_samp<- eval_samp
    }
    
    if(i %% 50 == 0){
      print(paste0(county, ": ", i))
    }
  }
  results[k, "Eval"]<- eval_func(paste0("Summer_results/ORL_RL_eval_samp-R_obs-W_Tune_F-", 
                                        param_df$forecast[j], "_Rstr-HI-", param_df$HI[j],
                                        "_arch-", param_df$NHL[j], "-", param_df$NHU[j],
                                        "_ns-", param_df$n_steps[j], "_fips-", county, "_fips_", county, ".csv"))
  results[k, "Eval_samp"]<- Eval_samp
  results[k, "OT"]<- param_df$HI[j]
  results[k, "NHL"]<- param_df$NHL[j]
  results[k, "NHU"]<- param_df$NHU[j]
  results[k, "n_steps"]<- param_df$n_steps[j]
  
}

results$County<- counties
write.csv(results, paste0("Fall_results/Main_analysis_trpo_F-", forecasts, ".csv"), row.names=FALSE)


### Inspect results:

# DF<- read.csv("Fall_results/Main_analysis_batch1.csv")
DF<- read.csv("Fall_results/Main_analysis_trpo_F-Q-D10.csv")
DF<- read.csv("Fall_results/Main_analysis_trpo_F-none.csv")
bench_df<- read.csv(paste0("Fall_results/Benchmarks_", r_model, "_", eval_func_name, ".csv"))

wmw<- wilcox.test(DF$Eval, bench_df$NWS, paired = TRUE, alternative = "greater", exact=FALSE)
print(paste0("Median_diff = ", round(median(DF$Eval - bench_df$NWS),3),
             ", WMW = ", wmw$statistic, ", p_value = ", round(wmw$p.value,5)))

wmw<- wilcox.test(DF$Eval, bench_df$AA_QHI, paired = TRUE, alternative = "greater", exact=FALSE)
print(paste0("Median_diff = ", round(median(DF$Eval - bench_df$AA_QHI),3),
             ", WMW = ", wmw$statistic, ", p_value = ", round(wmw$p.value,5)))

other_DF<- read.csv("Fall_results/Other_algos_F-none.csv")
new<- other_DF$Eval_ppo
wmw<- wilcox.test(new, bench_df$NWS, paired = TRUE, alternative = "greater", exact=FALSE)
print(paste0("Median_diff = ", round(median(new - bench_df$NWS),3),
             ", WMW = ", wmw$statistic, ", p_value = ", round(wmw$p.value,5)))
new<- other_DF$Eval_dqn
wmw<- wilcox.test(new, bench_df$NWS, paired = TRUE, alternative = "greater", exact=FALSE)
print(paste0("Median_diff = ", round(median(new - bench_df$NWS),3),
             ", WMW = ", wmw$statistic, ", p_value = ", round(wmw$p.value,5)))


################# Now look at per alert (compared to Zero) metric:

eval_func_name<- "compare_to_zero"
eval_func<- compare_to_zero

RL_df<- read.csv("Fall_results/Main_analysis_trpo_F-Q-D10.csv")
forecasts<- "Q_D10"
RL_df<- read.csv("Fall_results/Main_analysis_trpo_F-none.csv")
forecasts<- "none"

results<- rep(0, length(counties))

for(k in counties){
  i<- which(counties == k)

  results[i]<- eval_func(paste0("Summer_results/ORL_RL_eval_samp-R_obs-W_", 
                  "Tune_F-", forecasts, "_Rstr-HI-", RL_df[i, "OT"], 
                  "_arch-", RL_df[i, "NHL"], "-", RL_df[i, "NHU"],
                  "_ns-", RL_df[i, "n_steps"], "_fips-", k, "_fips_", k, ".csv"),
            paste0("Summer_results/ORL_NA_eval_samp-R_obs-W_", r_model, "_fips_", k, ".csv"))
  print(k)
}

write.csv(results, paste0("Fall_results/Main_analysis_trpo_F-", forecasts, "_", eval_func_name, ".csv"), row.names=FALSE)

#### Inspect:

DF<- read.csv("Fall_results/Main_analysis_trpo_F-Q_D10_compare_to_zero.csv")
DF<- read.csv("Fall_results/Main_analysis_trpo_F-none_compare_to_zero.csv")
bench_df<- read.csv(paste0("Fall_results/Benchmarks_", "mixed_constraints", "_", "compare_to_zero", ".csv"))

wmw<- wilcox.test(DF$x, bench_df$NWS, paired = TRUE, alternative = "greater", exact=FALSE)
print(paste0("Median_diff = ", round(median(DF$x - bench_df$NWS),3),
             ", WMW = ", wmw$statistic, ", p_value = ", round(wmw$p.value,5)))

wmw<- wilcox.test(DF$x, bench_df$AA_QHI, paired = TRUE, alternative = "greater", exact=FALSE)
print(paste0("Median_diff = ", round(median(DF$x - bench_df$AA_QHI),3),
             ", WMW = ", wmw$statistic, ", p_value = ", round(wmw$p.value,5)))


######## Comparison algos:

results<- matrix(
  0, nrow=length(counties), 
  ncol=4 # [ppo, dqn]*[eval_qhi, qhi_ot]
)

results<- data.frame(results)
names(results)<- paste0(c("Eval_", "OT_"), c(rep("ppo", 2), rep("dqn", 2)))

for(algo in c("dqn", "ppo")){
  all_files<- list.files("Summer_results", algo)
  half<- length(all_files)/2
  files_eval<- all_files[1:half]
  files_ES<- all_files[(half+1):length(all_files)]

  for(k in 1:length(counties)){
    county<- counties[k]
    county_files<- str_detect(files_eval, as.character(county))
    Eval_samp<- eval_func(paste0("Summer_results/", files_ES[county_files][1]))
    j<-1
    for(h in 2:length(HI_thresholds)){
      eval_samp<- eval_func(paste0("Summer_results/", files_ES[county_files][h]))
      if(eval_samp > Eval_samp){
        j<- h
        Eval_samp<- eval_samp
      }
    }
    results[k, paste0("Eval_", algo)]<- eval_func(paste0("Summer_results/", files_eval[county_files][j]))
    results[k, paste0("OT_", algo)]<- HI_thresholds[j]
    print(paste0(algo,": ", county))
  }
}

write.csv(results, paste0("Fall_results/Other_algos_F-", "none", ".csv"), row.names=FALSE)







