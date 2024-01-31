
source("heat_alerts/scripts/Evaluation_functions.R")

library(stringr)

r_model<- "mixed_constraints" 


#### Run evaluations:

# algos<- c("trpo", "dqn")
algos<- c("a2c", "qrdqn")
prefix<- "December_part-2"

HI_thresholds<- seq(0.5, 0.9, 0.05)
forecasts<- c("none", "Q_D10")

NHU<- c(16, 32) 
NHL<- c(2, 3) 
n_steps<- c(1500, 3000) 

N<- length(counties)
A<- length(algos)
F<- length(forecasts)

############ Average-return metric:

eval_func<- avg_return

#### Plain (no QHI threshold):

results<- matrix(
  0, nrow=N*A*F, 
  ncol=7 # [algo][forecast][Eval][Eval_samp][NHL][NHU][n_steps]
)
results<- data.frame(results)
names(results)<- c("Algo", "Forecast", "Eval", "Eval_samp", "NHL", "NHU", "n_steps")

param_df<- expand.grid(NHL, NHU, n_steps)
names(param_df)<- c("NHL", "NHU", "n_steps")

for(k in 1:N){
  county<- counties[k]
  for(a in 1:A){
    for(f in 1:F){
      j<- 1 # keeping track of the best model
      Eval_samp<- eval_func(paste0("Summer_results/ORL_RL_eval_samp-R_samp-W_December_",
                                   algos[a], "_F-", forecasts[f], "_Rstr-HI-", "none",
                                   "_arch-", param_df$NHL[1], "-", param_df$NHU[1], "_ns-", param_df$n_steps[1],
                                   "_fips-", county, "_fips_", county, ".csv"))
      for(i in 2:nrow(param_df)){
        eval_samp<- eval_func(paste0("Summer_results/ORL_RL_eval_samp-R_samp-W_December_",
                                     algos[a], "_F-", forecasts[f], "_Rstr-HI-", "none",
                                     "_arch-", param_df$NHL[i], "-", param_df$NHU[i], "_ns-", param_df$n_steps[i],
                                     "_fips-", county, "_fips_", county, ".csv"))
        if(eval_samp > Eval_samp){
          j<- i
          Eval_samp<- eval_samp
        }
      }
      pos<- (k-1)*A*F + (a-1)*A + f
      results[pos, "Algo"]<- algos[a]
      results[pos, "Forecast"]<- forecasts[f]
      results[pos, "Eval"]<- eval_func(paste0("Summer_results/ORL_RL_eval_samp-R_obs-W_December_",
                                              algos[a], "_F-", forecasts[f], "_Rstr-HI-", "none",
                                              "_arch-", param_df$NHL[j], "-", param_df$NHU[j], "_ns-", param_df$n_steps[j],
                                              "_fips-", county, "_fips_", county, ".csv"))
      results[pos, "Eval_samp"]<- Eval_samp
      results[pos, "NHL"]<- param_df$NHL[j]
      results[pos, "NHU"]<- param_df$NHU[j]
      results[pos, "n_steps"]<- param_df$n_steps[j]
    }
  }
  print(county)
}

results$County<- rep(counties, each=A*F)
write.csv(results, paste0("Fall_results/", prefix, "_plain_RL_avg_return.csv"), row.names=FALSE)

DF<- read.csv(paste0("Fall_results/", prefix, "_plain_RL_avg_return.csv"))
bench_df<- read.csv(paste0("Fall_results/Benchmarks_", r_model, "_avg_return", ".csv"))

for(algo in algos){
  for(forecast in forecasts){
    df<- DF[which(DF$Algo == algo & DF$Forecast == forecast),]
    wmw<- wilcox.test(df$Eval, bench_df$NWS, paired = TRUE, alternative = "greater", exact=FALSE)
    print(paste0(algo, ", ", forecast, ": Median_diff = ", round(median(df$Eval - bench_df$NWS),3),
                 ", WMW = ", wmw$statistic, ", p_value = ", round(wmw$p.value,5)))
  }
}


#### With QHI threshold:

results<- matrix(
  0, nrow=length(counties), 
  ncol=8 # [algo][forecast][Eval][Eval_samp][OT][NHL][NHU][n_steps]
)
results<- data.frame(results)
names(results)<- c("Algo", "Forecast", "Eval", "Eval_samp", "OT", "NHL", "NHU", "n_steps")

param_df<- expand.grid(NHL, NHU, n_steps, HI_thresholds)
names(param_df)<- c("NHL", "NHU", "n_steps", "HI")

for(k in 1:N){
  county<- counties[k]
  for(a in 1:A){
    for(f in 1:F){
      j<- 1 # keeping track of the best model
      Eval_samp<- eval_func(paste0("Summer_results/ORL_RL_eval_samp-R_samp-W_December_",
                                   algos[a], "_F-", forecasts[f], "_Rstr-HI-", param_df$HI[1],
                                   "_arch-", param_df$NHL[1], "-", param_df$NHU[1], "_ns-", param_df$n_steps[1],
                                   "_fips-", county, "_fips_", county, ".csv"))
      for(i in 2:nrow(param_df)){
        eval_samp<- eval_func(paste0("Summer_results/ORL_RL_eval_samp-R_samp-W_December_",
                                     algos[a], "_F-", forecasts[f], "_Rstr-HI-", param_df$HI[i],
                                     "_arch-", param_df$NHL[i], "-", param_df$NHU[i], "_ns-", param_df$n_steps[i],
                                     "_fips-", county, "_fips_", county, ".csv"))
        if(eval_samp > Eval_samp){
          j<- i
          Eval_samp<- eval_samp
        }
      }
      pos<- (k-1)*A*F + (a-1)*A + f
      results[pos, "Algo"]<- algos[a]
      results[pos, "Forecast"]<- forecasts[f]
      results[pos, "Eval"]<- eval_func(paste0("Summer_results/ORL_RL_eval_samp-R_obs-W_December_",
                                              algos[a], "_F-", forecasts[f], "_Rstr-HI-", param_df$HI[j],
                                              "_arch-", param_df$NHL[j], "-", param_df$NHU[j], "_ns-", param_df$n_steps[j],
                                              "_fips-", county, "_fips_", county, ".csv"))
      results[pos, "Eval_samp"]<- Eval_samp
      results[pos, "OT"]<- param_df$HI[j]
      results[pos, "NHL"]<- param_df$NHL[j]
      results[pos, "NHU"]<- param_df$NHU[j]
      results[pos, "n_steps"]<- param_df$n_steps[j]
    }
  }
  print(county)
}


results$County<- rep(counties, each=A*F)
write.csv(results, paste0("Fall_results/", prefix, "_Rstr-QHI_RL_avg_return.csv"), row.names=FALSE)

DF<- read.csv(paste0("Fall_results/", prefix, "_Rstr-QHI_RL_avg_return.csv"))
bench_df<- read.csv(paste0("Fall_results/Benchmarks_", r_model, "_avg_return", ".csv"))

for(algo in algos){
  for(forecast in forecasts){
    df<- DF[which(DF$Algo == algo & DF$Forecast == forecast),]
    wmw<- wilcox.test(df$Eval, bench_df$NWS, paired = TRUE, alternative = "greater", exact=FALSE)
    print(paste0(algo, ", ", forecast, ": Median_diff = ", round(median(df$Eval - bench_df$NWS),3),
                 ", WMW = ", wmw$statistic, ", p_value = ", round(wmw$p.value,5)))
  }
}



############ Per-alert metric:

eval_func<- compare_to_zero

#### Plain (no QHI threshold):

AR_DF<- read.csv("Fall_results/December_plain_RL_avg_return.csv")

results<- rep(0, N*A*F)

i<- 1
for(k in 1:N){
  for(a in 1:A){
    for(f in 1:F){
      county<- AR_DF[i, "County"]
      results[i]<- eval_func(paste0("Summer_results/ORL_RL_eval_samp-R_obs-W_December_",
                                    AR_DF[i, "Algo"], "_F-", AR_DF[i, "Forecast"], "_Rstr-HI-", "none", 
                                    "_arch-", AR_DF[i, "NHL"], "-", AR_DF[i, "NHU"],
                                    "_ns-", AR_DF[i, "n_steps"], "_fips-", county, "_fips_", county, ".csv"),
                             paste0("Summer_results/ORL_NA_eval_samp-R_obs-W_", r_model, "_fips_", county, ".csv"))
      print(county)
      i<- i+1
    }
  }
}

Results<- data.frame(Eval=results, Algo=AR_DF$Algo, Forecast=AR_DF$Forecast)
write.csv(Results, "Fall_results/December_plain_RL_compare_to_zero.csv", row.names=FALSE)

DF<- read.csv("Fall_results/December_plain_RL_compare_to_zero.csv")
bench_df<- read.csv(paste0("Fall_results/Benchmarks_", r_model, "_compare_to_zero", ".csv"))

for(algo in algos){
  for(forecast in forecasts){
    df<- DF[which(DF$Algo == algo & DF$Forecast == forecast),]
    df[which(is.na(df$Eval)), "Eval"]<- 0
    wmw<- wilcox.test(df$Eval, bench_df$NWS, paired = TRUE, alternative = "greater", exact=FALSE)
    print(paste0(algo, ", ", forecast, ": Median_diff = ", round(median(df$Eval - bench_df$NWS),3),
                 ", WMW = ", wmw$statistic, ", p_value = ", round(wmw$p.value,5)))
  }
}

#### With QHI threshold:

AR_DF<- read.csv("Fall_results/December_Rstr-QHI_RL_avg_return.csv")

results<- rep(0, N*A*F)

i<- 1
for(k in 1:N){
  for(a in 1:A){
    for(f in 1:F){
      county<- AR_DF[i, "County"]
      results[i]<- eval_func(paste0("Summer_results/ORL_RL_eval_samp-R_obs-W_December_", 
                                    AR_DF[i, "Algo"], "_F-", AR_DF[i, "Forecast"], "_Rstr-HI-", AR_DF[i, "OT"], 
                                    "_arch-", AR_DF[i, "NHL"], "-", AR_DF[i, "NHU"],
                                    "_ns-", AR_DF[i, "n_steps"], "_fips-", county, "_fips_", county, ".csv"),
                             paste0("Summer_results/ORL_NA_eval_samp-R_obs-W_", r_model, "_fips_", county, ".csv"))
      print(county)
      i<- i+1
    }
  }
}

Results<- data.frame(Eval=results, Algo=AR_DF$Algo, Forecast=AR_DF$Forecast)
write.csv(Results, "Fall_results/December_Rstr-QHI_RL_compare_to_zero.csv", row.names=FALSE)


DF<- read.csv("Fall_results/December_Rstr-QHI_RL_compare_to_zero.csv")
bench_df<- read.csv(paste0("Fall_results/Benchmarks_", r_model, "_compare_to_zero", ".csv"))

for(algo in algos){
  for(forecast in forecasts){
    df<- DF[which(DF$Algo == algo & DF$Forecast == forecast),]
    df[which(is.na(df$Eval)), "Eval"]<- 0
    wmw<- wilcox.test(df$Eval, bench_df$NWS, paired = TRUE, alternative = "greater", exact=FALSE)
    print(paste0(algo, ", ", forecast, ": Median_diff = ", round(median(df$Eval - bench_df$NWS),3),
                 ", WMW = ", wmw$statistic, ", p_value = ", round(wmw$p.value,5)))
  }
}

################################################ OLD:

# ## First round:
# 
# results<- matrix(
#   0, nrow=length(counties), 
#   ncol=3 # [Eval][OT][Eval_samp]
# )
# 
# for(k in 1:length(counties)){
#   county<- counties[k]
#   # Models<- paste0("Tune_F-", "Q_D10", "_Rstr-HI-", HI_thresholds, "_arch-2-32", "_ns-2048", "_fips-", county)
#   Models<- list.files("Summer_results/", paste0("_arch-2-32", "_ns-2048", "_fips-", county, "_fips_", county, ".csv"))
#   m_val<- Models[str_detect(Models, "samp-W")]
#   m_test<- Models[str_detect(Models, "obs-W")]
#   for(h in 1:length(m_val)){
#     eval_samp<- eval_func(paste0("Summer_results/", m_val[h]))
#     eval<- eval_func(paste0("Summer_results/", m_test[h]))
#     if(h == 1){
#       Eval_samp<- eval_samp
#       Eval<- eval
#       j<- 1
#     }else{
#       if(eval_samp > Eval_samp){
#         Eval_samp<- eval_samp
#         Eval<- eval
#         j<- h
#       }
#     }
#   }
#   results[k,1]<- Eval
#   results[k,2]<- HI_thresholds[j]
#   results[k,3]<- Eval_samp
#   print(county)
# }
# 
# DF<- data.frame(results)
# colnames(DF)<- c("Eval", "OT", "Eval_samp")
# DF$County<- counties
# 
# write.csv(DF, "Fall_results/Main_analysis_batch1.csv", row.names=FALSE)


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
write.csv(results, paste0("Fall_results/NEW_Main_analysis_trpo_F-", forecasts, ".csv"), row.names=FALSE)


### Inspect results:

# DF<- read.csv("Fall_results/Main_analysis_batch1.csv")
DF<- read.csv("Fall_results/NEW_Main_analysis_trpo_F-Q_D10.csv")
DF<- read.csv("Fall_results/NEW_Main_analysis_trpo_F-none.csv")
bench_df<- read.csv(paste0("Fall_results/Benchmarks_", r_model, "_avg_return", ".csv"))

wmw<- wilcox.test(DF$Eval, bench_df$NWS, paired = TRUE, alternative = "greater", exact=FALSE)
print(paste0("Median_diff = ", round(median(DF$Eval - bench_df$NWS),3),
             ", WMW = ", wmw$statistic, ", p_value = ", round(wmw$p.value,5)))

wmw<- wilcox.test(DF$Eval, bench_df$AA_QHI, paired = TRUE, alternative = "greater", exact=FALSE)
print(paste0("Median_diff = ", round(median(DF$Eval - bench_df$AA_QHI),3),
             ", WMW = ", wmw$statistic, ", p_value = ", round(wmw$p.value,5)))

other_DF<- read.csv("Fall_results/NEW_Other_algos_F-none.csv")
# new<- other_DF$Eval_ppo
# wmw<- wilcox.test(new, bench_df$NWS, paired = TRUE, alternative = "greater", exact=FALSE)
# print(paste0("Median_diff = ", round(median(new - bench_df$NWS),3),
#              ", WMW = ", wmw$statistic, ", p_value = ", round(wmw$p.value,5)))
new<- other_DF$Eval_dqn
wmw<- wilcox.test(new, bench_df$NWS, paired = TRUE, alternative = "greater", exact=FALSE)
print(paste0("Median_diff = ", round(median(new - bench_df$NWS),3),
             ", WMW = ", wmw$statistic, ", p_value = ", round(wmw$p.value,5)))


################# Now look at per alert (compared to Zero) metric:

eval_func_name<- "compare_to_zero"
eval_func<- compare_to_zero

RL_df<- read.csv("Fall_results/NEW_Main_analysis_trpo_F-Q_D10.csv")
forecasts<- "Q_D10"
RL_df<- read.csv("Fall_results/NEW_Main_analysis_trpo_F-none.csv")
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

write.csv(results, paste0("Fall_results/NEW_Main_analysis_trpo_F-", forecasts, "_", eval_func_name, ".csv"), row.names=FALSE)

#### Inspect:

DF<- read.csv("Fall_results/NEW_Main_analysis_trpo_F-Q_D10_compare_to_zero.csv")
DF<- read.csv("Fall_results/NEW_Main_analysis_trpo_F-none_compare_to_zero.csv")
DF$x[which(is.na(DF$x))]<- 0
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
  # ncol=4 # [ppo, dqn]*[eval_qhi, qhi_ot]
  ncol=2 # [dqn]*[eval_qhi, qhi_ot]
)

results<- data.frame(results)
# names(results)<- paste0(c("Eval_", "OT_"), c(rep("ppo", 2), rep("dqn", 2)))
names(results)<- paste0(c("Eval_", "OT_"),  rep("dqn", 2))

for(algo in c("dqn"
              # , "ppo"
    )){
  
  files_eval<- list.files("Summer_results", paste0("eval_samp-R_obs-W_", algo, "_F-none_Rstr-HI-0"))
  files_ES<- list.files("Summer_results", paste0("eval_samp-R_samp-W_", algo, "_F-none_Rstr-HI-0"))

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

write.csv(results, paste0("Fall_results/NEW_Other_algos_F-", "none", ".csv"), row.names=FALSE)


#### Plain (no QHI) results:

results<- matrix(
  0, nrow=length(counties), 
  ncol=2 # [dqn, trpo] # ppo
)

algos<- c("trpo", "dqn") # "ppo"

results<- data.frame(results)
names(results)<- paste0("Eval_", algos)

# eval_func<- compare_to_zero

for(algo in algos){
  all_files<- list.files("Summer_results", paste0(algo, "_F-none", "_Rstr-HI-none"))
  half<- length(all_files)/2
  files_eval<- all_files[1:half]
  files_ES<- all_files[(half+1):length(all_files)]
  
  for(k in 1:length(counties)){
    county<- counties[k]
    county_files<- str_detect(files_eval, as.character(county))
    results[k, paste0("Eval_", algo)]<- eval_func(paste0("Summer_results/", files_eval[county_files]))
    # results[k, paste0("Eval_", algo)]<- eval_func(paste0("Summer_results/", files_eval[county_files]),
    #                               paste0("Summer_results/ORL_NA_eval_samp-R_obs-W_", r_model, "_fips_", county, ".csv"))
    
    print(paste0(algo,": ", county))
  }
}

write.csv(results, paste0("Fall_results/NEW_No-QHI_F-", "none", ".csv"), row.names=FALSE)

DF<- results
DF$Eval_trpo[which(is.na(DF$Eval_trpo))]<- 0
DF$Eval_dqn[which(is.na(DF$Eval_dqn))]<- 0


