
source("heat_alerts/scripts/Evaluation_functions.R")

library(stringr)

r_model<- "mixed_constraints" 


#### Run evaluations:

# algos<- c("trpo", "dqn")
algos<- c("a2c", "qrdqn")
read_prefix<- "December"
write_prefix<- "December_part-2"

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
      Eval_samp<- eval_func(paste0("Summer_results/ORL_RL_eval_samp-R_samp-W_", read_prefix, "_",
                                   algos[a], "_F-", forecasts[f], "_Rstr-HI-", "none",
                                   "_arch-", param_df$NHL[1], "-", param_df$NHU[1], "_ns-", param_df$n_steps[1],
                                   "_fips-", county, "_fips_", county, ".csv"))
      for(i in 2:nrow(param_df)){
        eval_samp<- eval_func(paste0("Summer_results/ORL_RL_eval_samp-R_samp-W_", read_prefix, "_",
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
      results[pos, "Eval"]<- eval_func(paste0("Summer_results/ORL_RL_eval_samp-R_obs-W_", read_prefix, "_",
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
write.csv(results, paste0("Fall_results/", write_prefix, "_plain_RL_avg_return.csv"), row.names=FALSE)

DF<- read.csv(paste0("Fall_results/", write_prefix, "_plain_RL_avg_return.csv"))
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
      Eval_samp<- eval_func(paste0("Summer_results/ORL_RL_eval_samp-R_samp-W_", read_prefix, "_",
                                   algos[a], "_F-", forecasts[f], "_Rstr-HI-", param_df$HI[1],
                                   "_arch-", param_df$NHL[1], "-", param_df$NHU[1], "_ns-", param_df$n_steps[1],
                                   "_fips-", county, "_fips_", county, ".csv"))
      for(i in 2:nrow(param_df)){
        eval_samp<- eval_func(paste0("Summer_results/ORL_RL_eval_samp-R_samp-W_", read_prefix, "_",
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
      results[pos, "Eval"]<- eval_func(paste0("Summer_results/ORL_RL_eval_samp-R_obs-W_", read_prefix, "_",
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
write.csv(results, paste0("Fall_results/", write_prefix, "_Rstr-QHI_RL_avg_return.csv"), row.names=FALSE)

DF<- read.csv(paste0("Fall_results/", write_prefix, "_Rstr-QHI_RL_avg_return.csv"))
bench_df<- read.csv(paste0("Fall_results/Benchmarks_", r_model, "_avg_return", ".csv"))

for(algo in algos){
  for(forecast in forecasts){
    df<- DF[which(DF$Algo == algo & DF$Forecast == forecast),]
    wmw<- wilcox.test(df$Eval, bench_df$NWS, paired = TRUE, alternative = "greater", exact=FALSE)
    print(paste0(algo, ", ", forecast, ": Median_diff = ", round(median(df$Eval - bench_df$NWS),3),
                 ", WMW = ", wmw$statistic, ", p_value = ", round(wmw$p.value,5)))
  }
}

