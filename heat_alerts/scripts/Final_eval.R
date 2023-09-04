
library(dplyr)

library(reticulate)
np<- import("numpy")

## Define functions:
n_days<- 153

my_proc<- function(filename){
  f<- file.exists(filename)
  if(f){
    df<- read.csv(filename)[,-1]
    df$Count = 1
    agg_df<- aggregate(. ~ Year, df, sum)
    agg_df$Budget<- agg_df$Budget/(n_days-1)
    agg_df$budget_frac<- agg_df$Actions/agg_df$Budget
    agg_df$Frac<- agg_df$Count/sum(agg_df$Count)
    estimated_reward<- sum(agg_df$Rewards*(1/nrow(agg_df))/agg_df$Frac)/1000
    # return(list(agg_df, estimated_reward))
    return(estimated_reward)
  }else{
    return(NA)
  }
}

assess<- function(filename){
  f<- file.exists(filename)
  if(f){
    df<- read.csv(filename)[,-1]
    n_eps<- nrow(df)/(n_days-1)
    Days<- rep(1:(n_days-1),n_eps)
    D<- Days[which(df$Actions == 1)]
    if(length(D)>0){
      num_alerts<- length(D)/n_eps
      summary_dos<- summary(D)
      diffs<- D[2:length(D)] - D[1:(length(D)-1)]
      L<- rle(diffs)
      streaks<- L$lengths[which(L$values == 1)]
      num_streaks<- length(streaks)/n_eps
      avg_streak_length<- mean(streaks + 1)
      avg_streak_length_overall<- mean(c(streaks + 1, rep(1,length(D)-length(streaks))))
      b_50<- mean(D[which(df$B_50 == 1)], na.rm=TRUE)
      b_80<- mean(D[which(df$B_80 == 1)], na.rm=TRUE)
      b_100<- mean(D[which(df$B_100 == 1)], na.rm=TRUE)
      above_thresh_skipped<- sum(df$Above_Thresh_Skipped)/n_eps
      fraction_skipped<- above_thresh_skipped / num_alerts
      # return(list(agg_df, estimated_reward))
      # x<- c(num_alerts, as.vector(summary_dos), num_streaks, avg_streak_length, avg_streak_length_overall)
      x<- c(num_alerts, summary_dos["Min."], b_50, b_80, b_100, 
            num_streaks, avg_streak_length, avg_streak_length_overall,
            above_thresh_skipped, fraction_skipped)
      result<- data.frame(t(x))
      # names(result)<- c("AvNAl", "Min_dos", "Q1_dos", "Median_dos", "Mean_dos", "Q3_dos", "Max_dos", "AvNStrk", "AvStrkLn", "AvStrkLn_all")
      names(result)<- c("AvNAl", "Min_dos", "B_50pct", "B_80pct", "B_last", 
                        "AvNStrk", "AvStrkLn", "AvStrkLn_all",
                        "Abv_Skp", "Frac_Abv_Skp")
      return(result)
    }else{
      return(rep(NA,10))
    }
  }else{
    return(NA)
  }
}


### Identify optimal HI threshold and save the associated eval:

prefix<- c("T8") # 
splitvar<- "Rstr-HI-"
these<- c("TRPO") # , "PPO", "DQN", "LSTM", "QRDQN"
Algo<- rep(these, 4)
type<- c("eval", "eval_samp", "train", "train_samp")
Type<- rep(type, each=length(these))

# counties<- c(41067, 53015, 20161, 37085, 48157,
#              28049, 19153, 17167, 31153, 6071, 4013)
# 
# counties<- c(34021, 19155, 17115, 29021, 29019, 5045, 40017, 21059,
#              47113, 42017, 22109, 45015, 13031, 48367, 22063, 41053,
#              32003, 4015, 6025)

counties<- c(41067, 53015, 20161, 37085, 48157,
             28049, 19153, 17167, 31153, 6071, 4013,
             34021, 19155, 17115, 29021, 29019, 5045, 40017, 21059,
             47113, 42017, 22109, 45015, 13031, 48367, 22063, 41053,
             32003, 4015, 6025)

HI_thresholds<- seq(0.5, 0.9, 0.05)
opt_HI_thr<- rep(0, length(counties))
Eval_samp<- rep(0, length(counties))
Eval<- rep(0, length(counties))
NWS<- rep(0, length(counties))
Random<- rep(0, length(counties))

for(k in 1:length(counties)){
  county<- counties[k]
  
  folders<- list.files("logs/SB", pattern=paste0(prefix, "_fips-", county))
  Models<- sapply(folders, function(s){strsplit(s, splitvar)[[1]][2]})
  Models<- paste0(splitvar, as.vector(unique(Models)))
  
  proc_NWS_eval<- my_proc(paste0("Summer_results/ORL_NWS_eval_samp-R_obs-W_test_fips_", county, ".csv"))
  proc_random_eval<- my_proc(paste0("Summer_results/ORL_random_eval_samp-R_obs-W_test_fips_", county, ".csv"))
  
  NWS[k]<- proc_NWS_eval
  Random[k]<- proc_random_eval
  
  for(i in 1:length(Models)){
    model<- Models[i]
    proc_TRPO_eval_samp<- my_proc(paste0("Summer_results/ORL_RL_eval_samp-R_samp-W_", prefix, "_fips-", county, "_", model, "_fips_", county, ".csv"))
    proc_TRPO_eval<- my_proc(paste0("Summer_results/ORL_RL_eval_samp-R_obs-W_", prefix, "_fips-", county, "_", model, "_fips_", county, ".csv"))
  
    if(i == 1){
      Eval_samp[k]<- proc_TRPO_eval_samp
      Eval[k]<- proc_TRPO_eval
      j<- 1
    }else{
      if(proc_TRPO_eval_samp > Eval_samp[k]){
        Eval_samp[k]<- proc_TRPO_eval_samp
        Eval[k]<- proc_TRPO_eval
        j<- i
      }
    }
  }
  opt_HI_thr[k]<- HI_thresholds[j]
  print(county) 
}

results<- data.frame(Fips=counties, Random, NWS, Eval, opt_HI_thr) # Eval_samp
results[,c("Random", "NWS", "Eval")]<- apply(results[,c("Random", "NWS", "Eval")],
                                                          MARGIN=2, function(x){round(x,3)})
results

write.csv(results, "Fall_results/Final_eval_30_T8.csv")

## Choosing best size of net_arch based on eval_samp:
T7_results<- read.csv("Fall_results/Final_eval_30_T7.csv")
T8_results<- read.csv("Fall_results/Final_eval_30_T8.csv")

Best_Model<- rep("", length(counties))
Best_Eval_Samp<- rep(0, length(counties))
Eval<- rep(0, length(counties))
opt_HI_thr<- rep(0, length(counties))
NWS_samp<- rep(0, length(counties))

for(k in 1:nrow(T7_results)){
  NWS_samp[k]<- my_proc(paste0("Summer_results/ORL_NWS_eval_samp-R_samp-W_test_fips_", T7_results[k,"Fips"], ".csv"))
  
  t7<- my_proc(paste0("Summer_results/ORL_RL_eval_samp-R_samp-W_", "T7", "_fips-", T7_results[k,"Fips"], "_", "Rstr-HI-", T7_results[k,"opt_HI_thr"], "_fips_", T7_results[k,"Fips"], ".csv"))
  t8<- my_proc(paste0("Summer_results/ORL_RL_eval_samp-R_samp-W_", "T8", "_fips-", T8_results[k,"Fips"], "_", "Rstr-HI-", T8_results[k,"opt_HI_thr"], "_fips_", T8_results[k,"Fips"], ".csv"))
  
  if(t8 > t7){
    Best_Model[k]<- "NN_2-16"
    Best_Eval_Samp[k]<- t8
    Eval[k]<- T8_results[k, "Eval"]
    opt_HI_thr[k]<- T8_results[k, "opt_HI_thr"]
  }else{
    Best_Model[k]<- "NN_1-16"
    Best_Eval_Samp[k]<- t7
    Eval[k]<- T7_results[k, "Eval"]
    opt_HI_thr[k]<- T7_results[k, "opt_HI_thr"]
  }
  print(T7_results[k,"Fips"])
}
results<- data.frame(T7_results[,c("Fips", "Random", "NWS")], 
                     Best_Model, Eval, opt_HI_thr, NWS_samp, Best_Eval_Samp)
results[,c("Random", "NWS", "Eval", "NWS_samp", "Best_Eval_Samp")]<- apply(results[,c("Random", "NWS", "Eval", "NWS_samp", "Best_Eval_Samp")],
                                             MARGIN=2, function(x){round(x,3)})
results
write.csv(results, "Fall_results/Final_eval_30_best-T7-T8.csv")

### Make table of alert issuance characteristics for the best models:

alerts_results<- data.frame(matrix(ncol = 10, nrow = 0))

for(k in 1:length(counties)){
  county<- counties[k]
  
  folders<- list.files("logs/SB", pattern=paste0(prefix, "_fips-", county))
  Models<- sapply(folders, function(s){strsplit(s, splitvar)[[1]][2]})
  Models<- paste0(splitvar, as.vector(unique(Models)))
  
  as_NWS_eval<- assess(paste0("Summer_results/ORL_NWS_eval_samp-R_obs-W_test_fips_", county, ".csv"))
  as_random_eval<- assess(paste0("Summer_results/ORL_random_eval_samp-R_obs-W_test_fips_", county, ".csv"))
  
  as_random_eval$Policy<- "random"
  as_NWS_eval$Policy<- "NWS"
  
  as_df<- rbind(as_random_eval, as_NWS_eval)
  
  if(opt_HI_thr[k] %% 0.1 == 0.1){
    rl<- assess(paste0("Summer_results/ORL_RL_eval_samp-R_obs-W_", prefix, "_fips-", county, "_", splitvar, round(opt_HI_thr[k],1), "_fips_", county, ".csv"))
  }else{
    rl<- assess(paste0("Summer_results/ORL_RL_eval_samp-R_obs-W_", prefix, "_fips-", county, "_", splitvar, opt_HI_thr[k], "_fips_", county, ".csv"))
  }
  rl$Policy<- "TRPO"
  as_df<- rbind(as_df, rl)
  
  alerts_results<- rbind(alerts_results, as_df)
  print(county) 
}

alerts_results$Fips<- rep(counties, each=3)
alerts_results[,c(12, 11, 1:10)]

### Making an expanded table to compare across experiments:
earlier<- read.csv("Fall_results/Final_eval_30_T7.csv")
earlier<- earlier[,c(2:4,6,5)]

# P_0<- rep(0, length(counties))
# Obs_W<- rep(0, length(counties))
NN_2l<- rep(0, length(counties))
Saved_Iter<- rep(0, length(counties))

for(k in 1:length(counties)){
  county<- counties[k]
  
  # P_0[k]<- my_proc(paste0("Summer_results/ORL_RL_eval_samp-R_obs-W_", "E1", "_fips-", county, "_P-0_Rstr-HI-opt", "_fips_", county, ".csv"))
  # Obs_W[k]<- my_proc(paste0("Summer_results/ORL_RL_eval_samp-R_obs-W_", "E1", "_fips-", county, "_obs-W_Rstr-HI-opt", "_fips_", county, ".csv"))
  NN_2l[k]<- my_proc(paste0("Summer_results/ORL_RL_eval_samp-R_obs-W_", "T7", "_fips-", county, "_Rstr-HI-opt_", "2-16", "_fips_", county, ".csv"))
  
  trpo_npz<- np$load(paste0("logs/SB/", "T7", "_fips-", county, "_Rstr-HI-", earlier$opt_HI_thr[k], "/results/evaluations.npz")) 
  trpo_iters<- trpo_npz$f[["timesteps"]]
  trpo_evals<- rowMeans(trpo_npz$f[["results"]])
  Saved_Iter[k]<- trpo_iters[which.max(trpo_evals)]
  
  print(county)
}

# results<- data.frame(earlier, Saved_Iter, P_0, Obs_W)
results<- data.frame(earlier, Saved_Iter, NN_2l)
results[,-1]<- apply(results[,-1], MARGIN=2, function(x){round(x,3)})
results



###### Baseline comparisons:

prefix<- "E0g1"
splitvar<-"_"  # "P-"

Eval<- data.frame(matrix(ncol = 2, nrow = length(counties)))
NWS<- rep(0, length(counties))
Random<- rep(0, length(counties))

for(k in 1:length(counties)){
  county<- counties[k]
  
  folders<- list.files("logs/SB", pattern=paste0(prefix, "_fips-", county))
  # Models<- sapply(folders, function(s){strsplit(s, splitvar)[[1]][2]})
  Models<- sapply(folders, function(s){strsplit(s, splitvar)[[1]][3]})
  # Models<- paste0(splitvar, as.vector(unique(Models)))
  Models<- as.vector(unique(Models))
  
  proc_NWS_eval<- my_proc(paste0("Summer_results/ORL_NWS_eval_samp-R_obs-W_test_fips_", county, ".csv"))
  proc_random_eval<- my_proc(paste0("Summer_results/ORL_random_eval_samp-R_obs-W_test_fips_", county, ".csv"))
  
  NWS[k]<- proc_NWS_eval
  Random[k]<- proc_random_eval
  
  for(i in 1:length(Models)){
    model<- Models[i]
    Eval[k,i]<- my_proc(paste0("Summer_results/ORL_RL_eval_samp-R_obs-W_", prefix, "_fips-", county, "_", model, "_fips_", county, ".csv"))
  }
  print(county)
}

results<- data.frame(Fips=counties, Random, NWS, Eval)
# names(results)[4:5]<- c("Eval_P-0", "Eval_P-0.01")
names(results)[4:5]<- c("Eval_NN-1-16", "Eval_NN-2-16")
results[,-1]<- apply(results[,-1], MARGIN=2, function(x){round(x,3)})
results


alerts_results<- data.frame(matrix(ncol = 10, nrow = 0))

for(k in 1:length(counties)){
  county<- counties[k]
  
  folders<- list.files("logs/SB", pattern=paste0(prefix, "_fips-", county))
  Models<- sapply(folders, function(s){strsplit(s, splitvar)[[1]][2]})
  Models<- paste0(splitvar, as.vector(unique(Models)))
  
  as_NWS_eval<- assess(paste0("Summer_results/ORL_NWS_eval_samp-R_obs-W_test_fips_", county, ".csv"))
  as_random_eval<- assess(paste0("Summer_results/ORL_random_eval_samp-R_obs-W_test_fips_", county, ".csv"))
  
  as_random_eval$Policy<- "random"
  as_NWS_eval$Policy<- "NWS"
  
  as_df<- rbind(as_random_eval, as_NWS_eval)
  
  for(m in Models){
    rl<- assess(paste0("Summer_results/ORL_RL_eval_samp-R_obs-W_", prefix, "_fips-", county, "_", m, "_fips_", county, ".csv"))
    rl$Policy<- "TRPO"
    as_df<- rbind(as_df, rl)
  }
  
  alerts_results<- rbind(alerts_results, as_df)
  print(county) 
}

alerts_results$Fips<- rep(counties, each=4)
alerts_results$Penalty<- rep(c(NA, NA, 0, 0.01), length(counties))
alerts_results[,c(12, 11, 13, 1:10)]


