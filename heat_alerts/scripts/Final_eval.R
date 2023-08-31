
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

prefix<- c("T5")
splitvar<- "Rstr-HI-"
these<- c("TRPO") # , "PPO", "DQN", "LSTM", "QRDQN"
Algo<- rep(these, 4)
type<- c("eval", "eval_samp", "train", "train_samp")
Type<- rep(type, each=length(these))

counties<- c(41067, 53015, 20161, 37085, 48157,
             28049, 19153, 17167, 31153, 6071, 4013)

counties<- c(34021, 19155, 17115, 29021, 29019, 5045, 40017, 21059,
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

write.csv(results, "Fall_results/Final_eval_30.csv")

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


