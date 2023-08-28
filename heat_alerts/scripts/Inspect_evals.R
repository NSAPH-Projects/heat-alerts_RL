
library(dplyr)


library(reticulate)
np<- import("numpy")


## Manually change:
county<- 36005 

prefix<- c("T3")
folders<- list.files("logs/SB", pattern=paste0(prefix, "_fips-", county))
splitvar<- "Rstr-HI-" # "PD"
Models<- sapply(folders, function(s){strsplit(s, splitvar)[[1]][2]})
Models<- paste0(splitvar, as.vector(unique(Models)))

# Models[5:6]<- paste0("Rstr-HI_", Models[5:6])
# Models<- Models[-1]

these<- c("TRPO", "PPO", "DQN", "LSTM") # "LSTM", "QRDQN"
Algo<- rep(these, 4)
type<- c("eval", "eval_samp", "train", "train_samp")
Type<- rep(type, each=length(these))

## Run from here:
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

## First to get the estimated rewards:

proc_NWS_eval_samp<- my_proc(paste0("Summer_results/ORL_NWS_eval_samp-R_samp-W_test_fips_", county, ".csv"))
proc_NWS_train_samp<- my_proc(paste0("Summer_results/ORL_NWS_train_samp-R_samp-W_test_fips_", county, ".csv"))
proc_NWS_eval<- my_proc(paste0("Summer_results/ORL_NWS_eval_samp-R_obs-W_test_fips_", county, ".csv"))
proc_NWS_train<- my_proc(paste0("Summer_results/ORL_NWS_train_samp-R_obs-W_test_fips_", county, ".csv"))

proc_NA_eval_samp<- my_proc(paste0("Summer_results/ORL_NA_eval_samp-R_samp-W_test_fips_", county, ".csv"))
proc_NA_train_samp<- my_proc(paste0("Summer_results/ORL_NA_train_samp-R_samp-W_test_fips_", county, ".csv"))
proc_NA_eval<- my_proc(paste0("Summer_results/ORL_NA_eval_samp-R_obs-W_test_fips_", county, ".csv"))
proc_NA_train<- my_proc(paste0("Summer_results/ORL_NA_train_samp-R_obs-W_test_fips_", county, ".csv"))

proc_random_eval_samp<- my_proc(paste0("Summer_results/ORL_random_eval_samp-R_samp-W_test_fips_", county, ".csv"))
proc_random_train_samp<- my_proc(paste0("Summer_results/ORL_random_train_samp-R_samp-W_test_fips_", county, ".csv"))
proc_random_eval<- my_proc(paste0("Summer_results/ORL_random_eval_samp-R_obs-W_test_fips_", county, ".csv"))
proc_random_train<- my_proc(paste0("Summer_results/ORL_random_train_samp-R_obs-W_test_fips_", county, ".csv"))

proc_na<- round(c(proc_NA_eval, proc_NA_eval_samp, proc_NA_train, proc_NA_train_samp),3)
proc_random<- round(c(proc_random_eval, proc_random_eval_samp, proc_random_train, proc_random_train_samp),3)
proc_nws<- round(c(proc_NWS_eval, proc_NWS_eval_samp, proc_NWS_train, proc_NWS_train_samp),3)

proc_na
proc_random
proc_nws

### Make a table, updated:
proc_Random<- rep(proc_random, each=length(these))
proc_NWS<- rep(proc_nws, each=length(these))

for(model in Models){
  
  ## Read in data and calculate estimated rewards:
  proc_PPO_eval_samp<- my_proc(paste0("Summer_results/ORL_RL_eval_samp-R_samp-W_", prefix, "_fips-", county,"_ppo_", model, "_fips_", county, ".csv"))
  proc_PPO_train_samp<- my_proc(paste0("Summer_results/ORL_RL_train_samp-R_samp-W_", prefix, "_fips-", county,"_ppo_", model, "_fips_", county, ".csv"))
  proc_PPO_eval<- my_proc(paste0("Summer_results/ORL_RL_eval_samp-R_obs-W_", prefix, "_fips-", county,"_ppo_", model, "_fips_", county, ".csv"))
  proc_PPO_train<- my_proc(paste0("Summer_results/ORL_RL_train_samp-R_obs-W_", prefix, "_fips-", county,"_ppo_", model, "_fips_", county, ".csv"))
  
  proc_TRPO_eval_samp<- my_proc(paste0("Summer_results/ORL_RL_eval_samp-R_samp-W_", prefix, "_fips-", county,"_trpo_", model, "_fips_", county, ".csv"))
  proc_TRPO_train_samp<- my_proc(paste0("Summer_results/ORL_RL_train_samp-R_samp-W_", prefix, "_fips-", county,"_trpo_", model, "_fips_", county, ".csv"))
  proc_TRPO_eval<- my_proc(paste0("Summer_results/ORL_RL_eval_samp-R_obs-W_", prefix, "_fips-", county,"_trpo_", model, "_fips_", county, ".csv"))
  proc_TRPO_train<- my_proc(paste0("Summer_results/ORL_RL_train_samp-R_obs-W_", prefix, "_fips-", county,"_trpo_", model, "_fips_", county, ".csv"))
  
  proc_DQN_eval_samp<- my_proc(paste0("Summer_results/ORL_RL_eval_samp-R_samp-W_", prefix, "_fips-", county,"_dqn_", model, "_fips_", county, ".csv"))
  proc_DQN_train_samp<- my_proc(paste0("Summer_results/ORL_RL_train_samp-R_samp-W_", prefix, "_fips-", county,"_dqn_", model, "_fips_", county, ".csv"))
  proc_DQN_eval<- my_proc(paste0("Summer_results/ORL_RL_eval_samp-R_obs-W_", prefix, "_fips-", county,"_dqn_", model, "_fips_", county, ".csv"))
  proc_DQN_train<- my_proc(paste0("Summer_results/ORL_RL_train_samp-R_obs-W_", prefix, "_fips-", county,"_dqn_", model, "_fips_", county, ".csv"))
  
  # proc_QRDQN_eval_samp<- my_proc(paste0("Summer_results/ORL_RL_eval_samp-R_samp-W_", prefix, "_fips-", county,"_qrdqn_", model, "_fips_", county, ".csv"))
  # proc_QRDQN_train_samp<- my_proc(paste0("Summer_results/ORL_RL_train_samp-R_samp-W_", prefix, "_fips-", county,"_qrdqn_", model, "_fips_", county, ".csv"))
  # proc_QRDQN_eval<- my_proc(paste0("Summer_results/ORL_RL_eval_samp-R_obs-W_", prefix, "_fips-", county,"_qrdqn_", model, "_fips_", county, ".csv"))
  # proc_QRDQN_train<- my_proc(paste0("Summer_results/ORL_RL_train_samp-R_obs-W_", prefix, "_fips-", county,"_qrdqn_", model, "_fips_", county, ".csv"))

  proc_LSTM_eval_samp<- my_proc(paste0("Summer_results/ORL_RL_eval_samp-R_samp-W_", prefix, "_fips-", county,"_lstm_", model, "_fips_", county, ".csv"))
  proc_LSTM_train_samp<- my_proc(paste0("Summer_results/ORL_RL_train_samp-R_samp-W_", prefix, "_fips-", county,"_lstm_", model, "_fips_", county, ".csv"))
  proc_LSTM_eval<- my_proc(paste0("Summer_results/ORL_RL_eval_samp-R_obs-W_", prefix, "_fips-", county,"_lstm_", model, "_fips_", county, ".csv"))
  proc_LSTM_train<- my_proc(paste0("Summer_results/ORL_RL_train_samp-R_obs-W_", prefix, "_fips-", county,"_lstm_", model, "_fips_", county, ".csv"))

  ## Format for table:
  r<- sapply(these, function(x){
    return(c(get(paste0("proc_", x, "_eval")), get(paste0("proc_", x, "_eval_samp")),
             get(paste0("proc_", x, "_train")), get(paste0("proc_", x, "_train_samp"))))
  })
  Reward<- round(as.vector(t(r)),3)
  
  if(model == Models[1]){
    DF<- data.frame(Type, proc_Random, proc_NWS, Algo, Reward)
    names(DF)[which(names(DF) == "Reward")]<- paste0("R_", model)
    proc_df<- DF
  }else{
    DF<- data.frame(Type, proc_Random, proc_NWS, Algo, Reward)
    names(DF)[which(names(DF) == "Reward")]<- paste0("R_", model)
    proc_df<- left_join(proc_df, DF)
  }
  print(model)
}

proc_df#[,c(1:4, 9, 10, 5:8)]

## Now to get alert statistics:

as_NWS_eval_samp<- assess(paste0("Summer_results/ORL_NWS_eval_samp-R_samp-W_test_fips_", county, ".csv"))
as_NWS_train_samp<- assess(paste0("Summer_results/ORL_NWS_train_samp-R_samp-W_test_fips_", county, ".csv"))
as_NWS_eval<- assess(paste0("Summer_results/ORL_NWS_eval_samp-R_obs-W_test_fips_", county, ".csv"))
as_NWS_train<- assess(paste0("Summer_results/ORL_NWS_train_samp-R_obs-W_test_fips_", county, ".csv"))

as_random_eval_samp<- assess(paste0("Summer_results/ORL_random_eval_samp-R_samp-W_test_fips_", county, ".csv"))
as_random_train_samp<- assess(paste0("Summer_results/ORL_random_train_samp-R_samp-W_test_fips_", county, ".csv"))
as_random_eval<- assess(paste0("Summer_results/ORL_random_eval_samp-R_obs-W_test_fips_", county, ".csv"))
as_random_train<- assess(paste0("Summer_results/ORL_random_train_samp-R_obs-W_test_fips_", county, ".csv"))

as_random<- round(rbind(as_random_eval, as_random_eval_samp, as_random_train, as_random_train_samp),2)
as_nws<- round(rbind(as_NWS_eval, as_NWS_eval_samp, as_NWS_train, as_NWS_train_samp),2)

as_random
as_nws

### Make a table, updated:

as_Random<- data.frame(as_random)
as_Random$Type<- type
as_Random$Algo<- "random"
as_Random$Model<- ""

as_NWS<- data.frame(as_nws)
as_NWS$Type<- type
as_NWS$Algo<- "NWS"
as_NWS$Model<- ""

as_Type<- rep(type, length(these))
as_Algo<- rep(these, each=4)

# Models<- Models[5:6]

for(model in Models){
  
  ## Read in data and calculate estimated rewards, also get iteration of best model from eval:
  ppo_npz<- np$load(paste0("logs/SB/", prefix, "_fips-", county,"_ppo_", model, "/results/evaluations.npz")) 
  ppo_iters<- ppo_npz$f[["timesteps"]]
  ppo_evals<- rowMeans(ppo_npz$f[["results"]])
  as_PPO_eval_samp<- assess(paste0("Summer_results/ORL_RL_eval_samp-R_samp-W_", prefix, "_fips-", county,"_ppo_", model, "_fips_", county, ".csv"))
  as_PPO_train_samp<- assess(paste0("Summer_results/ORL_RL_train_samp-R_samp-W_", prefix, "_fips-", county,"_ppo_", model, "_fips_", county, ".csv"))
  as_PPO_eval<- assess(paste0("Summer_results/ORL_RL_eval_samp-R_obs-W_", prefix, "_fips-", county,"_ppo_", model, "_fips_", county, ".csv"))
  as_PPO_train<- assess(paste0("Summer_results/ORL_RL_train_samp-R_obs-W_", prefix, "_fips-", county,"_ppo_", model, "_fips_", county, ".csv"))
  
  trpo_npz<- np$load(paste0("logs/SB/", prefix, "_fips-", county,"_trpo_", model, "/results/evaluations.npz")) 
  trpo_iters<- trpo_npz$f[["timesteps"]]
  trpo_evals<- rowMeans(trpo_npz$f[["results"]])
  as_TRPO_eval_samp<- assess(paste0("Summer_results/ORL_RL_eval_samp-R_samp-W_", prefix, "_fips-", county,"_trpo_", model, "_fips_", county, ".csv"))
  as_TRPO_train_samp<- assess(paste0("Summer_results/ORL_RL_train_samp-R_samp-W_", prefix, "_fips-", county,"_trpo_", model, "_fips_", county, ".csv"))
  as_TRPO_eval<- assess(paste0("Summer_results/ORL_RL_eval_samp-R_obs-W_", prefix, "_fips-", county,"_trpo_", model, "_fips_", county, ".csv"))
  as_TRPO_train<- assess(paste0("Summer_results/ORL_RL_train_samp-R_obs-W_", prefix, "_fips-", county,"_trpo_", model, "_fips_", county, ".csv"))
  
  dqn_npz<- np$load(paste0("logs/SB/", prefix, "_fips-", county,"_dqn_", model, "/results/evaluations.npz")) 
  dqn_iters<- dqn_npz$f[["timesteps"]]
  dqn_evals<- rowMeans(dqn_npz$f[["results"]])
  as_DQN_eval_samp<- assess(paste0("Summer_results/ORL_RL_eval_samp-R_samp-W_", prefix, "_fips-", county,"_dqn_", model, "_fips_", county, ".csv"))
  as_DQN_train_samp<- assess(paste0("Summer_results/ORL_RL_train_samp-R_samp-W_", prefix, "_fips-", county,"_dqn_", model, "_fips_", county, ".csv"))
  as_DQN_eval<- assess(paste0("Summer_results/ORL_RL_eval_samp-R_obs-W_", prefix, "_fips-", county,"_dqn_", model, "_fips_", county, ".csv"))
  as_DQN_train<- assess(paste0("Summer_results/ORL_RL_train_samp-R_obs-W_", prefix, "_fips-", county,"_dqn_", model, "_fips_", county, ".csv"))
  
  # as_QRDQN_eval_samp<- assess(paste0("Summer_results/ORL_RL_eval_samp-R_samp-W_", prefix, "_fips-", county,"_qrdqn_", model, "_fips_", county, ".csv"))
  # as_QRDQN_train_samp<- assess(paste0("Summer_results/ORL_RL_train_samp-R_samp-W_", prefix, "_fips-", county,"_qrdqn_", model, "_fips_", county, ".csv"))
  # as_QRDQN_eval<- assess(paste0("Summer_results/ORL_RL_eval_samp-R_obs-W_", prefix, "_fips-", county,"_qrdqn_", model, "_fips_", county, ".csv"))
  # as_QRDQN_train<- assess(paste0("Summer_results/ORL_RL_train_samp-R_obs-W_", prefix, "_fips-", county,"_qrdqn_", model, "_fips_", county, ".csv"))

  lstm_npz<- np$load(paste0("logs/SB/", prefix, "_fips-", county,"_lstm_", model, "/results/evaluations.npz")) 
  lstm_iters<- lstm_npz$f[["timesteps"]]
  lstm_evals<- rowMeans(lstm_npz$f[["results"]])
  as_LSTM_eval_samp<- assess(paste0("Summer_results/ORL_RL_eval_samp-R_samp-W_", prefix, "_fips-", county,"_lstm_", model, "_fips_", county, ".csv"))
  as_LSTM_train_samp<- assess(paste0("Summer_results/ORL_RL_train_samp-R_samp-W_", prefix, "_fips-", county,"_lstm_", model, "_fips_", county, ".csv"))
  as_LSTM_eval<- assess(paste0("Summer_results/ORL_RL_eval_samp-R_obs-W_", prefix, "_fips-", county,"_lstm_", model, "_fips_", county, ".csv"))
  as_LSTM_train<- assess(paste0("Summer_results/ORL_RL_train_samp-R_obs-W_", prefix, "_fips-", county,"_lstm_", model, "_fips_", county, ".csv"))
  
  ## Format for table:
  cols<- names(as_TRPO_eval)
  r<- matrix(0, nrow=0, ncol=length(cols))
  r<- data.frame(r)
  names(r)<- cols
  
  for(t in these){
    r<- rbind(r, get(paste0("as_", t, "_eval")), get(paste0("as_", t, "_eval_samp")),
          get(paste0("as_", t, "_train")), get(paste0("as_", t, "_train_samp")))
  }
  
  if(model == Models[1]){
    as_df<- as_Random
    as_df<- rbind(as_df, as_NWS)
    r$Type<- as_Type
    r$Algo<- as_Algo
    r$Model<- model
    as_df<- rbind(as_df, r)
    saved_iter<- c(NA, NA, # for random and NWS
                   ppo_iters[which.max(ppo_evals)], 
                   trpo_iters[which.max(trpo_evals)],
                   dqn_iters[which.max(dqn_evals)],
                   lstm_iters[which.max(lstm_evals)])
    saved_eval<- c(NA, NA, # for random and NWS
                   max(ppo_evals), 
                   max(trpo_evals),
                   max(dqn_evals),
                   max(lstm_evals))
    as_df$Best_iter<- rep(saved_iter, each=4)
    as_df$Best_eval<- rep(saved_eval, each=4)
    
  }else{
    r$Type<- as_Type
    r$Algo<- as_Algo
    r$Model<- model
    saved_iter<- c(ppo_iters[which.max(ppo_evals)], 
                   trpo_iters[which.max(trpo_evals)],
                   dqn_iters[which.max(dqn_evals)],
                   lstm_iters[which.max(lstm_evals)])
    saved_eval<- c(max(ppo_evals), 
                   max(trpo_evals),
                   max(dqn_evals),
                   max(lstm_evals))
    r$Best_iter<- rep(saved_iter, each=4)
    r$Best_eval<- rep(saved_eval, each=4)
    as_df<- rbind(as_df, r)
  }
  print(model)
}

as_df[,c( "Model", "Algo", "Best_iter", "Best_eval", "Type", cols)]



