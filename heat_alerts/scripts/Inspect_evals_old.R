library(dplyr)


## Manually change:
county<- 36005 
# Models<- c("0", "p1", "p0", "p-01", "p-005", "p-001", "p-001_ee25")
# Models<- c("p-05_ME", "p-001_ME", "p-0_ME")
Models<- c("p_decay", "small", "ND_small")
# Models<- c("obs-W_decay-false", "obs-W_decay-true")
# Models<- c("p_decay", "small_4013", "ND_small_4013")

these<- c("TRPO", "PPO", "DQN") # "LSTM", "QRDQN"
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
    eps<- nrow(df)/(n_days-1)
    Days<- rep(1:(n_days-1),eps)
    D<- Days[which(df$Actions == 1)]
    if(length(D)>0){
      num_alerts<- length(D)/eps
      summary_dos<- summary(D)
      diffs<- D[2:length(D)] - D[1:(length(D)-1)]
      L<- rle(diffs)
      streaks<- L$lengths[which(L$values == 1)]
      num_streaks<- length(streaks)/eps
      avg_streak_length<- mean(streaks + 1)
      avg_streak_length_overall<- mean(c(streaks + 1, rep(1,length(D)-length(streaks))))
      # return(list(agg_df, estimated_reward))
      x<- c(num_alerts, as.vector(summary_dos), num_streaks, avg_streak_length, avg_streak_length_overall)
      result<- data.frame(t(x))
      names(result)<- c("AvNAl", "Min_dos", "Q1_dos", "Median_dos", "Mean_dos", "Q3_dos", "Max_dos", "NStrk", "AvStrkLn", "AvStrkLn_all")
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
  proc_PPO_eval_samp<- my_proc(paste0("Summer_results/ORL_RL_eval_samp-R_samp-W_ppo_", model, "_fips_", county, ".csv"))
  proc_PPO_train_samp<- my_proc(paste0("Summer_results/ORL_RL_train_samp-R_samp-W_ppo_", model, "_fips_", county, ".csv"))
  proc_PPO_eval<- my_proc(paste0("Summer_results/ORL_RL_eval_samp-R_obs-W_ppo_", model, "_fips_", county, ".csv"))
  proc_PPO_train<- my_proc(paste0("Summer_results/ORL_RL_train_samp-R_obs-W_ppo_", model, "_fips_", county, ".csv"))
  
  proc_TRPO_eval_samp<- my_proc(paste0("Summer_results/ORL_RL_eval_samp-R_samp-W_trpo_", model, "_fips_", county, ".csv"))
  proc_TRPO_train_samp<- my_proc(paste0("Summer_results/ORL_RL_train_samp-R_samp-W_trpo_", model, "_fips_", county, ".csv"))
  proc_TRPO_eval<- my_proc(paste0("Summer_results/ORL_RL_eval_samp-R_obs-W_trpo_", model, "_fips_", county, ".csv"))
  proc_TRPO_train<- my_proc(paste0("Summer_results/ORL_RL_train_samp-R_obs-W_trpo_", model, "_fips_", county, ".csv"))
  
  proc_DQN_eval_samp<- my_proc(paste0("Summer_results/ORL_RL_eval_samp-R_samp-W_dqn_", model, "_fips_", county, ".csv"))
  proc_DQN_train_samp<- my_proc(paste0("Summer_results/ORL_RL_train_samp-R_samp-W_dqn_", model, "_fips_", county, ".csv"))
  proc_DQN_eval<- my_proc(paste0("Summer_results/ORL_RL_eval_samp-R_obs-W_dqn_", model, "_fips_", county, ".csv"))
  proc_DQN_train<- my_proc(paste0("Summer_results/ORL_RL_train_samp-R_obs-W_dqn_", model, "_fips_", county, ".csv"))
  
  # proc_QRDQN_eval_samp<- my_proc(paste0("Summer_results/ORL_RL_eval_samp-R_samp-W_", prefix, "_fips-", county,"_qrdqn_", model, "_fips_", county, ".csv"))
  # proc_QRDQN_train_samp<- my_proc(paste0("Summer_results/ORL_RL_train_samp-R_samp-W_", prefix, "_fips-", county,"_qrdqn_", model, "_fips_", county, ".csv"))
  # proc_QRDQN_eval<- my_proc(paste0("Summer_results/ORL_RL_eval_samp-R_obs-W_", prefix, "_fips-", county,"_qrdqn_", model, "_fips_", county, ".csv"))
  # proc_QRDQN_train<- my_proc(paste0("Summer_results/ORL_RL_train_samp-R_obs-W_", prefix, "_fips-", county,"_qrdqn_", model, "_fips_", county, ".csv"))
  # 
  # proc_LSTM_eval_samp<- my_proc(paste0("Summer_results/ORL_RL_eval_samp-R_samp-W_", prefix, "_fips-", county,"_lstm_", model, "_fips_", county, ".csv"))
  # proc_LSTM_train_samp<- my_proc(paste0("Summer_results/ORL_RL_train_samp-R_samp-W_", prefix, "_fips-", county,"_lstm_", model, "_fips_", county, ".csv"))
  # proc_LSTM_eval<- my_proc(paste0("Summer_results/ORL_RL_eval_samp-R_obs-W_", prefix, "_fips-", county,"_lstm_", model, "_fips_", county, ".csv"))
  # proc_LSTM_train<- my_proc(paste0("Summer_results/ORL_RL_train_samp-R_obs-W_", prefix, "_fips-", county,"_lstm_", model, "_fips_", county, ".csv"))
  
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

proc_df

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


for(model in Models){
  ## Read in data and calculate estimated rewards:
  as_PPO_eval_samp<- assess(paste0("Summer_results/ORL_RL_eval_samp-R_samp-W_ppo_", model, "_fips_", county, ".csv"))
  as_PPO_train_samp<- assess(paste0("Summer_results/ORL_RL_train_samp-R_samp-W_ppo_", model, "_fips_", county, ".csv"))
  as_PPO_eval<- assess(paste0("Summer_results/ORL_RL_eval_samp-R_obs-W_ppo_", model, "_fips_", county, ".csv"))
  as_PPO_train<- assess(paste0("Summer_results/ORL_RL_train_samp-R_obs-W_ppo_", model, "_fips_", county, ".csv"))
  
  as_TRPO_eval_samp<- assess(paste0("Summer_results/ORL_RL_eval_samp-R_samp-W_trpo_", model, "_fips_", county, ".csv"))
  as_TRPO_train_samp<- assess(paste0("Summer_results/ORL_RL_train_samp-R_samp-W_trpo_", model, "_fips_", county, ".csv"))
  as_TRPO_eval<- assess(paste0("Summer_results/ORL_RL_eval_samp-R_obs-W_trpo_", model, "_fips_", county, ".csv"))
  as_TRPO_train<- assess(paste0("Summer_results/ORL_RL_train_samp-R_obs-W_trpo_", model, "_fips_", county, ".csv"))
  
  as_DQN_eval_samp<- assess(paste0("Summer_results/ORL_RL_eval_samp-R_samp-W_dqn_", model, "_fips_", county, ".csv"))
  as_DQN_train_samp<- assess(paste0("Summer_results/ORL_RL_train_samp-R_samp-W_dqn_", model, "_fips_", county, ".csv"))
  as_DQN_eval<- assess(paste0("Summer_results/ORL_RL_eval_samp-R_obs-W_dqn_", model, "_fips_", county, ".csv"))
  as_DQN_train<- assess(paste0("Summer_results/ORL_RL_train_samp-R_obs-W_dqn_", model, "_fips_", county, ".csv"))
  
  # as_QRDQN_eval_samp<- assess(paste0("Summer_results/ORL_RL_eval_samp-R_samp-W_", prefix, "_fips-", county,"_qrdqn_", model, "_fips_", county, ".csv"))
  # as_QRDQN_train_samp<- assess(paste0("Summer_results/ORL_RL_train_samp-R_samp-W_", prefix, "_fips-", county,"_qrdqn_", model, "_fips_", county, ".csv"))
  # as_QRDQN_eval<- assess(paste0("Summer_results/ORL_RL_eval_samp-R_obs-W_", prefix, "_fips-", county,"_qrdqn_", model, "_fips_", county, ".csv"))
  # as_QRDQN_train<- assess(paste0("Summer_results/ORL_RL_train_samp-R_obs-W_", prefix, "_fips-", county,"_qrdqn_", model, "_fips_", county, ".csv"))
  # 
  # as_LSTM_eval_samp<- assess(paste0("Summer_results/ORL_RL_eval_samp-R_samp-W_", prefix, "_fips-", county,"_lstm_", model, "_fips_", county, ".csv"))
  # as_LSTM_train_samp<- assess(paste0("Summer_results/ORL_RL_train_samp-R_samp-W_", prefix, "_fips-", county,"_lstm_", model, "_fips_", county, ".csv"))
  # as_LSTM_eval<- assess(paste0("Summer_results/ORL_RL_eval_samp-R_obs-W_", prefix, "_fips-", county,"_lstm_", model, "_fips_", county, ".csv"))
  # as_LSTM_train<- assess(paste0("Summer_results/ORL_RL_train_samp-R_obs-W_", prefix, "_fips-", county,"_lstm_", model, "_fips_", county, ".csv"))
  
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
    
  }else{
    r$Type<- as_Type
    r$Algo<- as_Algo
    r$Model<- model
    as_df<- rbind(as_df, r)
  }
  print(model)
}

as_df[,c( "Model", "Algo", "Type", cols)]


### Compare:
NA_eval
NWS_eval
TRPO_eval
LSTM_eval
PPO_eval
DQN_eval
QRDQN_eval

NA_train
NWS_train
PPO_train
TRPO_train
LSTM_train
DQN_train
QRDQN_train

NA_eval_samp
NWS_eval_samp
PPO_eval_samp
TRPO_eval_samp
LSTM_eval_samp
DQN_eval_samp
QRDQN_eval_samp

NA_train_samp
NWS_train_samp
PPO_train_samp
TRPO_train_samp
LSTM_train_samp
DQN_train_samp
QRDQN_train_samp


################################ Out of date...

### Make a table:

all<- data.frame(Policy = c("No alerts", "NWS", "TRPO", "PPO",
                            "QRDQN", "DQN"),
                 Actions = c(0, sum(NWS$Actions),
                             sum(trpo$Actions),
                             sum(ppo$Actions),
                             sum(qrdqn$Actions),
                             sum(dqn$Actions)),
                 Rewards = c(sum(None$Rewards), 
                             sum(NWS$Rewards),
                             sum(trpo$Rewards),
                             sum(ppo$Rewards),
                             sum(qrdqn$Rewards),
                             sum(dqn$Rewards))
)

