
library(dplyr)

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
    estimated_reward<- sum(agg_df$Rewards*(1/nrow(agg_df))/agg_df$Frac)
    # return(list(agg_df, estimated_reward))
    return(estimated_reward)
  }else{
    return(NA)
  }
}

# eval.val_years=false --> train
# eval.match_similar=false --> obs-W
# eval.eval_mode=true --> avg-R

NWS_eval_samp<- my_proc("Summer_results/ORL_NWS_eval_samp-R_samp-W_test_fips_36005.csv")
NWS_train_samp<- my_proc("Summer_results/ORL_NWS_train_samp-R_samp-W_test_fips_36005.csv")
NWS_eval<- my_proc("Summer_results/ORL_NWS_eval_samp-R_obs-W_test_fips_36005.csv")
NWS_train<- my_proc("Summer_results/ORL_NWS_train_samp-R_obs-W_test_fips_36005.csv")

NA_eval_samp<- my_proc("Summer_results/ORL_NA_eval_samp-R_samp-W_test_fips_36005.csv")
NA_train_samp<- my_proc("Summer_results/ORL_NA_train_samp-R_samp-W_test_fips_36005.csv")
NA_eval<- my_proc("Summer_results/ORL_NA_eval_samp-R_obs-W_test_fips_36005.csv")
NA_train<- my_proc("Summer_results/ORL_NA_train_samp-R_obs-W_test_fips_36005.csv")

### Make a table, updated:

these<- c("TRPO", "LSTM", "PPO", "DQN", "QRDQN")
Algo<- rep(these, 4)
# Model<- append(Model, rep(model, length(these)*4))
Type<- rep(c("eval", "eval_samp", "train", "train_samp"), each=length(these))
nws<- round(c(NWS_eval, NWS_eval_samp, NWS_train, NWS_train_samp))
NWS<- rep(nws, each=length(these))

# for(model in c("0", "p1", "p0", "p-01", "p-005", "p-001", "p-001_ee25")){
for(model in c("p-05_ME", "p-001_ME", "p-0_ME")){ # "p-0"
  ## Read in data and calculate estimated rewards:
  PPO_eval_samp<- my_proc(paste0("Summer_results/ORL_RL_eval_samp-R_samp-W_ppo_", model, "_fips_36005.csv"))
  PPO_train_samp<- my_proc(paste0("Summer_results/ORL_RL_train_samp-R_samp-W_ppo_", model, "_fips_36005.csv"))
  PPO_eval<- my_proc(paste0("Summer_results/ORL_RL_eval_samp-R_obs-W_ppo_", model, "_fips_36005.csv"))
  PPO_train<- my_proc(paste0("Summer_results/ORL_RL_train_samp-R_obs-W_ppo_", model, "_fips_36005.csv"))
  
  TRPO_eval_samp<- my_proc(paste0("Summer_results/ORL_RL_eval_samp-R_samp-W_trpo_", model, "_fips_36005.csv"))
  TRPO_train_samp<- my_proc(paste0("Summer_results/ORL_RL_train_samp-R_samp-W_trpo_", model, "_fips_36005.csv"))
  TRPO_eval<- my_proc(paste0("Summer_results/ORL_RL_eval_samp-R_obs-W_trpo_", model, "_fips_36005.csv"))
  TRPO_train<- my_proc(paste0("Summer_results/ORL_RL_train_samp-R_obs-W_trpo_", model, "_fips_36005.csv"))
  
  DQN_eval_samp<- my_proc(paste0("Summer_results/ORL_RL_eval_samp-R_samp-W_dqn_", model, "_fips_36005.csv"))
  DQN_train_samp<- my_proc(paste0("Summer_results/ORL_RL_train_samp-R_samp-W_dqn_", model, "_fips_36005.csv"))
  DQN_eval<- my_proc(paste0("Summer_results/ORL_RL_eval_samp-R_obs-W_dqn_", model, "_fips_36005.csv"))
  DQN_train<- my_proc(paste0("Summer_results/ORL_RL_train_samp-R_obs-W_dqn_", model, "_fips_36005.csv"))
  
  QRDQN_eval_samp<- my_proc(paste0("Summer_results/ORL_RL_eval_samp-R_samp-W_qrdqn_", model, "_fips_36005.csv"))
  QRDQN_train_samp<- my_proc(paste0("Summer_results/ORL_RL_train_samp-R_samp-W_qrdqn_", model, "_fips_36005.csv"))
  QRDQN_eval<- my_proc(paste0("Summer_results/ORL_RL_eval_samp-R_obs-W_qrdqn_", model, "_fips_36005.csv"))
  QRDQN_train<- my_proc(paste0("Summer_results/ORL_RL_train_samp-R_obs-W_qrdqn_", model, "_fips_36005.csv"))
  
  LSTM_eval_samp<- my_proc(paste0("Summer_results/ORL_RL_eval_samp-R_samp-W_lstm_", model, "_fips_36005.csv"))
  LSTM_train_samp<- my_proc(paste0("Summer_results/ORL_RL_train_samp-R_samp-W_lstm_", model, "_fips_36005.csv"))
  LSTM_eval<- my_proc(paste0("Summer_results/ORL_RL_eval_samp-R_obs-W_lstm_", model, "_fips_36005.csv"))
  LSTM_train<- my_proc(paste0("Summer_results/ORL_RL_train_samp-R_obs-W_lstm_", model, "_fips_36005.csv"))
  
  ## Format for table:
  r<- sapply(these, function(x){
    return(c(get(paste0(x, "_eval")), get(paste0(x, "_eval_samp")),
             get(paste0(x, "_train")), get(paste0(x, "_train_samp"))))
  })
  Reward<- round(as.vector(t(r)))
  
  if(model == "p-05_ME"){
    DF<- data.frame(Type, NWS, Algo, Reward)
    names(DF)[which(names(DF) == "Reward")]<- paste0("R_", model)
    df<- DF
  }else{
    DF<- data.frame(Type, NWS, Algo, Reward)
    names(DF)[which(names(DF) == "Reward")]<- paste0("R_", model)
    df<- left_join(df, DF)
  }
  print(model)
}

df

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

