
n_days<- 153

my_proc<- function(df){
  df$Count = 1
  agg_df<- aggregate(. ~ Year, df, sum)
  agg_df$Budget<- agg_df$Budget/(n_days-1)
  agg_df$budget_frac<- agg_df$Actions/agg_df$Budget
  agg_df$Frac<- agg_df$Count/sum(agg_df$Count)
  estimated_reward<- sum(agg_df$Rewards*(1/nrow(agg_df))/agg_df$Frac)
  return(list(agg_df, estimated_reward))
}

# eval.val_years=false --> train
# eval.match_similar=false --> obs-W
# eval.eval_mode=true --> avg-R

model<- "0"

NWS_eval_samp<- my_proc(read.csv("Summer_results/ORL_NWS_eval_samp-R_samp-W_test_fips_36005.csv")[,-1])
NWS_train_samp<- my_proc(read.csv("Summer_results/ORL_NWS_train_samp-R_samp-W_test_fips_36005.csv")[,-1])
NWS_eval<- my_proc(read.csv("Summer_results/ORL_NWS_eval_samp-R_obs-W_test_fips_36005.csv")[,-1])
NWS_train<- my_proc(read.csv("Summer_results/ORL_NWS_train_samp-R_obs-W_test_fips_36005.csv")[,-1])

NA_eval_samp<- my_proc(read.csv("Summer_results/ORL_NA_eval_samp-R_samp-W_test_fips_36005.csv")[,-1])
NA_train_samp<- my_proc(read.csv("Summer_results/ORL_NA_train_samp-R_samp-W_test_fips_36005.csv")[,-1])
NA_eval<- my_proc(read.csv("Summer_results/ORL_NA_eval_samp-R_obs-W_test_fips_36005.csv")[,-1])
NA_train<- my_proc(read.csv("Summer_results/ORL_NA_train_samp-R_obs-W_test_fips_36005.csv")[,-1])

PPO_eval_samp<- my_proc(read.csv(paste0("Summer_results/ORL_RL_eval_samp-R_samp-W_ppo_", model, "_fips_36005.csv"))[,-1])
PPO_train_samp<- my_proc(read.csv(paste0("Summer_results/ORL_RL_train_samp-R_samp-W_ppo_", model, "_fips_36005.csv"))[,-1])
PPO_eval<- my_proc(read.csv(paste0("Summer_results/ORL_RL_eval_samp-R_obs-W_ppo_", model, "_fips_36005.csv"))[,-1])
PPO_train<- my_proc(read.csv(paste0("Summer_results/ORL_RL_train_samp-R_obs-W_ppo_", model, "_fips_36005.csv"))[,-1])

TRPO_eval_samp<- my_proc(read.csv(paste0("Summer_results/ORL_RL_eval_samp-R_samp-W_trpo_", model, "_fips_36005.csv"))[,-1])
TRPO_train_samp<- my_proc(read.csv(paste0("Summer_results/ORL_RL_train_samp-R_samp-W_trpo_", model, "_fips_36005.csv"))[,-1])
TRPO_eval<- my_proc(read.csv(paste0("Summer_results/ORL_RL_eval_samp-R_obs-W_trpo_", model, "_fips_36005.csv"))[,-1])
TRPO_train<- my_proc(read.csv(paste0("Summer_results/ORL_RL_train_samp-R_obs-W_trpo_", model, "_fips_36005.csv"))[,-1])

DQN_eval_samp<- my_proc(read.csv(paste0("Summer_results/ORL_RL_eval_samp-R_samp-W_dqn_", model, "_fips_36005.csv"))[,-1])
DQN_train_samp<- my_proc(read.csv(paste0("Summer_results/ORL_RL_train_samp-R_samp-W_dqn_", model, "_fips_36005.csv"))[,-1])
DQN_eval<- my_proc(read.csv(paste0("Summer_results/ORL_RL_eval_samp-R_obs-W_dqn_", model, "_fips_36005.csv"))[,-1])
DQN_train<- my_proc(read.csv(paste0("Summer_results/ORL_RL_train_samp-R_obs-W_dqn_", model, "_fips_36005.csv"))[,-1])

QRDQN_eval_samp<- my_proc(read.csv(paste0("Summer_results/ORL_RL_eval_samp-R_samp-W_qrdqn_", model, "_fips_36005.csv"))[,-1])
QRDQN_train_samp<- my_proc(read.csv(paste0("Summer_results/ORL_RL_train_samp-R_samp-W_qrdqn_", model, "_fips_36005.csv"))[,-1])
QRDQN_eval<- my_proc(read.csv(paste0("Summer_results/ORL_RL_eval_samp-R_obs-W_qrdqn_", model, "_fips_36005.csv"))[,-1])
QRDQN_train<- my_proc(read.csv(paste0("Summer_results/ORL_RL_train_samp-R_obs-W_qrdqn_", model, "_fips_36005.csv"))[,-1])

### Compare:
NA_eval
NWS_eval
PPO_eval
TRPO_eval
DQN_eval
QRDQN_eval

NA_train
NWS_train
PPO_train
TRPO_train
DQN_train
QRDQN_train

NA_eval_samp
NWS_eval_samp
PPO_eval_samp
TRPO_eval_samp
DQN_eval_samp
QRDQN_eval_samp

NA_train_samp
NWS_train_samp
PPO_train_samp
TRPO_train_samp
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

