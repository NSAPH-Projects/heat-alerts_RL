
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

NWS_eval_samp<- my_proc(read.csv("Summer_results/ORL_NWS_eval_samp-R_samp-W_test_fips_36005.csv")[,-1])
NWS_train_samp<- my_proc(read.csv("Summer_results/ORL_NWS_train_samp-R_samp-W_test_fips_36005.csv")[,-1])
NWS_eval<- my_proc(read.csv("Summer_results/ORL_NWS_eval_samp-R_obs-W_test_fips_36005.csv")[,-1])
NWS_train<- my_proc(read.csv("Summer_results/ORL_NWS_train_samp-R_obs-W_test_fips_36005.csv")[,-1])

NA_eval_samp<- my_proc(read.csv("Summer_results/ORL_NA_eval_samp-R_samp-W_test_fips_36005.csv")[,-1])
NA_train_samp<- my_proc(read.csv("Summer_results/ORL_NA_train_samp-R_samp-W_test_fips_36005.csv")[,-1])
NA_eval<- my_proc(read.csv("Summer_results/ORL_NA_eval_samp-R_obs-W_test_fips_36005.csv")[,-1])
NA_train<- my_proc(read.csv("Summer_results/ORL_NA_train_samp-R_obs-W_test_fips_36005.csv")[,-1])


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

