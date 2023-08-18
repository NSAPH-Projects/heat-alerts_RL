

NWS<- read.csv("Summer_results/ORL_NWS_eval_samp-R_obs-W_test_fips_36005.csv")[,-1]

agg_NWS<- aggregate(. ~ Year + Budget, NWS, sum)
agg_None<- aggregate(. ~ Year, None, sum)
agg_trpo<- aggregate(. ~ Year, trpo, sum)
agg_ppo<- aggregate(. ~ Year, ppo, sum)
agg_qrdqn<- aggregate(. ~ Year, qrdqn, sum)
agg_dqn<- aggregate(. ~ Year, dqn, sum)

agg_None
agg_NWS
agg_dqn
agg_qrdqn
agg_ppo
agg_trpo

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

