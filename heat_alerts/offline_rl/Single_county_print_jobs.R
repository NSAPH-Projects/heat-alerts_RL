## All the different options:

algos<- c("DQN", "DoubleDQN", "CPQ")
MR<- c("True", "False")
elig<- c("all", "90pct")
seeds<- c("321", "221", "121")
fips<- c("4013", "6085", "36061")
NHU<- c(16, 32, 64, 128, 256)
NHL<- c(1, 2, 3)
LR<- c(0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05)
SR<- c(1, 2, 3, 5, 10)
b_size<- c(16, 32, 64, 128)
ma<- c(20, 50, 100, 200)

## Preliminary:
algos<- c("DoubleDQN", "CPQ")
MR<- c("T", "F")
seed<- c("321")
fips<- c("4013")
NHU<- c(16, 32, 64, 128)
NHL<- c(2, 3)
LR<- c(0.0001, 0.001, 0.01)
SR<- c(3)
b_size<- c(32)
ma<- c(20)

## Next:
algos<- c("CPQ", "DoubleDQN") # , "DoubleDQN"
seed<- c("321", "221", "121")
fips<- c("4013", "36061")
NHU<- c(256)
NHL<- c(3)
LR<- c(0.05) 
SR<- c(3)
HER<- c("T")
std_b<- c(0)
mult_a<- c(0.9)
mult_alag1<- c(0.95)
b_size<- c(1200) # 2048*2


tests<- expand.grid(algos, std_b, HER, seed, LR, SR, b_size,
                    mult_a, mult_alag1, fips)
colnames(tests)<- c("algo", "Std_B", "HER", "seed", "LR",
                    "SR", "b_size", "mult_a", "mult_alag1", "fips")

# rm_pos<- which(tests$algo == "DoubleDQN" & tests$HER == "T")
# tests<- tests[-rm_pos,]



sink("Run_jobs/Single_county_dqn_tests")
for(i in 1:nrow(tests)){
  cat(paste("python heat_alerts/Single_county_RL.py",
                "--n_gpus", 0,
                "--lr", tests[i, "LR"],
                "--modeled_r", tests[i, "modeled_r"],
                "--eligible 'all'",
                "--std_budget", tests[i, "Std_B"],
                "--Pct90", tests[i, "Pct90"],
                "--algo", tests[i, "algo"],
                "--HER", tests[i, "HER"],
                "--n_layers", tests[i, "NHL"],
                "--n_hidden", tests[i, "NHU"],
                "--n_epochs", 10000,
                "--b_size", tests[i, "b_size"],
                "--sync_rate", tests[i, "SR"],
                "--seed", tests[i, "seed"],
                "--fips", tests[i, "fips"],
                "--model_name", paste0("Best_SC_", tests[i, "algo"],
                                       "_MR-", tests[i, "modeled_r"],
                                       "_Pct90-", tests[i, "Pct90"],
                                       "_Std_B-", tests[i, "Std_B"],
                                       "_HER-", tests[i, "HER"],
                                       "_LR-", tests[i, "LR"],
                                       # "_NH-", tests[i, "NHL"],
                                       # "-", tests[i, "NHU"],
                                       # "_B-", tests[i, "b_size"],
                                       "_SR-", tests[i, "SR"],
                                       "_fips-", tests[i, "fips"],
                                       "_seed-", tests[i, "seed"],
                                       "\n"
                                       )
                ))
}

sink()

