
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
algos<- c("CPQ") # "DoubleDQN", 
MR<- c("T")
seed<- c("321", "221", "121")
fips<- c("4013", "36047")
NHU<- c(256)
NHL<- c(3)
LR<- c(0.1, 0.5) # 0.001
SR<- c(3)
b_size<- c(1200) # 32


tests<- expand.grid(algos, MR, seed, fips, NHU, NHL, LR, SR, b_size)
colnames(tests)<- c("algo", "modeled_r", "seed", "fips", "NHU", "NHL", "LR",
                    "SR", "b_size")

sink("Run_jobs/Single_county_dqn_tests")
for(i in 1:nrow(tests)){
  cat(paste("python heat_alerts/Single_county_RL.py",
                "--n_gpus", 0,
                "--lr", tests[i, "LR"],
                "--modeled_r", tests[i, "modeled_r"],
                "--eligible 'all'",
                "--algo", tests[i, "algo"],
                "--n_layers", tests[i, "NHL"],
                "--n_hidden", tests[i, "NHU"],
                "--n_epochs", 10000,
                "--b_size", tests[i, "b_size"],
                "--sync_rate", tests[i, "SR"],
                "--seed", tests[i, "seed"],
                "--fips", tests[i, "fips"],
                "--model_name", paste0("B-1200_SC_", tests[i, "algo"],
                                       "_Elig-", "all",
                                       "_MR-", tests[i, "modeled_r"],
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

