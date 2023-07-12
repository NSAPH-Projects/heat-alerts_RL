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
algos<- c("CPQ") # , "DoubleDQN"
seed<- c("321", "221", "121")
fips<- c("4013", "36061")
NHU<- c(256)
NHL<- c(3)
LR<- c(0.05) 
SR<- c(3)
HER<- c("T")
std_b<- c(0)
mult_a<- c(0.9)
mult_alag1<- c(0.95,0.99)
b_size<- c(1200) # 2048*2


tests<- expand.grid(algos, std_b, HER, seed, LR, SR, b_size,
                    mult_a, mult_alag1, fips)
colnames(tests)<- c("algo", "Std_B", "HER", "seed", "LR",
                    "SR", "b_size", "mult_a", "mult_alag1", "fips")

# rm_pos<- which(tests$algo == "DoubleDQN" & tests$HER == "T")
# tests<- tests[-rm_pos,]

# tests<- rbind(tests, tests)
# tests$mult_a[7:12]<- 0.8
# tests$mult_alag1[7:12]<- 0.82

sink("Run_jobs/Single_county_dqn_tests")
for(i in 1:nrow(tests)){
  cat(paste("python heat_alerts/Simulated_RL.py",
                "--n_gpus", 0,
                "--lr", tests[i, "LR"],
                "--std_budget", tests[i, "Std_B"],
                "--mult_a", tests[i, "mult_a"],
                "--mult_alag1", tests[i, "mult_alag1"],
                "--algo", tests[i, "algo"],
                "--HER", tests[i, "HER"],
                "--n_epochs", 10000,
                "--sa", 10,
                "--sync_rate", tests[i, "SR"],
                "--fips", tests[i, "fips"],
                "--seed", tests[i, "seed"],
                "--b_size", tests[i, "b_size"],
                "--model_name", paste0("SC_SimAR_trunc-2_", tests[i, "algo"],
                                       "_Std_B-", tests[i, "Std_B"],
                                       "_Mult-", 1-tests[i, "mult_a"],
                                       "_Mlag-", tests[i, "mult_alag1"],
                                       # "_SR-", tests[i, "SR"],
                                       "_LR-", tests[i, "LR"],
                                       "_fips-", tests[i, "fips"],
                                       "_seed-", tests[i, "seed"],
                                       "\n"
                                       )
                ))
}

sink()
