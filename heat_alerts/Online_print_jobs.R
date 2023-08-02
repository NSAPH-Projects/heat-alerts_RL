
algos<- c("DoubleDQN") # , "DQN" , "SAC"
seed<- c("321") # , "221", "121"
fips<- c("4013", "36061")
NHU<- c(32)
NHL<- c(2)
LR<- c(0.01, 0.001) 
SR<- c(3, 7)
# std_b<- c(0)
b_size<- c(600)


tests<- expand.grid(algos,
                    # seed,
                    fips,
                    # NHU,
                    # NHL, 
                    LR, 
                    SR, 
                    b_size
                    )
colnames(tests)<- c("algo",
                    # "seed",
                    "fips",
                    # "n_hidden",
                    # "n_layers", 
                    "LR",
                    "SR",
                    "b_size"
                    )

# rm_pos<- which(tests$algo == "DoubleDQN" & tests$HER == "T")
# tests<- tests[-rm_pos,]


sink("Run_jobs/Online_tests")
for(i in 1:nrow(tests)){
  cat(paste("python Online_RL.py",
            "--n_gpus", 0,
            "--lr", tests[i, "LR"],
            "--algo", tests[i, "algo"],
            # "--n_layers", tests[i, "NHL"],
            # "--n_hidden", tests[i, "NHU"],
            "--n_epochs", 5000,
            "--b_size", tests[i, "b_size"],
            "--sync_rate", tests[i, "SR"],
            # "--seed", tests[i, "seed"],
            "--fips", tests[i, "fips"],
            "--model_name", paste0("Online-0_", tests[i, "algo"],
                                   "_LR-", tests[i, "LR"],
                                   # "_NH-", tests[i, "NHL"],
                                   # "-", tests[i, "NHU"],
                                   "_B-", tests[i, "b_size"],
                                   "_SR-", tests[i, "SR"],
                                   "_fips-", tests[i, "fips"],
                                   # "_seed-", tests[i, "seed"],
                                   "\n"
            )
  ))
}

sink()

