
algos<- c("DoubleDQN", "SAC") # , "DQN", "DoubleDQN", "SAC"
seed<- c("321") # , "221", "121"
fips<- c("36005") # c("36005", "41067", "28035", "6071", "4013") # c("53057", "37171", "27019", "35045", "12085") # c("4013", "36061")
NHU<- c(32, 64) 
NHL<- c(2, 3) # 3
LR<- c(0.001, 0.01) # 0.01
SR<- c(3)
# std_b<- c(0)
b_size<- c(1200) # 600, 1200
xpl<- c("T", "F")
eps_dur<- c(1.0)
eps_t<- c(0.00000001) # , 0.05, 0.01, 0.1
penalty = c(0, -0.1, -1) # c(-1, -5, -10)

tests<- expand.grid(algos,
                    # seed,
                    fips,
                    NHU,
                    NHL, 
                    LR, 
                    SR, 
                    b_size,
                    xpl,
                    eps_dur,
                    eps_t,
                    penalty
                    )
colnames(tests)<- c("algo",
                    # "seed",
                    "fips",
                    "NHU",
                    "NHL",
                    "LR",
                    "SR",
                    "b_size",
                    "xpl",
                    "eps_dur",
                    "eps_t",
                    "penalty"
                    )

rm_pos<- which((tests$algo == "DoubleDQN" & tests$xpl == "F") |
                 (tests$algo == "SAC" & tests$xpl == "T"))
tests<- tests[-rm_pos,]


sink("Run_jobs/Online_tests")
for(i in 1:nrow(tests)){
  cat(paste("python Online_RL.py",
            "--n_gpus", 0,
            "--lr", tests[i, "LR"],
            "--algo", tests[i, "algo"],
            "--n_layers", tests[i, "NHL"],
            "--n_hidden", tests[i, "NHU"],
            "--n_epochs", 5000,
            "--b_size", tests[i, "b_size"],
            "--sync_rate", tests[i, "SR"],
            # "--seed", tests[i, "seed"],
            "--fips", tests[i, "fips"],
            "--hold_out", "2015 2011 2007",
            "--sa", 50,
            "--xpl", tests[i, "xpl"],
            "--eps_dur", tests[i, "eps_dur"],
            "--eps_t", tests[i, "eps_t"],
            "--penalty", tests[i, "penalty"],
            "--model_name", paste0("Online-2_", 
                                   tests[i, "algo"],
                                   # "_xpl-", tests[i, "xpl"],
                                   "_LR-", tests[i, "LR"],
                                   "_NH-", tests[i, "NHL"],
                                   "-", tests[i, "NHU"],
                                   "_P", tests[i, "penalty"],
                                   # "_B-", tests[i, "b_size"],
                                   # "_SR-", tests[i, "SR"],
                                   # "_ET-", tests[i, "eps_t"],
                                   # "_ED-", tests[i, "eps_dur"],
                                   "_fips-", tests[i, "fips"],
                                   # "_seed-", tests[i, "seed"],
                                   "\n"
            )
  ))
}

sink()

