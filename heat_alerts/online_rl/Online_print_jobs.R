
algos<- c("trpo", "ppo", "dqn", "qrdqn", "lstm")
eval.val_years<- c("true", "false")
eval.match_similar<- c("true", "false")
eval_mode<- c("true") # , "false"
eval.eval_mode<- c("true") # , "false"

penalty<- c(0.005) # 0.0, 0.001, 0.01, 1.0, 5.0

training<- expand.grid(algos, penalty, eval_mode, eval.eval_mode)
colnames(training)<- c("algo", "penalty", "eval_mode", "eval.eval_mode")
evaluation<- expand.grid(algos, eval.val_years, eval.match_similar,
                         eval.eval_mode)
colnames(evaluation)<- c("algo", "eval.val_years", "eval.match_similar",
                         "eval.eval_mode")

training$model_name<- paste0(training$algo, "_EM_p-005") # _EM_p-005
evaluation$model_name<- paste0(evaluation$algo, "_EM_p-005") # _EM_p-005

training_script<- "python train_online_rl_sb3.py"
evaluation_script<- "python old_evaluation_SB3.py"

Training<- sapply(1:ncol(training), function(i){paste0(colnames(training)[i], "=", training[,i])})
Evaluation<- sapply(1:ncol(evaluation), function(i){paste0(colnames(evaluation)[i], "=", evaluation[,i])})


for(i in 1:length(algos)){
  cat(training_script,
    paste(
      Training[i,]
    ), " \n"
  )
  for(j in which(evaluation$algo == algos[i])){
    cat(evaluation_script,
        paste(
          Evaluation[j,]
        ), " \n"
    )
  }
  cat("\n")
}


################################## Out of date...

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
            "--n_epochs", 3500, # 5000,
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
            "--model_name", paste0("Online-0_", 
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

