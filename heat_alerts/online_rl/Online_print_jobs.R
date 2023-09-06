
# county<- c(41067, 53015, 20161, 37085, 48157, 
#            28049, 19153, 17167, 31153, 6071, 4013) # 36005, 4013
# 
# county<- c(34021, 19155, 17115, 29021, 29019, 5045, 40017, 21059,
#            47113, 42017, 22109, 45015, 13031, 48367, 22063, 41053, 
#            32003, 4015, 6025)

county<- c(41067, 53015, 20161, 37085, 48157, 
           28049, 19153, 17167, 31153, 6071, 4013,
           34021, 19155, 17115, 29021, 29019, 5045, 40017, 21059,
           47113, 42017, 22109, 45015, 13031, 48367, 22063, 41053, 
           32003, 4015, 6025)

algos<- c("trpo") # , "ppo", "dqn", "lstm" "qrdqn"
# match_similar<- c("false") # "true"
eval.val_years<- c("true", "false")
eval.match_similar<- c("true", "false") 
# eval_mode<- c("true") # , "false"
# eval.eval_mode<- c("true") # , "false"

learning_rate<- c(0.001) #, 0.0001
eval.episodes<- c(100) # 25
policy_kwargs.net_arch<- c("[16,16]") # "[16]", "[16,16]", "[32,32]", "[16,16,16]"
penalty_decay<- c("false") # "true", "false"
explore_budget<- c("false") # "true", "false"
restrict_alerts<- c("true") # "true", "false"
hi_penalty<- c("false") # "true", "false"
# HI_restriction<- c(0.7, 0.75, 0.8, 0.85, 0.9)
# HI_restriction<- c(0.5, 0.55, 0.6, 0.65)
HI_restriction<- seq(0.5, 0.9, 0.05)
hi_rstr_decay<- c("false") # "true", "false"
# penalty<- c(0.0) # , 0.01


training<- expand.grid(county, 
                       algos, 
                       # match_similar,
                       # explore_budget,
                       eval.episodes, 
                       policy_kwargs.net_arch,
                       # penalty,
                       penalty_decay,
                       restrict_alerts,
                       HI_restriction,
                       hi_rstr_decay,
                       hi_penalty,
                       # eval.match_similar,
                       # eval.val_years,
                       learning_rate)
colnames(training)<- c("county", 
                       "algo", 
                       # "match_similar",
                       # "explore_budget",
                       "eval.episodes",
                       "algo.policy_kwargs.net_arch", 
                       # "penalty",
                       "penalty_decay",
                       "restrict_alerts",
                       "HI_restriction",
                       "hi_rstr_decay",
                       "hi_penalty",
                       # "eval.match_similar",
                       # "eval.val_years",
                       "algo.learning_rate")

training$penalty<- c(0.0) # 0.01
# training[which(training$penalty_decay == "true"), "penalty"]<- 0.1
training$eval.freq<- 1000
training[which(training$eval.episodes == 100), "eval.freq"]<- 2500 # rather than 4000
training$training_timesteps<- 15000000 # original is 10 million
training[which(training$algo.learning_rate == 0.0001), "training_timesteps"]<- 100000000

results<- read.csv("Fall_results/Final_eval_30.csv")
training$HI_restriction<- results$opt_HI_thr

# training$HI_restriction<- 0.8
# training[which(training$county == 4013), "HI_restriction"]<- 0.7

training$model_name<- paste0("T8", "_fips-", training$county, 
                             # "_P-", training$penalty,
                             # "_", training$algo,
                             # "_obs-W",
                             # "_LR-", training$algo.learning_rate,
                             # "_EB-", training$explore_budget, 
                             # "_EE-", training$eval.episodes
                             # "_Rstr-HI-opt",
                             "_Rstr-HI-", training$HI_restriction #,
                             # "_Rstr-HI-decay-", training$hi_rstr_decay,
                             # "_PD-", training$penalty_decay,
                             # "_HIP-", training$hi_penalty,
                             # "_arch-", training$algo.policy_kwargs.net_arch
                             ) 
# training$model_name<- sapply(training$model_name, function(s){
#   x<- strsplit(s, "\\[")[[1]]
#   if(nchar(x[2]) < 4){
#     return(paste0(x[1], "small"))
#   }else{
#     return(paste0(x[1], "large"))
#   }
# })

# training$model_name<- sapply(training$model_name, function(s){
#   x<- strsplit(s, "arch-")[[1]]
#   if(x[2] == "[16,16]"){
#     return(paste0(x[1], "2-16"))
#   }else if(x[2] == "[16,16,16]"){
#     return(paste0(x[1], "3-16"))
#   }else if(x[2] == "[32,32]"){
#     return(paste0(x[1], "2-32"))
#   }else{
#     return(paste0(x[1], "1-16"))
#   }
# })

training_script<- "python train_online_rl_sb3.py"
evaluation_script<- "python old_evaluation_SB3.py"

Training<- sapply(1:ncol(training), function(i){paste0(colnames(training)[i], "=", training[,i])})
Short<- Training[which(training$training_timesteps < 100000000 & training$algo != "lstm"),]
Long<- Training[which(training$training_timesteps >= 100000000 | training$algo == "lstm"),]

sink("Run_jobs/Online_tests_short")
for(i in 1:nrow(Short)){
  cat(# evaluation_script,
    training_script,
      paste(
        Short[i,]
      ), " \n")
  # for(v in eval.val_years){
  #   for(m in eval.match_similar){
  #     cat(evaluation_script,
  #         paste0(
  #           Short[i,which(names(training) == "county")], " ",
  #           Short[i,which(names(training) == "algo")], " ",
  #           Short[i,which(names(training) == "algo.policy_kwargs.net_arch")], " ",
  #           Short[i,which(names(training) == "algo.learning_rate")], " ",
  #           "eval.val_years=", v,  " ",
  #           "eval.match_similar=", m, " ",
  #           Short[i,which(names(training) == "model_name")],
  #           " \n"
  #         ))
  #   }
  # }
  # cat(" \n")
}
sink()

sink("Run_jobs/Online_tests_long")
for(i in 1:nrow(Long)){
  cat(training_script,
      paste(
        Long[i,]
      ), " \n")
  # for(v in eval.val_years){
  #   for(m in eval.match_similar){
  #     cat(evaluation_script,
  #         paste0(
  #           Long[i,which(names(training) == "county")], " ",
  #           Long[i,which(names(training) == "algo")], " ",
  #           Long[i,which(names(training) == "algo.policy_kwargs.net_arch")], " ",
  #           Long[i,which(names(training) == "algo.learning_rate")], " ",
  #           "eval.val_years=", v,  " ",
  #           "eval.match_similar=", m, " ",
  #           Long[i,which(names(training) == "model_name")],
  #           " \n"
  #         ))
  #   }
  # }
  # cat(" \n")
}
sink()


## For NWS and NA:

# cd heat-alerts_mortality_RL/
# source activate heatrl

# counties<- c(41067, 53015, 20161, 37085, 48157, 
#              28049, 19153, 17167, 31153, 6071, 4013)
# 
# counties<- c(34021, 19155, 17115, 29021, 29019, 5045, 40017, 21059,
#              47113, 42017, 22109, 45015, 13031, 48367, 22063, 41053, 
#              32003, 4015, 6025)

counties<- c(41067, 53015, 20161, 37085, 48157, 
             28049, 19153, 17167, 31153, 6071, 4013,
             34021, 19155, 17115, 29021, 29019, 5045, 40017, 21059,
             47113, 42017, 22109, 45015, 13031, 48367, 22063, 41053, 
             32003, 4015, 6025)

sink("Run_jobs/Online_tests_short")
for(k in counties){
  county<- k
  
  # cat(paste0("python old_evaluation_SB3.py policy_type=NWS eval.val_years=true eval.match_similar=true ", "county=", county, "\n"),
  #     paste0("python old_evaluation_SB3.py policy_type=NWS eval.val_years=true eval.match_similar=false ", "county=", county, "\n"),
  #     paste0("python old_evaluation_SB3.py policy_type=NWS eval.val_years=false eval.match_similar=true ", "county=", county, "\n"),
  #     paste0("python old_evaluation_SB3.py policy_type=NWS eval.val_years=false eval.match_similar=false ", "county=", county, "\n"))

  # cat(paste0("python old_evaluation_SB3.py policy_type=NA eval.val_years=true eval.match_similar=true ", "county=", county, "\n"),
  #     paste0("python old_evaluation_SB3.py policy_type=NA eval.val_years=true eval.match_similar=false ", "county=", county, "\n"),
  #     paste0("python old_evaluation_SB3.py policy_type=NA eval.val_years=false eval.match_similar=true ", "county=", county, "\n"),
  #     paste0("python old_evaluation_SB3.py policy_type=NA eval.val_years=false eval.match_similar=false ", "county=", county, "\n"))
  
  # cat(paste0("python old_evaluation_SB3.py policy_type=random eval.val_years=true eval.match_similar=true ", "county=", county, "\n"),
  #     paste0("python old_evaluation_SB3.py policy_type=random eval.val_years=true eval.match_similar=false ", "county=", county, "\n"),
  #     paste0("python old_evaluation_SB3.py policy_type=random eval.val_years=false eval.match_similar=true ", "county=", county, "\n"),
  #     paste0("python old_evaluation_SB3.py policy_type=random eval.val_years=false eval.match_similar=false ", "county=", county, "\n"))

  for(h in HI_restriction){
    cat(paste0("python old_evaluation_SB3.py policy_type=random eval.val_years=true eval.match_similar=true restrict_alerts=true ",
               "county=", county, " HI_restriction=", h, " model_name=", "Rstr-HI-", h, " \n"),
        paste0("python old_evaluation_SB3.py policy_type=random eval.val_years=true eval.match_similar=false restrict_alerts=true ",
               "county=", county, " HI_restriction=", h, " model_name=", "Rstr-HI-", h, " \n") #,
        # paste0("python old_evaluation_SB3.py policy_type=random eval.val_years=false eval.match_similar=true restrict_alerts=true ",
        #        "county=", county, " HI_restriction=", h, " \n"),
        # paste0("python old_evaluation_SB3.py policy_type=random eval.val_years=false eval.match_similar=false restrict_alerts=true ",
        #        "county=", county, " HI_restriction=", h, " \n")
        )
  }
}
sink()


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


####### Scratch:

for(n in training$model_name){
  df<- read.csv(paste0("logs/SB/", n, "/training_metrics/progress.csv"))
  print(df[nrow(df),"time.total_timesteps"])
}



