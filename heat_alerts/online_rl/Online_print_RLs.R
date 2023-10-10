
counties<- c(41067, 53015, 20161, 37085, 48157, 
           28049, 19153, 17167, 31153, 6071, 4013,
           34021, 19155, 17115, 29021, 29019, 5045, 40017, 21059,
           47113, 42017, 22109, 45015, 13031, 48367, 22063, 41053, 
           32003, 4015, 6025)

r_model<- "mixed_constraints"

sink("Run_jobs/Online_tests_short") # total = 2700; without DQN = 2100; without individual forecast vars or DQN = 600
for(k in counties){
  county<- k
  
  for(algo in c( "trpo" #,
                # "dqn",
                # "ppo"
  )){
      if(algo %in% c("dqn", "ppo")){
        for(forecasts in c("none", "all")){
          cat(paste0("python train_online_rl_sb3.py", " county=", county, " r_model=", r_model, " algo=", algo,
                     " restrict_days=none", " forecasts=", forecasts, 
                     " model_name=", r_model, "_", algo, "_F-", forecasts, "_fips-", county, " \n"))
          for(h in seq(0.5, 0.9, 0.05)){
            cat(paste0("python train_online_rl_sb3.py", " county=", county, " r_model=", r_model, " algo=", algo,
                       " restrict_days=qhi", " forecasts=", forecasts, " restrict_days.HI_restriction=", h, 
                       " model_name=", r_model, "_", algo, "_F-", forecasts, "_Rstr-HI-", h, "_fips-", county, " \n"))
          }
        }
      }else{
        for(forecasts in c( # "none", "all",
                           "num_elig", "quarters", "three_day", "ten_day", "quantiles", "N_Av4_D3"
        )){
          cat(paste0("python train_online_rl_sb3.py", " county=", county, " r_model=", r_model, " algo=trpo",
                     " restrict_days=none", " forecasts=", forecasts,
                     " model_name=", r_model, "_trpo", "_F-", forecasts, "_fips-", county, " \n"))
          for(h in seq(0.5, 0.9, 0.05)){
            cat(paste0("python train_online_rl_sb3.py", " county=", county, " r_model=", r_model, " algo=trpo",
                       " restrict_days=qhi", " forecasts=", forecasts, " restrict_days.HI_restriction=", h, 
                       " model_name=", r_model, "_trpo", "_F-", forecasts, "_Rstr-HI-", h, "_fips-", county, " \n"))
        }
      }
    }
  }
}
sink()


#### Hyperparameter tuning:

# r_model<- "mixed_constraints"

tune_counties<- c(32003, 29019, 45015, 19153, 41053)
tune_HI<- c(0.55, 0.7, 0.85)
tune_forecasts<- c("none", "all", "quantiles")
NHU<- c(16, 32, 64)
NHL<- c(1, 2, 3)
LR<- c(0.001, 0.0001, 0.01)
gamma<- c(1, 0.999, 0.99)
n_steps<- c(2048, 4096, 512)


sink("Run_jobs/Online_tuning") 
for(k in tune_counties){
  county<- k
  
  for(algo in c( "trpo"
                 # , "ppo"
  )){
    for(forecasts in tune_forecasts){
      for(nhl in NHL){
        for(nhu in NHU){
          for(lr in LR){
            for(g in gamma){
              for(s in n_steps){
                
                if(nhl == 1){
                  arch<- paste0("[", nhu, "]")
                }else if(nhl == 2){
                  arch<- paste0("[", nhu, ",", nhu, "]")
                }else if(nhl == 3){
                  arch<- paste0("[", nhu, ",", nhu, ",", nhu, "]")
                }
                
                # cat(paste0("python train_online_rl_sb3.py", " county=", county,
                #            " restrict_days=none", " forecasts=", forecasts,
                #            " algo.policy_kwargs.net_arch=", arch, " algo.learning_rate=", lr, 
                #            " algo.gamma=", g, " algo.n_steps=", s,
                #            " model_name=tune_F-", forecasts, "_fips-", county, " \n"))
                for(h in tune_HI){
                  cat(paste0("python train_online_rl_sb3.py", " county=", county,
                             " restrict_days=qhi", " forecasts=", forecasts, " restrict_days.HI_restriction=", h, 
                             " algo.policy_kwargs.net_arch=", arch, " algo.learning_rate=", lr, 
                             " algo.gamma=", g, " algo.n_steps=", s,
                             " model_name=tune_F-", forecasts, "_Rstr-HI-", h, 
                             "_arch-", nhl, "-", nhu, "_lr-", lr, "_g-", g, "_ns-", s,
                             "_fips-", county, " \n"))
                }
              }
            }
          }
        }
      }
    }
  }
}
sink()






#### If we want to do a sensitivity analysis across r_models:

# sink("Run_jobs/Online_tests_short") # 3900 models total; without DQN = 3300; without DQN or individual forecast vars = 1800
# for(k in counties){
#   county<- k
#   
#   for(algo in c("trpo"
#                 # , "dqn"
#                 )){
#     for(r_model in c("all_constraints", "no_constraints", "hi_constraints")){
#       if(algo == "dqn"){
#         if(r_model == "all_constraints"){
#           for(forecasts in c("none", "all")){
#             cat(paste0("python train_online_rl_sb3.py", " r_model=", r_model, " algo=dqn",
#                        " restrict_days=none", " forecasts=", forecasts, 
#                        " model_name=", r_model, "_dqn", "_F-", forecasts, " \n"))
#             for(h in seq(0.5, 0.9, 0.05)){
#               cat(paste0("python train_online_rl_sb3.py", " r_model=", r_model, " algo=dqn",
#                          " restrict_days=qhi", " forecasts=", forecasts, " restrict_days.HI_restriction=", h, 
#                          " model_name=", r_model, "_dqn", "_F-", forecasts, "_Rstr-HI-", h, " \n"))
#             }
#           }
#         } # pass on dqn and other r_models
#       }else{
#         if(r_model == "all_constraints"){
#           for(forecasts in c("none", "all"
#                              # , "N", "Av4", "D3", "D10", "Q"
#                              )){
#             cat(paste0("python train_online_rl_sb3.py", " r_model=", r_model, " algo=trpo",
#                        " restrict_days=none", " forecasts=", forecasts,
#                        " model_name=", r_model, "_trpo", "_F-", forecasts, " \n"))
#             for(h in seq(0.5, 0.9, 0.05)){
#               cat(paste0("python train_online_rl_sb3.py", " r_model=", r_model, " algo=trpo",
#                          " restrict_days=qhi", " forecasts=", forecasts, " restrict_days.HI_restriction=", h, 
#                          " model_name=", r_model, "_trpo", "_F-", forecasts, "_Rstr-HI-", h, " \n"))
#             }
#           }
#         }else{
#           for(forecasts in c("none", "all")){
#             cat(paste0("python train_online_rl_sb3.py", " r_model=", r_model, " algo=trpo",
#                        " restrict_days=none", " forecasts=", forecasts,
#                        " model_name=", r_model, "_trpo", "_F-", forecasts, " \n"))
#             for(h in seq(0.5, 0.9, 0.05)){
#               cat(paste0("python train_online_rl_sb3.py", " r_model=", r_model, " algo=trpo",
#                          " restrict_days=qhi", " forecasts=", forecasts, " restrict_days.HI_restriction=", h, 
#                          " model_name=", r_model, "_trpo", "_F-", forecasts, "_Rstr-HI-", h, " \n"))
#             }
#           }
#         }
#       }
#     }
#   }
# }
# sink()







