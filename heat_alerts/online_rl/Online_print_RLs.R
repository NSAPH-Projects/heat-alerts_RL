
counties<- c(41067, 53015, 20161, 37085, 48157, 
           28049, 19153, 17167, 31153, 6071, 4013,
           34021, 19155, 17115, 29021, 29019, 5045, 40017, 21059,
           47113, 42017, 22109, 45015, 13031, 48367, 22063, 41053, 
           32003, 4015, 6025)

r_model<- "mixed_constraints"

HI_thresholds<- seq(0.5, 0.9, 0.05)
Forecasts<- c("Q_D10") # "none"
NHU<- c(16, 32, 64)
NHL<- c(1, 2, 3)
n_steps<- c(1024, 2048, 4096)
# NHU<- c(32)
# NHL<- c(2)
# n_steps<- c(2048)

missing<- c()
i<- 1

## First batch:

sink("Run_jobs/Online_tests_short")
for(k in counties){
  county<- k
  
  for(algo in c( "trpo"
                 # , "ppo"
  )){
    for(forecasts in Forecasts){
      for(nhl in NHL){
        for(nhu in NHU){
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
            #            " algo.policy_kwargs.net_arch=", arch, " algo.n_steps=", s,
            #            " model_name=Tune_F-", forecasts, "_fips-", county, " \n"))
            for(h in HI_thresholds){
              f<- paste0("Summer_results/ORL_RL_eval_samp-R_samp-W_",
                         "Tune_F-", forecasts, "_Rstr-HI-", h,
                         "_arch-", nhl, "-", nhu, "_ns-", s,
                         "_fips-", county, "_fips_", county, ".csv")
              if(!file.exists(f)){
                if(!(nhl==2 & nhu==32 & s==2048)){
                }else{
                  missing<- append(missing, i)
                  cat(paste0("python train_online_rl_sb3.py", " county=", county,
                             " restrict_days=qhi", " forecasts=", forecasts, " restrict_days.HI_restriction=", h, 
                             " algo.policy_kwargs.net_arch=", arch, " algo.n_steps=", s,
                             " model_name=Tune_F-", forecasts, "_Rstr-HI-", h, 
                             "_arch-", nhl, "-", nhu, "_ns-", s,
                             "_fips-", county, " \n"))
                }
                i<- i+1
              }
            }
          }
        }
      }
    }
  }
}
sink()


## The rest:

sink("Run_jobs/Online_tuning")
for(k in counties){
  county<- k
  
  for(algo in c( "trpo"
                 # , "ppo"
  )){
    for(forecasts in Forecasts){
      for(nhl in NHL){
        for(nhu in NHU){
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
            #            " algo.policy_kwargs.net_arch=", arch, " algo.n_steps=", s,
            #            " model_name=Tune_F-", forecasts, "_fips-", county, " \n"))
            for(h in HI_thresholds){
              # f<- paste0("Summer_results/ORL_RL_eval_samp-R_samp-W_",
              #            "Tune_F-", forecasts, "_Rstr-HI-", h,
              #            "_arch-", nhl, "-", nhu, "_ns-", s,
              #            "_fips-", county, "_fips_", county, ".csv")
              # if(!file.exists(f)){
                d<- paste0("logs/SB/Tune_F-", forecasts, "_Rstr-HI-", h,
                           "_arch-", nhl, "-", nhu, "_ns-", s,
                           "_fips-", county)
                if(!file.exists(d)){
                if((nhl==2 & nhu==32 & s==2048)){
                }else{
                  missing<- append(missing, i)
                  cat(paste0("python train_online_rl_sb3_continue.py", " county=", county,
                             " restrict_days=qhi", " forecasts=", forecasts, " restrict_days.HI_restriction=", h, 
                             " algo.policy_kwargs.net_arch=", arch, " algo.n_steps=", s,
                             " model_name=Tune_F-", forecasts, "_Rstr-HI-", h, 
                             "_arch-", nhl, "-", nhu, "_ns-", s,
                             "_fips-", county, " \n"))
                }
                i<- i+1
              }
            }
          }
        }
      }
    }
  }
}
sink()


### Re-running those that didn't make it far enough:

# sink("Run_jobs/Online_tuning")
sink("Run_jobs/Eval_jobs")
for(k in counties){
  county<- k
  
  for(algo in c( "trpo"
                 # , "ppo"
  )){
    for(forecasts in Forecasts){
      for(nhl in NHL){
        for(nhu in NHU){
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
            #            " algo.policy_kwargs.net_arch=", arch, " algo.n_steps=", s,
            #            " model_name=Tune_F-", forecasts, "_fips-", county, " \n"))
            for(h in HI_thresholds){
              f<- paste0("Summer_results/ORL_RL_eval_samp-R_samp-W_",
                         "Tune_F-", forecasts, "_Rstr-HI-", h,
                         "_arch-", nhl, "-", nhu, "_ns-", s,
                         "_fips-", county, "_fips_", county, ".csv")
              if(!file.exists(f)){
                test<- tryCatch(read.csv(paste0("logs/SB/",
                                                "Tune_F-", forecasts, "_Rstr-HI-", h,
                                                "_arch-", nhl, "-", nhu, "_ns-", s,
                                                "_fips-", county, "/training_metrics/progress.csv")), error=function(e){NULL})
                if(is.null(test)){
                  over_13m<- FALSE
                  last_t<- 10
                  best_eval_t<- 0
                }else{
                  progress<- read.csv(paste0("logs/SB/",
                                             "Tune_F-", forecasts, "_Rstr-HI-", h,
                                             "_arch-", nhl, "-", nhu, "_ns-", s,
                                             "_fips-", county, "/training_metrics/progress.csv"))
                  last_t<- progress$time.total_timesteps[nrow(progress)]
                  over_13m<- last_t >= 13000000
                  if(!is.null(progress$eval.mean_reward)){
                    best_eval_t<- progress$time.total_timesteps[which.max(progress$eval.mean_reward)]
                  }else{
                    best_eval_t<- 0
                  }
                }
                
                if((!over_13m) & (last_t - best_eval_t < 3000000)){
                  # cat(paste0("python train_online_rl_sb3.py", " county=", county,
                  #            " restrict_days=qhi", " forecasts=", forecasts, " restrict_days.HI_restriction=", h,
                  #            " algo.policy_kwargs.net_arch=", arch, " algo.n_steps=", s,
                  #            " model_name=Tune_F-", forecasts, "_Rstr-HI-", h,
                  #            "_arch-", nhl, "-", nhu, "_ns-", s,
                  #            "_fips-", county, " \n"))
                }else{
                  # print(paste0("Rstr-HI-", h,
                  #              "_arch-", nhl, "-", nhu, "_ns-", s,
                  #              "_fips-", county, ": last_t=", last_t/1000000, "m, diff=", (last_t - best_eval_t)))
                  cat(paste0("python old_evaluation_SB3.py eval.val_years=true eval.match_similar=true", " county=", county,
                             " restrict_days=qhi", " forecasts=", forecasts, " restrict_days.HI_restriction=", h,
                             " algo.policy_kwargs.net_arch=", arch, " algo.n_steps=", s,
                             " model_name=Tune_F-", forecasts, "_Rstr-HI-", h,
                             "_arch-", nhl, "-", nhu, "_ns-", s,
                             "_fips-", county, " \n"))
                  cat(paste0("python old_evaluation_SB3.py eval.val_years=true eval.match_similar=false", " county=", county,
                             " restrict_days=qhi", " forecasts=", forecasts, " restrict_days.HI_restriction=", h,
                             " algo.policy_kwargs.net_arch=", arch, " algo.n_steps=", s,
                             " model_name=Tune_F-", forecasts, "_Rstr-HI-", h,
                             "_arch-", nhl, "-", nhu, "_ns-", s,
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



############################ If we want to do a sensitivity analysis across r_models:

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







