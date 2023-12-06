library(evaluate)

counties<- c(41067, 53015, 20161, 37085, 48157, 
             28049, 19153, 17167, 31153, 6071, 4013,
             34021, 19155, 17115, 29021, 29019, 5045, 40017, 21059,
             47113, 42017, 22109, 45015, 13031, 48367, 22063, 41053, 
             32003, 4015, 6025)

r_model<- "mixed_constraints"

HI_thresholds<- seq(0.5, 0.9, 0.05)
Forecasts<- c("none", "Q_D10")
NHU<- c(16, 32, 64) 
NHL<- c(1, 2, 3)
n_steps<- c(1024, 2048, 4096)
algos<- c("trpo", "dqn")

i<- 1

sink("Run_jobs/Eval_jobs")

for(k in counties){
  county<- k
  for(r_model in c( "mixed_constraints"
                    # , "alert_constraints","all_constraints", "no_constraints", "hi_constraints"
  )){
    for(algo in algos){
      for(nhl in NHL){
        for(nhu in NHU){
          for(s in n_steps){
            
            if(i < 9964){
              if(nhl == 1){
                arch<- paste0("[", nhu, "]")
              }else if(nhl == 2){
                arch<- paste0("[", nhu, ",", nhu, "]")
              }else if(nhl == 3){
                arch<- paste0("[", nhu, ",", nhu, ",", nhu, "]")
              }
              
              if(arch=="[16,16]" & s==2048){
                cat(paste0("python old_evaluation_SB3.py policy_type=RL eval.val_years=false eval.match_similar=true ", 
                           "county=", county, " r_model=", r_model, " algo=", algo,
                           " restrict_days=none", " forecasts=none", " algo.policy_kwargs.net_arch=", arch, # " algo.n_steps=", s,
                           " model_name=", r_model, "_", algo, "_F-none", "_fips-", county, " \n"),
                    paste0("python old_evaluation_SB3.py policy_type=RL eval.val_years=true eval.match_similar=false ", 
                           "county=", county, " r_model=", r_model, " algo=", algo,
                           " restrict_days=none", " forecasts=none", " algo.policy_kwargs.net_arch=", arch, # " algo.n_steps=", s,
                           " model_name=", r_model, "_", algo, "_F-none", "_fips-", county, "\n")
                )
              }
              
              if(algo == "trpo"){
                for(forecasts in Forecasts){
                  for(h in HI_thresholds){
                    cat(paste0("python old_evaluation_SB3.py policy_type=RL eval.val_years=true eval.match_similar=true restrict_days=qhi ",
                               "county=", county, " restrict_days.HI_restriction=", h, " r_model=", r_model, 
                               " forecasts=", forecasts, " algo.policy_kwargs.net_arch=", arch, " algo.n_steps=", s,
                               " model_name=Tune_F-", forecasts, "_Rstr-HI-", h,
                               "_arch-", nhl, "-", nhu, "_ns-", s,
                               "_fips-", county, " \n"),
                        paste0("python old_evaluation_SB3.py policy_type=RL eval.val_years=true eval.match_similar=false restrict_days=qhi ",
                               "county=", county, " restrict_days.HI_restriction=", h, " r_model=", r_model, 
                               " forecasts=", forecasts, " algo.policy_kwargs.net_arch=", arch, " algo.n_steps=", s,
                               " model_name=Tune_F-", forecasts, "_Rstr-HI-", h,
                               "_arch-", nhl, "-", nhu, "_ns-", s,
                               "_fips-", county, " \n")
                    )
                  }
                }
              }
            }
            if(algo == "trpo"){
              i<- i+36
            }
            if(arch=="[16,16]" & s==2048){
              i<- i+2
            }
          }
        }
      }
    }
  }
}

sink()

i<- 1

sink("Run_jobs/Eval_jobs_2")

for(k in counties){
  county<- k
  for(r_model in c( "mixed_constraints"
                    # , "alert_constraints","all_constraints", "no_constraints", "hi_constraints"
  )){
    for(algo in algos){
      for(nhl in NHL){
        for(nhu in NHU){
          for(s in n_steps){
            
            if(i >= 9964 & i < (9964*2)){
              if(nhl == 1){
                arch<- paste0("[", nhu, "]")
              }else if(nhl == 2){
                arch<- paste0("[", nhu, ",", nhu, "]")
              }else if(nhl == 3){
                arch<- paste0("[", nhu, ",", nhu, ",", nhu, "]")
              }
              
              if(arch=="[16,16]" & s==2048){
                cat(paste0("python old_evaluation_SB3.py policy_type=RL eval.val_years=false eval.match_similar=true ", 
                           "county=", county, " r_model=", r_model, " algo=", algo,
                           " restrict_days=none", " forecasts=none", " algo.policy_kwargs.net_arch=", arch, # " algo.n_steps=", s,
                           " model_name=", r_model, "_", algo, "_F-none", "_fips-", county, " \n"),
                    paste0("python old_evaluation_SB3.py policy_type=RL eval.val_years=true eval.match_similar=false ", 
                           "county=", county, " r_model=", r_model, " algo=", algo,
                           " restrict_days=none", " forecasts=none", " algo.policy_kwargs.net_arch=", arch, # " algo.n_steps=", s,
                           " model_name=", r_model, "_", algo, "_F-none", "_fips-", county, "\n")
                )
              }
              
              if(algo == "trpo"){
                for(forecasts in Forecasts){
                  for(h in HI_thresholds){
                    cat(paste0("python old_evaluation_SB3.py policy_type=RL eval.val_years=true eval.match_similar=true restrict_days=qhi ",
                               "county=", county, " restrict_days.HI_restriction=", h, " r_model=", r_model, 
                               " forecasts=", forecasts, " algo.policy_kwargs.net_arch=", arch, " algo.n_steps=", s,
                               " model_name=Tune_F-", forecasts, "_Rstr-HI-", h,
                               "_arch-", nhl, "-", nhu, "_ns-", s,
                               "_fips-", county, " \n"),
                        paste0("python old_evaluation_SB3.py policy_type=RL eval.val_years=true eval.match_similar=false restrict_days=qhi ",
                               "county=", county, " restrict_days.HI_restriction=", h, " r_model=", r_model, 
                               " forecasts=", forecasts, " algo.policy_kwargs.net_arch=", arch, " algo.n_steps=", s,
                               " model_name=Tune_F-", forecasts, "_Rstr-HI-", h,
                               "_arch-", nhl, "-", nhu, "_ns-", s,
                               "_fips-", county, " \n")
                    )
                  }
                }
              }
            }
            if(algo == "trpo"){
              i<- i+36
            }
            if(arch=="[16,16]" & s==2048){
              i<- i+2
            }
          }
        }
      }
    }
  }
}

sink()

i<- 1

sink("Run_jobs/Eval_jobs_3")

for(k in counties){
  county<- k
  for(r_model in c( "mixed_constraints"
                    # , "alert_constraints","all_constraints", "no_constraints", "hi_constraints"
  )){
    for(algo in algos){
      for(nhl in NHL){
        for(nhu in NHU){
          for(s in n_steps){
            
            if(i >= (9964*2)){
              if(nhl == 1){
                arch<- paste0("[", nhu, "]")
              }else if(nhl == 2){
                arch<- paste0("[", nhu, ",", nhu, "]")
              }else if(nhl == 3){
                arch<- paste0("[", nhu, ",", nhu, ",", nhu, "]")
              }
              
              if(arch=="[16,16]" & s==2048){
                cat(paste0("python old_evaluation_SB3.py policy_type=RL eval.val_years=false eval.match_similar=true ", 
                           "county=", county, " r_model=", r_model, " algo=", algo,
                           " restrict_days=none", " forecasts=none", " algo.policy_kwargs.net_arch=", arch, # " algo.n_steps=", s,
                           " model_name=", r_model, "_", algo, "_F-none", "_fips-", county, " \n"),
                    paste0("python old_evaluation_SB3.py policy_type=RL eval.val_years=true eval.match_similar=false ", 
                           "county=", county, " r_model=", r_model, " algo=", algo,
                           " restrict_days=none", " forecasts=none", " algo.policy_kwargs.net_arch=", arch, # " algo.n_steps=", s,
                           " model_name=", r_model, "_", algo, "_F-none", "_fips-", county, "\n")
                )
              }
              
              if(algo == "trpo"){
                for(forecasts in Forecasts){
                  for(h in HI_thresholds){
                    cat(paste0("python old_evaluation_SB3.py policy_type=RL eval.val_years=true eval.match_similar=true restrict_days=qhi ",
                               "county=", county, " restrict_days.HI_restriction=", h, " r_model=", r_model, 
                               " forecasts=", forecasts, " algo.policy_kwargs.net_arch=", arch, " algo.n_steps=", s,
                               " model_name=Tune_F-", forecasts, "_Rstr-HI-", h,
                               "_arch-", nhl, "-", nhu, "_ns-", s,
                               "_fips-", county, " \n"),
                        paste0("python old_evaluation_SB3.py policy_type=RL eval.val_years=true eval.match_similar=false restrict_days=qhi ",
                               "county=", county, " restrict_days.HI_restriction=", h, " r_model=", r_model, 
                               " forecasts=", forecasts, " algo.policy_kwargs.net_arch=", arch, " algo.n_steps=", s,
                               " model_name=Tune_F-", forecasts, "_Rstr-HI-", h,
                               "_arch-", nhl, "-", nhu, "_ns-", s,
                               "_fips-", county, " \n")
                    )
                  }
                }
              }
            }
            if(algo == "trpo"){
              i<- i+36
            }
            if(arch=="[16,16]" & s==2048){
              i<- i+2
            }
          }
        }
      }
    }
  }
}

sink()



