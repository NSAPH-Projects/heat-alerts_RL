
counties<- c(41067, 53015, 20161, 37085, 48157, 
           28049, 19153, 17167, 31153, 6071, 4013,
           34021, 19155, 17115, 29021, 29019, 5045, 40017, 21059,
           47113, 42017, 22109, 45015, 13031, 48367, 22063, 41053, 
           32003, 4015, 6025)

r_model<- "mixed_constraints"

HI_thresholds<- seq(0.5, 0.9, 0.05)
Forecasts<- c("none", "Q_D10")
NHU<- c(16, 32) # c(16, 32, 64) # 32
NHL<- c(2, 3) # c(1, 2, 3) # 2
n_steps<- c(1500, 3000) # c(1024, 2048, 4096) # 2048

prefix<- "February"


sink("run_jobs/Online_tests_short")
for(k in counties){
  county<- k
  
  for(algo in c( # "trpo", "dqn"
                 "a2c", "qrdqn"
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
            
            if(algo %in% c("dqn", "qrdqn")){
              cat(paste0("python train_online_rl_sb3.py", " county=", county, " algo=", algo,
                         " deterministic=true",
                         " restrict_days=none", " forecasts=", forecasts,
                         " algo.policy_kwargs.net_arch=", arch, " algo.batch_size=", s,
                         " model_name=", prefix, "_", algo, "_F-", forecasts, "_Rstr-HI-", "none",
                         "_arch-", nhl, "-", nhu, "_ns-", s,
                         "_fips-", county, " \n"))
              
              for(h in HI_thresholds){
                cat(paste0("python train_online_rl_sb3.py", " county=", county, " algo=", algo,
                           " deterministic=true",
                           " restrict_days=qhi", " forecasts=", forecasts, " restrict_days.HI_restriction=", h,
                           " algo.policy_kwargs.net_arch=", arch, " algo.batch_size=", s,
                           " model_name=", prefix, "_", algo, "_F-", forecasts, "_Rstr-HI-", h,
                           "_arch-", nhl, "-", nhu, "_ns-", s,
                           "_fips-", county, " \n"))
              }
              
            }else{
              cat(paste0("python train_online_rl_sb3.py", " county=", county, " algo=", algo,
                         " deterministic=false",
                         # " deterministic=true", # sensitivity analysis
                         " restrict_days=none", " forecasts=", forecasts,
                         " algo.policy_kwargs.net_arch=", arch, " algo.n_steps=", s,
                         " model_name=", prefix, "_", algo, "_F-", forecasts, "_Rstr-HI-", "none",
                         "_arch-", nhl, "-", nhu, "_ns-", s,
                         "_fips-", county, " \n"))
              
              for(h in HI_thresholds){
                cat(paste0("python train_online_rl_sb3.py", " county=", county, " algo=", algo,
                           " deterministic=false",
                           # " deterministic=true", # sensitivity analysis
                           " restrict_days=qhi", " forecasts=", forecasts, " restrict_days.HI_restriction=", h,
                           " algo.policy_kwargs.net_arch=", arch, " algo.n_steps=", s,
                           " model_name=", prefix, "_", algo, "_F-", forecasts, "_Rstr-HI-", h,
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
sink()


