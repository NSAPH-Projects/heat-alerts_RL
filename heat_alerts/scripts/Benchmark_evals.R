
source("heat_alerts/scripts/Evaluation_functions.R")

## Change manually:
eval_func_name<- "avg_return" # "per_alert" # "compare_to_zero"

#### Run evaluations:

if(eval_func_name == "per_alert"){
  eval_func<- per_alert
}else if(eval_func_name == "compare_to_zero"){ # if using compare_to_zero, need additional argument below
  eval_func<- compare_to_zero 
}else if(eval_func_name == "avg_return"){
  eval_func<- avg_return
}

HI_thresholds<- seq(0.5, 0.9, 0.05)

## If eval_func != compare_to_zero:

for(r_model in c("mixed_constraints" 
  # , "alert_constraints", "all_constraints", "no_constraints", "hi_constraints"
)){
  Zero<- rep(0,length(counties))
  NWS<- rep(0,length(counties))
  basic_NWS<- rep(0,length(counties))
  Random<- rep(0,length(counties))
  Top_K<- rep(0,length(counties))
  Random_QHI<- rep(0,length(counties))
  AA_QHI<- rep(0,length(counties))
  rqhi_ot<- rep(0,length(counties))
  aqhi_ot<- rep(0,length(counties))
  
  # old_DF<- read.csv(paste0("Fall_results/Benchmarks_", r_model, "_", eval_func_name, ".csv"))
  
  for(k in 1:length(counties)){
    county<- counties[k]
    
    Zero[k]<- eval_func(paste0("Summer_results/ORL_NA_eval_samp-R_obs-W_", r_model, "_fips_", county, ".csv"))
    NWS[k]<- eval_func(paste0("Summer_results/ORL_NWS_eval_samp-R_obs-W_", r_model, "_fips_", county, ".csv"))
    basic_NWS[k]<- eval_func(paste0("Summer_results/ORL_basic-NWS_eval_samp-R_obs-W_", r_model, "_fips_", county, ".csv"))
    Random[k]<- eval_func(paste0("Summer_results/ORL_random_eval_samp-R_obs-W_", r_model, "_fips_", county, ".csv"))
    Top_K[k]<- eval_func(paste0("Summer_results/ORL_TK_eval_samp-R_obs-W_", r_model, "_fips_", county, ".csv"))

    Models<- paste0(r_model, "_Rstr-HI-", HI_thresholds)
    for(pol in c("random", "AA")){
      for(i in 1:length(Models)){
        m<- Models[i]
        eval_samp<- eval_func(paste0("Summer_results/ORL_", pol, "_eval_samp-R_samp-W_", m, "_fips_", county, ".csv"))
        eval<- eval_func(paste0("Summer_results/ORL_", pol, "_eval_samp-R_obs-W_", m, "_fips_", county, ".csv"))
        if(i == 1){
          Eval_samp<- eval_samp
          Eval<- eval
          j<- 1
        }else{
          if(eval_samp > Eval_samp){
            Eval_samp<- eval_samp
            Eval<- eval
            j<- i
          }
        }
      }
      if(pol == "random"){
        Random_QHI[k]<- Eval
        rqhi_ot[k]<- HI_thresholds[j]
      }else{
        AA_QHI[k]<- Eval
        aqhi_ot[k]<- HI_thresholds[j]
      }
    }
    print(county)
  }
  DF<- round(data.frame(County=counties, Zero, NWS, basic_NWS, Random, Top_K
                        , Random_QHI, AA_QHI, rqhi_ot, aqhi_ot
                        ),3)
  # DF$Random_QHI<- old_DF$Random_QHI
  # DF$AA_QHI<- old_DF$AA_QHI
  # DF$rqhi_ot<- old_DF$rqhi_ot
  # DF$aqhi_ot<- old_DF$aqhi_ot
  
  write.csv(DF, paste0("Fall_results/Benchmarks_", r_model, "_", eval_func_name, ".csv"))
  print(paste("Finished with", r_model))
}

# basic_NWS<- rep(0,length(counties))
# for(k in 1:length(counties)){
#   county<- counties[k]
#   basic_NWS[k]<- eval_func(paste0("Summer_results/ORL_basic-NWS_eval_samp-R_obs-W_", "mixed_constraints", "_fips_", county, ".csv"))
# }
# 
# DF<- read.csv(paste0("Fall_results/Benchmarks_", "mixed_constraints", "_", eval_func_name, ".csv"))[,-1]
# DF$basic_NWS<- basic_NWS
# write.csv(DF, paste0("Fall_results/Benchmarks_", "mixed_constraints", "_", eval_func_name, ".csv"))


## If eval_func == compare_to_zero:

for(r_model in c( # "mixed_constraints", 
                 "alert_constraints"
                  , "all_constraints", "no_constraints", "hi_constraints"
)){
  NWS<- rep(0,length(counties))
  basic_NWS<- rep(0,length(counties))
  Random<- rep(0,length(counties))
  Top_K<- rep(0,length(counties))
  Random_QHI<- rep(0,length(counties))
  AA_QHI<- rep(0,length(counties))
  rqhi_ot<- rep(0,length(counties))
  aqhi_ot<- rep(0,length(counties))
  
  # old_DF<- read.csv(paste0("Fall_results/Benchmarks_", r_model, "_", eval_func_name, ".csv"))
  
  for(k in 1:length(counties)){
    county<- counties[k]
    
    NWS[k]<- eval_func(paste0("Summer_results/ORL_NWS_eval_samp-R_obs-W_", r_model, "_fips_", county, ".csv"),
                       paste0("Summer_results/ORL_NA_eval_samp-R_obs-W_", r_model, "_fips_", county, ".csv"))
    basic_NWS[k]<- eval_func(paste0("Summer_results/ORL_basic-NWS_eval_samp-R_obs-W_", r_model, "_fips_", county, ".csv"),
                       paste0("Summer_results/ORL_NA_eval_samp-R_obs-W_", r_model, "_fips_", county, ".csv"))
    Random[k]<- eval_func(paste0("Summer_results/ORL_random_eval_samp-R_obs-W_", r_model, "_fips_", county, ".csv"),
                          paste0("Summer_results/ORL_NA_eval_samp-R_obs-W_", r_model, "_fips_", county, ".csv"))
    Top_K[k]<- eval_func(paste0("Summer_results/ORL_TK_eval_samp-R_obs-W_", r_model, "_fips_", county, ".csv"),
                         paste0("Summer_results/ORL_NA_eval_samp-R_obs-W_", r_model, "_fips_", county, ".csv"))
    
    Models<- paste0(r_model, "_Rstr-HI-", HI_thresholds)
    for(pol in c("random", "AA")){
      for(i in 1:length(Models)){
        m<- Models[i]
        train_samp<- eval_func(paste0("Summer_results/ORL_", pol, "_train_samp-R_samp-W_", m, "_fips_", county, ".csv"),
                               paste0("Summer_results/ORL_NA_train_samp-R_samp-W_", r_model, "_fips_", county, ".csv"))
        eval<- eval_func(paste0("Summer_results/ORL_", pol, "_eval_samp-R_obs-W_", m, "_fips_", county, ".csv"),
                         paste0("Summer_results/ORL_NA_eval_samp-R_obs-W_", r_model, "_fips_", county, ".csv"))
        if(i == 1){
          Eval_samp<- train_samp
          Eval<- eval
          j<- 1
        }else{
          if(train_samp > Eval_samp){
            Eval_samp<- train_samp
            Eval<- eval
            j<- i
          }
        }
      }
      if(pol == "random"){
        Random_QHI[k]<- Eval
        rqhi_ot[k]<- HI_thresholds[j]
      }else{
        AA_QHI[k]<- Eval
        aqhi_ot[k]<- HI_thresholds[j]
      }
    }
    print(county)
  }
  DF<- round(data.frame(County=counties, NWS, basic_NWS, Random, Top_K
                        , Random_QHI, AA_QHI, rqhi_ot, aqhi_ot
  ),3)
  # DF$Random_QHI<- old_DF$Random_QHI
  # DF$AA_QHI<- old_DF$AA_QHI
  # DF$rqhi_ot<- old_DF$rqhi_ot
  # DF$aqhi_ot<- old_DF$aqhi_ot
  
  write.csv(DF, paste0("Fall_results/Benchmarks_", r_model, "_", eval_func_name, ".csv"))
  print(paste("Finished with", r_model))
}


### Inspect results:

for(r_model in c( "mixed_constraints"
  # , "alert_constraints", "all_constraints", "no_constraints", "hi_constraints"
)){
  print(r_model)
  DF<- read.csv(paste0("Fall_results/Benchmarks_", r_model, "_", eval_func_name, ".csv"))
  # print(DF)
  # hist(DF$NWS - DF$Zero, main=r_model)
  for(j in which(names(DF) %in% c("basic_NWS", "Random", "Random_QHI", "AA_QHI", "Top_K"))){
    wmw<- wilcox.test(DF[,j], DF$NWS, paired = TRUE, alternative = "greater", exact=FALSE)
    print(paste0(names(DF)[j], ": median_diff = ", round(median(DF[,j] - DF$NWS),3), 
                 ", WMW = ", wmw$statistic, ", p_val = ", round(wmw$p.value, 5)))
  }
  # print(paste("Random =", round(wilcox.test(DF$Random, DF$NWS, paired = TRUE, alternative = "greater", exact=FALSE)$p.value,8)))
  # print(paste("Random_QHI =", round(wilcox.test(DF$Random_QHI, DF$NWS, paired = TRUE, alternative = "greater", exact=FALSE)$p.value,8)))
  # print(paste("AA_QHI =", round(wilcox.test(DF$AA_QHI, DF$NWS, paired = TRUE, alternative = "greater", exact=FALSE)$p.value,8)))
  # print(paste("Top_K =", round(wilcox.test(DF$Top_K, DF$NWS, paired = TRUE, alternative = "greater", exact=FALSE)$p.value,8)))
}


