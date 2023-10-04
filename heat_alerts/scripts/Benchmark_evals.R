
source("heat_alerts/scripts/Evaluation_functions.R")

## Change manually:
eval_func_name<- "per_alert" # "compare_to_zero" # "avg_return"

## Run evaluations:

if(eval_func_name == "per_alert"){
  eval_func<- per_alert
}else if(eval_func_name == "compare_to_zero"){ # if using compare_to_zero, need additional argument below
  eval_func<- compare_to_zero 
}else if(eval_func_name == "avg_return"){
  eval_func<- avg_return
}

HI_thresholds<- seq(0.5, 0.9, 0.05)

for(r_model in c("alert_constraints"
  # ,"all_constraints", "no_constraints", "hi_constraints"
)){
  Zero<- rep(0,length(counties))
  NWS<- rep(0,length(counties))
  Random<- rep(0,length(counties))
  Top_K<- rep(0,length(counties))
  Random_QHI<- rep(0,length(counties))
  AA_QHI<- rep(0,length(counties))
  rqhi_ot<- rep(0,length(counties))
  aqhi_ot<- rep(0,length(counties))
  
  for(k in 1:length(counties)){
    county<- counties[k]
    
    Zero[k]<- eval_func(paste0("Summer_results/ORL_NA_eval_samp-R_obs-W_", r_model, "_fips_", county, ".csv"))
    NWS[k]<- eval_func(paste0("Summer_results/ORL_NWS_eval_samp-R_obs-W_", r_model, "_fips_", county, ".csv"))
    Random[k]<- eval_func(paste0("Summer_results/ORL_random_eval_samp-R_obs-W_", r_model, "_fips_", county, ".csv"))
    Top_K[k]<- eval_func(paste0("Summer_results/ORL_TK_eval_samp-R_obs-W_", r_model, "_fips_", county, ".csv"))

  #   Models<- paste0(r_model, "_Rstr-HI-", HI_thresholds)
  #   for(pol in c("random", "AA")){
  #     for(i in 1:length(Models)){
  #       m<- Models[i]
  #       train_samp<- eval_func(paste0("Summer_results/ORL_", pol, "_train_samp-R_samp-W_", m, "_fips_", county, ".csv"))
  #       eval<- eval_func(paste0("Summer_results/ORL_", pol, "_eval_samp-R_obs-W_", m, "_fips_", county, ".csv"))
  #       if(i == 1){
  #         Eval_samp<- train_samp
  #         Eval<- eval
  #         j<- 1
  #       }else{
  #         if(train_samp > Eval_samp){
  #           Eval_samp<- train_samp
  #           Eval<- eval
  #           j<- i
  #         }
  #       }
  #     }
  #     if(pol == "random"){
  #       Random_QHI[k]<- Eval
  #       rqhi_ot[k]<- HI_thresholds[j]
  #     }else{
  #       AA_QHI[k]<- Eval
  #       aqhi_ot[k]<- HI_thresholds[j]
  #     }
  #   }
  #   print(county)
  # }
  DF<- round(data.frame(County=counties, Zero, NWS, Random, Top_K,
                        Random_QHI, AA_QHI, rqhi_ot, aqhi_ot),3)
  # write.csv(DF, paste0("Fall_results/Benchmarks_", r_model, "_", eval_func_name, ".csv"))
  print(paste("Finished with", r_model))
}


### Inspect results:

for(r_model in c("alert_constraints"
  # ,"all_constraints", "no_constraints", "hi_constraints"
)){
  print(r_model)
  DF<- read.csv(paste0("Fall_results/Benchmarks_", r_model, "_", eval_func_name, ".csv"))
  print(DF)
  hist(DF$NWS - DF$Zero, main=r_model)
  # print(wilcox.test(DF$Random, DF$NWS, paired = TRUE, alternative = "greater", exact=FALSE))
  # print(wilcox.test(DF$Random_QHI, DF$NWS, paired = TRUE, alternative = "greater", exact=FALSE))
  # print(wilcox.test(DF$AA_QHI, DF$NWS, paired = TRUE, alternative = "greater", exact=FALSE))
  # print(wilcox.test(DF$Top_K, DF$NWS, paired = TRUE, alternative = "greater", exact=FALSE))
}


