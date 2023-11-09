
source("heat_alerts/scripts/Evaluation_functions.R")

# type<- "training"
type<- "evaluation"

## Change manually:
# eval_func_name<- "compare_to_zero" 
eval_func_name<- "avg_return"

#### Run evaluations: 

if(eval_func_name == "compare_to_zero"){ # if using compare_to_zero, need additional argument below
  eval_func<- compare_to_zero 
}else if(eval_func_name == "avg_return"){
  eval_func<- avg_return
}


#### Expand grid:
tune_counties<- c(32003, 29019, 45015, 19153, 41053)
tune_HI<- c(0.55, 0.7, 0.85)
# tune_forecasts<- c("none", "all")
# tune_forecasts<- "quantiles"
tune_forecasts<- c("none", "all", "quantiles")
NHU<- c(16, 32, 64)
NHL<- c(1, 2, 3)
LR<- c(0.001, 0.0001, 0.01)
gamma<- c(1, 0.999, 0.99)
n_steps<- c(2048, 4096, 512)

## First time:
filling<- FALSE
DF<- expand.grid(tune_counties, tune_HI, tune_forecasts, 
                 NHU, NHL, LR, gamma, n_steps)
names(DF)<- c("tune_counties", "tune_HI", "tune_forecasts", 
              "NHU", "NHL", "LR", "gamma", "n_steps")

DF$Eval<- 0

## Filling in:
filling<- TRUE
DF<- read.csv(paste0("Fall_results/Tuning_evals_", type, "_", eval_func_name, ".csv"))[,-1]


## If eval_func != compare_to_zero:

start<- Sys.time()

for(i in 1:nrow(DF)){
  m<- paste0("tune_F-", DF$tune_forecasts[i], "_Rstr-HI-", DF$tune_HI[i], 
             "_arch-", DF$NHL[i], "-", DF$NHU[i], 
             "_lr-", DF$LR[i], "_g-", DF$gamma[i], "_ns-", DF$n_steps[i],
             "_fips-", DF$tune_counties[i])
  if(filling & is.na(DF$Eval[i])){
    # DF$Eval[i]<- eval_func(paste0("Summer_results/ORL_RL_train_samp-R_samp-W_", m, "_fips_", DF$tune_counties[i], ".csv"))
    DF$Eval[i]<- eval_func(paste0("Summer_results/ORL_RL_eval_samp-R_obs-W_", m, "_fips_", DF$tune_counties[i], ".csv"))
    print(i)
  }else if(!filling){
    # DF$Eval[i]<- eval_func(paste0("Summer_results/ORL_RL_train_samp-R_samp-W_", m, "_fips_", DF$tune_counties[i], ".csv"))
    DF$Eval[i]<- eval_func(paste0("Summer_results/ORL_RL_eval_samp-R_obs-W_", m, "_fips_", DF$tune_counties[i], ".csv"))
    if(i %% 50 == 0){
      print(paste(i, "=", Sys.time() - start))
    }
  }
}

f<- paste0("Fall_results/Tuning_evals_", type, "_", eval_func_name, ".csv")
if(file.exists(f)){
  old<- read.csv(f)[,-1]
  DF<- rbind(old, DF)
}
write.csv(DF, f)


#########################################

# ## If eval_func == compare_to_zero:
# 
# start<- Sys.time()
# 
# for(i in 1:nrow(DF)){
#   m<- paste0("tune_F-", DF$tune_forecasts[i], "_Rstr-HI-", DF$tune_HI[i], 
#              "_arch-", DF$NHL[i], "-", DF$NHU[i], 
#              "_lr-", DF$LR[i], "_g-", DF$gamma[i], "_ns-", DF$n_steps[i],
#              "_fips-", DF$tune_counties[i])
#   
#   filename_zero_train<- paste0("Summer_results/ORL_NA_train_samp-R_samp-W_mixed_constraints_fips_", DF$tune_counties[i], ".csv")
#   
#   if(filling & is.na(DF$Eval[i])){
#     DF$Eval[i]<- eval_func(paste0("Summer_results/ORL_RL_train_samp-R_samp-W_", m, "_fips_", DF$tune_counties[i], ".csv"),
#                            filename_zero_train)
#     if(i %% 50 == 0){
#       print(paste(i, "=", Sys.time() - start))
#     }
#   }else if(!filling){
#     DF$Eval[i]<- eval_func(paste0("Summer_results/ORL_RL_train_samp-R_samp-W_", m, "_fips_", DF$tune_counties[i], ".csv"),
#                            filename_zero_train)
#     if(i %% 50 == 0){
#       print(paste(i, "=", Sys.time() - start))
#     }
#   }
# }
# 
# f<- paste0("Fall_results/Tuning_evals_", type, "_", eval_func_name, ".csv")
# if(file.exists(f)){
#   old<- read.csv(f)[,-1]
#   DF<- rbind(old, DF)
# }
# write.csv(DF, f)

#### Select the best HI threshold for each county-hyperparameter combo:

library(dplyr)

tune_counties<- c(32003, 29019, 45015, 19153, 41053)

## Change manually:
# eval_func_name<- "compare_to_zero" 
eval_func_name<- "avg_return"

DF<- read.csv(paste0("Fall_results/Tuning_evals_", type, "_", eval_func_name, ".csv"))[,-1]

DF$Index<- 1:nrow(DF)

vars<- c("tune_counties", "tune_forecasts", "NHU", "NHL", "LR", "gamma", "n_steps")
opt_DF<- distinct(DF[,vars])

opt_DF$Eval<- 0
opt_DF$OT<- 0

for(j in 1:nrow(opt_DF)){
  pos<- inner_join(opt_DF[j,vars], DF[,c(vars, "Index")], by=vars)$Index
  m<- which.max(DF[pos, "Eval"])
  opt_DF[j, "Eval"]<- DF[pos[m],"Eval"]
  opt_DF[j, "OT"]<- DF[pos[m],"tune_HI"]
  
  if(j %% 50 == 0){
    print(j)
  }
}

write.csv(opt_DF, paste0("Fall_results/Tuning_", type, "_opt-HI_", eval_func_name, ".csv"), row.names=FALSE)

#### Inspect results: regress Eval on c(tune_forecasts, NHU, NHL, LR, gamma, n_steps)

library(ggplot2)

opt_DF<- read.csv(paste0("Fall_results/Tuning_", type, "_opt-HI_", eval_func_name, ".csv"))

# fit0<- lm(Eval ~ tune_counties + tune_forecasts + NHU + NHL +
#             LR + gamma + n_steps, opt_DF)

for(k in tune_counties){
  # print(k)
  
  pos<- which(opt_DF[,"tune_counties"] == k)
  m<- which.max(opt_DF[pos, "Eval"])
  # m2<- which.max(opt_DF[pos[-m], "Eval"])
  print(opt_DF[pos[m],])
  # print(opt_DF[pos[-m][m2],])
  print(summary(opt_DF[pos,"Eval"]))
  
  # fit<- lm(Eval ~ tune_forecasts + NHU + NHL + NHU*NHL +
  #            LR + gamma + n_steps, opt_DF[which(opt_DF$tune_counties == k),])
  # print(summary(fit))
}


library(nlme)

lmm.0<- lme(Eval ~ tune_forecasts + NHU + NHL + NHU*NHL + as.factor(n_steps),
          random = ~1|tune_counties, opt_DF)

lmm.1<- lme(Eval ~ tune_forecasts + NHU + NHL + NHU*NHL + as.factor(n_steps),
            random = ~1 + as.factor(n_steps)|tune_counties, opt_DF)

lmm.2<- lme(Eval ~ tune_forecasts + NHU + as.factor(NHL) + as.factor(n_steps),
            random = ~1 + as.factor(n_steps)|tune_counties, opt_DF) # best for training set

# lmm.3<- lme(Eval ~ tune_forecasts + NHU + NHL + NHU*NHL + as.factor(n_steps),
#             random = ~1 + tune_forecasts|tune_counties, opt_DF)
# 
# lmm.4<- lme(Eval ~ tune_forecasts + NHU + NHL + NHU*NHL + as.factor(n_steps),
#             random = ~1 + NHL|tune_counties, opt_DF)

preds<- rowSums(lmm.2$fitted)

for(k in tune_counties){
  print(k)
  
  pos<- which(opt_DF[,"tune_counties"] == k)
  m<- which.max(preds[pos])
  print(opt_DF[pos[m],])
  print(summary(opt_DF[pos,"Eval"]))
}

