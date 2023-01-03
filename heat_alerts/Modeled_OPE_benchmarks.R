
## Setup
load("data/Small_S-A-R_prepped.RData")

n_days<- 153
H<- n_days-1
n<- length(A)/H

gamma<- 0.999
# Gamma<- rep(cumprod(rep(gamma, H))/gamma, n) 

## Using modeled rewards, not accounting for modeled past rewards:

pred_deaths<- read.csv("Fall_results/R_1-2_deaths.csv")
Pred_deaths<- sapply(1:length(A), function(i) pred_deaths[i,A[i]+2])
pred_OH<- read.csv("Fall_results/R_1-2_other-hosps.csv")
Pred_OH<- sapply(1:length(A), function(i) pred_OH[i,A[i]+2])
pred_hosps<- read.csv("Fall_results/R_1-2_all-hosps.csv")
Pred_hosps<- sapply(1:length(A), function(i) pred_hosps[i,A[i]+2])


## Using modeled rewards, accounting for modeled past rewards:

Mod2<- read.csv("Fall_results/Estimated_rewards_NWS_behavior_policy.csv")

#### Benchmark:

get_OPE<- function(Deaths, All_hosps, Other_hosps, discount = TRUE){
  this_n<- length(Deaths)/H
  this_Gamma<- rep(cumprod(rep(gamma, H))/gamma, this_n) 
  if(discount){
    return(c( (this_Gamma %*% Deaths)/this_n,
              (this_Gamma %*% All_hosps)/this_n,
              (this_Gamma %*% Other_hosps)/this_n ))
    
  }else{
    return(c( sum(Deaths)/this_n,
              sum(All_hosps)/this_n,
              sum(Other_hosps)/this_n ))
  }
}

#### Discounted:

Observed_NWS<- get_OPE(R_deaths[,1], R_all_hosps[,1], R_other_hosps[,1])
Modeled_NWS<- get_OPE(Pred_deaths, Pred_hosps, Pred_OH)
Mod2_NWS<- get_OPE(Mod2$X0, Mod2$X1, Mod2$X2)

results<- data.frame(Observed_NWS, Modeled_NWS)
Results<- t(results)
colnames(Results)<- c("Deaths", "All hospitalizations", "Non-heat hospitalizations")
Results


#### Not discounted:

Observed_NWS_nd<- get_OPE(R_deaths[,1], R_all_hosps[,1], R_other_hosps[,1], 
                          discount = FALSE)
Modeled_NWS_nd<- get_OPE(Pred_deaths, Pred_hosps, Pred_OH, discount = FALSE)

results_nd<- data.frame(Observed_NWS_nd, Modeled_NWS_nd)
Results_nd<- t(results_nd)
colnames(Results_nd)<- c("Deaths", "All hospitalizations", "Non-heat hospitalizations")
Results_nd

