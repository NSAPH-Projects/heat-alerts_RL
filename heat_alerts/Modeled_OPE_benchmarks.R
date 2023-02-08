
library(ggplot2)

## Setup
load("data/Small_S-A-R_prepped.RData")

n_days<- 153
H<- n_days-1
n<- length(A)/H

gamma<- 0.999
# Gamma<- rep(cumprod(rep(gamma, H))/gamma, n) 


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

subset_OPE<- function(Deaths, All_hosps, Other_hosps, discount = TRUE, 
                      split_vars, breaks = NULL){
  if(is.null(breaks)){
    results<- apply(DF[,split_vars], MARGIN=2, function(y){
      pos<- which(y == 1)
      return(get_OPE(Deaths[pos], All_hosps[pos], Other_hosps[pos], discount))
    })
    
    Results<- t(results)
    colnames(Results)<- c("Deaths", "All hospitalizations", "Non-heat hospitalizations")
    return(Results)
    
  }else{
    
    results<- sapply(1:length(breaks[-1]) + 1, function(i){
      if(i == length(breaks)){
        pos<- which(DF[,split_vars] >= breaks[i-1] & DF[,split_vars] <= breaks[i] )
      }else{
        pos<- which(DF[,split_vars] >= breaks[i-1] & DF[,split_vars] < breaks[i] )
      }
      return(get_OPE(Deaths[pos], All_hosps[pos], Other_hosps[pos], discount))
    })
    Results<- data.frame(t(results))
    row.names(Results)<- names(breaks[-1])
    colnames(Results)<- c("Deaths", "All hospitalizations", "Non-heat hospitalizations")
    return(Results)
  }
}

#### Get predictions:

## Using modeled rewards, not accounting for modeled past rewards:

pred_deaths<- read.csv("Fall_results/R_1-23_deaths.csv")
Pred_deaths<- sapply(1:length(A), function(i) pred_deaths[i,A[i]+2])
pred_OH<- read.csv("Fall_results/R_1-23_other-hosps.csv")
Pred_OH<- sapply(1:length(A), function(i) pred_OH[i,A[i]+2])
pred_hosps<- read.csv("Fall_results/R_1-23_all-hosps.csv")
Pred_hosps<- sapply(1:length(A), function(i) pred_hosps[i,A[i]+2])


## Using modeled rewards, accounting for modeled past rewards:

Mod2<- read.csv("Fall_results/Estimated_rewards_NWS_policy.csv")
Mod2_b<- read.csv("Fall_results/Estimated_rewards_NWS_hosps-only_policy.csv")
Mod2_oh<- read.csv("Fall_results/Estimated_rewards_NWS_OH-only_policy.csv")
Mod2_nh<- read.csv("Fall_results/Estimated_rewards_NWS_no-health-history_policy.csv")
# Mod2_c<- read.csv("Fall_results/Estimated_rewards_NWS_hosps-only-b_policy.csv")

## New policies:

None<- read.csv("Fall_results/Estimated_rewards_No_alerts_policy.csv")
None_b<- read.csv("Fall_results/Estimated_rewards_No_alerts_hosps-only_policy.csv")
None_oh<- read.csv("Fall_results/Estimated_rewards_No_alerts_OH-only_policy.csv")
None_nh<- read.csv("Fall_results/Estimated_rewards_No_alerts_no-health-history_policy.csv")
# None_c<- read.csv("Fall_results/Estimated_rewards_No_alerts_hosps-only-b_policy.csv")

DQN_nh<- read.csv("Fall_results/Estimated_rewards_DQN_2-5_hosps_no-health-history_policy.csv")

#### Discounted:

Observed_NWS<- get_OPE(R_deaths[,1], R_all_hosps[,1], R_other_hosps[,1])
Modeled_NWS<- get_OPE(Pred_deaths, Pred_hosps, Pred_OH)
Mod2_NWS<- get_OPE(Mod2$X0, Mod2$X1, Mod2$X2)
Mod2_NWS_b<- get_OPE(Mod2_b$X0, Mod2_b$X1, Mod2_b$X2)
Mod2_NWS_OH<- get_OPE(Mod2_oh$X0, Mod2_oh$X1, Mod2_oh$X2)
Mod2_NWS_NH<- get_OPE(Mod2_nh$X0, Mod2_nh$X1, Mod2_nh$X2)
# Mod2_NWS_c<- get_OPE(Mod2_c$X0, Mod2_c$X1, Mod2_c$X2)
No_alerts<- get_OPE(None$X0, None$X1, None$X2)
No_alerts_b<- get_OPE(None_b$X0, None_b$X1, None_b$X2)
No_alerts_OH<- get_OPE(None_oh$X0, None_oh$X1, None_oh$X2)
No_alerts_NH<- get_OPE(None_nh$X0, None_nh$X1, None_nh$X2)
# No_alerts_c<- get_OPE(None_c$X0, None_c$X1, None_c$X2)
DQN_NH<- get_OPE(DQN_nh$X0, DQN_nh$X1, DQN_nh$X2)

results<- data.frame(Observed_NWS, Modeled_NWS, Mod2_NWS,
                     Mod2_NWS_b, Mod2_NWS_OH, Mod2_NWS_NH,
                     # Mod2_NWS_c,
                     No_alerts, 
                     No_alerts_b, No_alerts_OH, No_alerts_NH,
                     # No_alerts_c
                     DQN_NH
                     )
Results<- t(results)
colnames(Results)<- c("Deaths", "All hospitalizations", "Non-heat hospitalizations")
Results

## Note: for the subset function, must keep together county-summers
Zone<- subset_OPE(Pred_deaths, Pred_hosps, Pred_OH,
           split_vars = c("ZoneCold", "ZoneHot.Dry", "ZoneHot.Humid", "ZoneMarine",
                          "ZoneMixed.Dry", "ZoneMixed.Humid", "ZoneVery.Cold"))

SES<- subset_OPE(Pred_deaths, Pred_hosps, Pred_OH,
           split_vars = "l.Med.HH.Income", 
           breaks = quantile(DF$l.Med.HH.Income, probs=seq(0,1,length.out=11)))
SES$Med.HH.Income_decile<- seq(0,1,length.out=11)[-1]

ggplot(SES, aes(x=Med.HH.Income_decile, y= `All hospitalizations`, col = "blue")) + 
  geom_line() + 
  geom_line(aes(y = Deaths, col = "red")) + 
  geom_line(aes(y = `Non-heat hospitalizations`, col = "green")) +
  ylab("Outcome")

#### Not discounted:

Observed_NWS_nd<- get_OPE(R_deaths[,1], R_all_hosps[,1], R_other_hosps[,1], 
                          discount = FALSE)
Modeled_NWS_nd<- get_OPE(Pred_deaths, Pred_hosps, Pred_OH, discount = FALSE)
Mod2_NWS_nd<- get_OPE(Mod2$X0, Mod2$X1, Mod2$X2, discount = FALSE)
Mod2_NWS_nd_b<- get_OPE(Mod2_b$X0, Mod2_b$X1, Mod2_b$X2, discount = FALSE)
Mod2_NWS_nd_c<- get_OPE(Mod2_c$X0, Mod2_c$X1, Mod2_c$X2, discount = FALSE)
No_alerts_nd<- get_OPE(None$X0, None$X1, None$X2, discount = FALSE)
No_alerts_nd_b<- get_OPE(None_b$X0, None_b$X1, None_b$X2, discount = FALSE)
No_alerts_nd_c<- get_OPE(None_c$X0, None_c$X1, None_c$X2, discount = FALSE)

results_nd<- data.frame(Observed_NWS_nd, Modeled_NWS_nd, Mod2_NWS_nd,
                        Mod2_NWS_nd_b, Mod2_NWS_nd_c, 
                        No_alerts_nd, No_alerts_nd_b, No_alerts_nd_c)
Results_nd<- t(results_nd)
colnames(Results_nd)<- c("Deaths", "All hospitalizations", "Non-heat hospitalizations")
Results_nd

