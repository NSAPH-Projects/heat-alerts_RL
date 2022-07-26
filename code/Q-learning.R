library(lubridate)
library(matrixStats)
library(dplyr)

#### Setup:

summer<- readRDS("data/Final_data_for_HARL.rds")

n_counties<- length(unique(summer$GEOID))
n_years<- 11
n_days<- 153

summer$dos<- rep(1:n_days, n_counties*n_years) # same idea as "time" variable

## Need to adjust alert_sum to account for 11 alerts from April (which I'm not including):
summer$day<- day(summer$Date)
april_alerts<- summer[which(summer$month==5 & summer$day == 1 & summer$alert_sum > 0), c("GEOID", "year")]
summer[which(summer$year == 2009 & summer$GEOID %in% april_alerts$GEOID),"alert_sum"]<- summer[which(summer$year == 2009 & summer$GEOID %in% april_alerts$GEOID),"alert_sum"] - 1

last_day<- summer[which(summer$month == 9 & summer$day == 30),]

summer$holiday<- factor(summer$holiday)
# summer<- as.data.frame(summer)

#### Evaluate Q:

eval_Q<- function(S, Q_model){
  
  data<- data.frame(S, A = 0)
  Q0<- predict(Q_model, data)
  
  data<- data.frame(S, A = 1)
  Q1<- predict(Q_model, data)
  
  return(cbind(Q0, Q1))
}

#### Choose action:

choose_a<- function(Q_mat){
  
  # budget<- last_day$alert_sum
  # max_b<- max(budget)
  # 
  # # diffs<- apply(Q_mat, MARGIN=1, function(x) x[2]-x[1])
  # diffs<- apply(Q_mat, MARGIN=1, function(x) (x[2]-x[1])/abs(x[1]))
  # 
  # Alerts <- data.frame(ID, diffs, Pos=1:nrow(Q_mat)) %>%
  #   arrange(desc(diffs)) %>%
  #   group_by(ID) %>%
  #   slice(1:max_b)
  # 
  # Keep<- c()
  # 
  # for(i in 1:max(ID)){
  #   these<- 1:max_b + (i-1)*max_b
  #   if(budget[i] > 0){
  #     Keep<- append(Keep, these[1:budget[i]])
  #   }
  # }
  # 
  # max.Q<- Q_mat[,1]
  # max.Q[Keep]<- Q_mat[Keep,2]
  # argmax<- rep(0, nrow(Q_mat))
  # argmax[Keep]<- 1
   
  max.Q<- rowMaxs(Q_mat)
  argmax<- max.col(Q_mat, ties.method = "first") - 1
  
  # ## Constrain the policy with the budget:
  # cumsum_q<- as.vector(t(aggregate(argmax ~ ID,
  #                                  data.frame(ID, argmax), cumsum)[[2]]))
  # 
  # over_budget<- which(cumsum_q > Budget)

  argmax[over_pos]<- 0
  max.Q[over_pos]<- Q_mat[over_pos, 1]
  
  return(cbind(argmax, max.Q))
}

#### Final setup:

budget<- last_day$alert_sum
Budget<- rep(last_day$alert_sum, each = (n_days - 1)) # alert budget

## Only look at data with one or more heat alerts?
positive<- which(Budget > 0)
# Budget<- Budget[positive] # updated alert budget

set.seed(321)

gamma<- 0.999
A<- summer[-seq(n_days, nrow(summer), n_days),"alert"]#[positive]
R<- (-1*(summer$N*100000/summer$Pop.65))[-seq(n_days, nrow(summer), n_days)]#[positive]
ep_end<- rep(c(rep(0,n_days-2),1),length(R)/(n_days-1))#[positive] # end-of-episode indicator

Target<- R

## Decide what to include in the states...

# States.1<- summer[, c("HImaxF_PopW", "HI_lag1",
#                       "HI_3days", "HI_fwd_avg",
#                       "BA_zone", "Pop_density", "Med.HH.Income",
#                       "year", "dos", "holiday", # "Holiday", "dow",
#                       "alert_sum")]
# 
# States.1<- summer[, c("quant_HI_county", "quant_HI_yest_county",
#                       "quant_HI_3d_county", "quant_HI_fwd_avg_county",
#                       "BA_zone", "Pop_density", "Med.HH.Income",
#                       "year", "dos", "holiday", # "Holiday", "dow",
#                       "alert_sum")]

States.1<- summer[, c("HImaxF_PopW", "quant_HI_county", "quant_HI_yest_county",
                      "quant_HI_3d_county", "quant_HI_fwd_avg_county",
                      "BA_zone", "Pop_density", "Med.HH.Income",
                      "year", "dos", "holiday", # "Holiday", "dow",
                      "alert_sum")]

States<- States.1[-seq(n_days, nrow(summer), n_days),]
States.1<- States.1[-seq(1, nrow(summer), n_days),]

## Normalize state variables: (careful with non-numeric vars)
S<- States %>% mutate_if(is.numeric, scale)
S.1<- States.1 %>% mutate_if(is.numeric, scale)

# S[,"dow"]<- as.factor(S[,"dow"])
# S.1[,"dow"]<- as.factor(S.1[,"dow"])

ID<- rep(1:(n_counties*n_years), each = (n_days-1))#[positive]

# over_budget<- rep(0, nrow(S))
# 
# for(i in 1:max(ID)){
#   pos<- which(ID == i)
#   b<- which(States[pos,"alert_sum"] == budget[i])[1]
#   over_budget[pos][(b+1):(n_days-1)]<- 1
# }
# 
# saveRDS(over_budget, "data/Over_budget.rds")

over_budget<- readRDS("data/Over_budget.rds")
over_pos<- which(over_budget==1)

#### Q-Learning loop:
iter<- 1

Q_model<- lm(Target ~ A*., data = data.frame(Target, S, A))
# Q_model<- lm(Target ~ A*., data = data.frame(Target, S[positive,], A))
old_coefs<- rnorm(length(coef(Q_model)))

while(sum((coef(Q_model)-old_coefs)^2) > 0.1 & iter < 20){
  
  Q_mat<- eval_Q(S.1, Q_model)
  # Q_mat<- eval_Q(S.1[positive,], Q_model)
  
  AMQ<- choose_a(Q_mat)
  
  Target<- R + gamma*(1-ep_end)*AMQ[,2] 
  
  old_coefs<- coef(Q_model)
  Q_model<- lm(Target ~ A*., data = data.frame(Target, S, A))
  # Q_model<- lm(Target ~ A*., data = data.frame(Target, S[positive,], A))
  
  print(iter)
  print(sum((coef(Q_model)-old_coefs)^2))
  print(Q_model$coefficients)
  iter<- iter+1
}

saveRDS(Q_model, "Lm_7-26.rds")
saveRDS(Target, "Targets_7-26.rds")

## See restricted results:
Q_mat.S<- eval_Q(S[positive,], Q_model)
policy<- choose_a(Q_mat.S)[,1]

## See on non-restricted results:
A<- Actions
Budget<- rep(last_day$alert_sum, each = (n_days - 1))
ID<- rep(1:(n_counties*n_years), each = (n_days-1))
Q_mat<- eval_Q(S.1, Q_model)
policy<- choose_a(Q_mat)[,1]

#### Inspect:

# hist(S[positive,2][which(policy==1)])

hist(S[which(policy==1),2])

hist(S[which(policy==1 & A==0),"quant_HI_county"])
hist(S[which(policy==0 & A==1),"quant_HI_county"])

hist(S[which(policy==1 & A==0),"quant_HI_yest_county"])
hist(S[which(policy==0 & A==1),"quant_HI_yest_county"])

hist(S[which(policy==1 & A==0),"quant_HI_3d_county"])
hist(S[which(policy==0 & A==1),"quant_HI_3d_county"])

hist(S[which(policy==1 & A==0),"quant_HI_fwd_avg_county"])
hist(S[which(policy==0 & A==1),"quant_HI_fwd_avg_county"])

hist(S[which(policy==1 & A==0),"dos"])
hist(S[which(policy==0 & A==1),"dos"])

S[which(policy==1 & A==0),]



