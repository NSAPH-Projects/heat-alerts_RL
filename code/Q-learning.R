library(lubridate)
library(matrixStats)
library(dplyr)

#### Setup:

load("data/HARL_prelim_image.RData")
rm(list=setdiff(ls(), "summer"))

n_counties<- length(unique(summer$GEOID))
n_years<- 11
n_days<- 153

summer$dos<- rep(1:n_days, n_counties*n_years) # same idea as "time" variable

## Need to adjust alert_sum to account for 11 alerts from April (which I'm not including):
april_alerts<- summer[which(summer$month==5 & summer$day == 1 & summer$alert_sum > 0), c("GEOID", "year")]
summer[which(summer$year == 2009 & summer$GEOID %in% april_alerts$GEOID),"alert_sum"]<- summer[which(summer$year == 2009 & summer$GEOID %in% april_alerts$GEOID),"alert_sum"] - 1


summer$day<- day(summer$Date)
last_day<- summer[which(summer$month == 9 & summer$day == 30),]

summer$holiday<- factor(summer$holiday)
# summer<- as.data.frame(summer)

States.1<- summer[, c("HImaxF_PopW", "quant_HI_county", 
                    "HI_lag1", "quant_HI_yest_county",
                    "HI_lag2", "quant_HI_3d_county",
                    "year", "dos", 
                    "dow", "holiday",
                    "Med.HH.Income", "Pop_density", 
                    "alert_sum")]

States<- States.1[-seq(n_days, nrow(summer), n_days),]
States.1<- States.1[-seq(1, nrow(summer), n_days),]

Actions<- summer[-seq(n_days, nrow(summer), n_days),"alert"]

## Normalize state variables: (careful with non-numeric vars)
S<- States %>% mutate_if(is.numeric, scale)
S.1<- States.1 %>% mutate_if(is.numeric, scale)

# S[,"dow"]<- as.factor(S[,"dow"])
# S.1[,"dow"]<- as.factor(S.1[,"dow"])

#### Evaluate Q:

eval_Q<- function(S, Q_model){
  
  data<- data.frame(S, A = 0)
  Q0<- predict(Q_model, data)
  
  data<- data.frame(S, A = 1)
  Q1<- predict(Q_model, data)
  
  return(cbind(Q0, Q1))
}

#### Choose action:

choose_a<- function(Q_mat, Budget){
  
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

  ## Constrain the policy with the budget:
  cumsum_q<- as.vector(t(aggregate(argmax ~ ID,
                                   data.frame(ID, argmax), cumsum)[[2]]))

  over_budget<- which(cumsum_q > Budget)

  argmax[over_budget]<- 0
  max.Q[over_budget]<- Q_mat[over_budget, 1]
  
  return(cbind(argmax, max.Q))
}

#### Build targets:

target<- function(AMQ, R, gamma){
  
  # return(R + gamma*(1-ep_end)*AMQ[,2]) # is this right?
  # false_pos<- (AMQ[,1] == 1)&(S.1[,"quant_HI_county"] < (0.8-mean(States.1$quant_HI_county))/sd(States.1$quant_HI_county))
  # false_neg<- (AMQ[,1] == 0)&(S.1[,"quant_HI_county"] > 0.95)
  return(R + gamma*(1-ep_end)*AMQ[,2])
}

#### Final setup:

Budget<- rep(penult_day$alert_sum, each = (n_days - 1)) # alert budget

## Only look at data with one or more heat alerts?
positive<- which(Budget > 0)

set.seed(321)

A<- Actions[positive]
R<- (-1*(summer$N*100000/summer$Population))[-seq(n_days, nrow(summer), n_days)][positive]
ep_end<- rep(c(rep(0,n_days-2),1),length(R)/(n_days-1))[positive] # end-of-episode indicator

ID<- rep(1:(n_counties*n_years), each = (n_days-1))[positive]

Target<- R

gamma<- 0.999

Budget<- Budget[positive] # updated alert budget

#### Q-Learning loop:
iter<- 1

# Q_model<- lm(Target ~ A*., data = data.frame(Target, S, A))
Q_model<- lm(Target ~ A*., data = data.frame(Target, S[positive,], A))
old_coefs<- rnorm(length(coef(Q_model)))

while(sum((coef(Q_model)-old_coefs)^2) > 0.01 & iter < 6){
  
  # Q_mat<- eval_Q(S.1, Q_model) # is this right?
  Q_mat<- eval_Q(S.1[positive,], Q_model)
  
  AMQ<- choose_a(Q_mat, Budget)
  
  Target<- target(AMQ, R, gamma)
  
  old_coefs<- coef(Q_model)
  # Q_model<- lm(Target ~ A*., data = data.frame(Target, S, A))
  Q_model<- lm(Target ~ A*., data = data.frame(Target, S[positive,], A))
  
  print(iter)
  print(Q_model$coefficients)
  iter<- iter+1
}

Q_mat.S<- eval_Q(S, Q_model)
policy<- choose_a(Q_mat.S, Budget)[,1]

#### Inspect:

hist(S[which(policy==1),2])
# hist(S.1[positive,1][which(policy1==1)])

hist(S[which(policy==1 & A==0),"quant_HI_county"])
hist(S[which(policy==0 & A==1),"quant_HI_county"])

hist(S[which(policy==1 & A==0),"quant_HI_yest_county"])
hist(S[which(policy==0 & A==1),"quant_HI_yest_county"])

hist(S[which(policy==1 & A==0),"quant_HI_3d_county"])
hist(S[which(policy==0 & A==1),"quant_HI_3d_county"])



