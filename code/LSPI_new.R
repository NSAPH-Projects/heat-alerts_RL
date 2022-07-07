library(lubridate)
library(matrixStats)

load("data/HARL_prelim_image.RData")
# rm(list=setdiff(ls(), "summer"))

n_counties<- length(unique(summer$GEOID))
n_years<- 11
n_days<- 153

summer$dos<- rep(1:n_days, n_counties*n_years) # same idea as "time" variable

summer$day<- day(summer$Date)
last_day<- summer[which(summer$month == 9 & summer$day == 30),]

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
  
  ep_end<- rep(c(rep(0,n_days-2),1),length(R)/(n_days-1)) # end-of-episode indicator
  
  return(R + gamma*(1-ep_end)*AMQ[,2]) # fix this??
  # false_pos<- (AMQ[,1] == 1)&(S.1[,1] < 0.8)
  # false_neg<- (AMQ[,1] == 0)&(S.1[,1] > 0.95)
  # return(R - false_pos + gamma*(1)*AMQ[,2])
}

#### Final setup:

Budget<- rep(last_day$alert_sum, each = (n_days - 1)) # alert budget

## Only look at data with one or more heat alerts?
positive<- which(Budget > 0)

set.seed(321)

A<- Actions#[positive]
R<- (-1*(summer$N*100000/summer$Population))[-seq(153, nrow(summer), 153)]#[positive]

ID<- rep(1:(n_counties*n_years), each = (n_days-1))#[positive]

policy0<- rbinom(length(A), 1, 0.1)
policy1<- A

Target<- R

gamma<- 0.999

#Budget<- Budget[positive] # updated alert budget

#### Q-Learning loop:
iter<- 1

while(abs(sum(policy1 - policy0)) > 0 & iter < 6){
  
  Q_model<- lm(Target ~ S*A, data = data.frame(Target, S, A))
  # Q_model<- lm(Target ~ S[positive,]*A, data = data.frame(Target, S[positive,], A))
  
  Q_mat<- eval_Q(S.1, Q_model) # is this right?
  # Q_mat<- eval_Q(S.1[positive,], Q_model)
  
  AMQ<- choose_a(Q_mat, Budget)
  
  policy0<- policy1
  policy1<- AMQ[,1]
  
  Target<- target(AMQ, R, gamma)
  
  print(iter)
  iter<- iter+1
}


#### Inspect:

hist(S.1[which(policy1==1),1])
# hist(S.1[positive,1][which(policy1==1)])

hist(S.1[which(policy1==1 & A==0),1])
hist(S.1[which(policy1==0 & A==1),1])

hist(S.1[which(policy1==1 & A==0),2])
hist(S.1[which(policy1==0 & A==1),2])

hist(S.1[which(policy1==1 & A==0),5])
hist(S.1[which(policy1==0 & A==1),5])


