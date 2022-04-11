setwd("/n/home_fasse/econsidine/")
library(lubridate)
library(Metrics)

## Setup:
setwd("/n/dominici_nsaph_l3/projects/heat-alerts_mortality_RL")

## Read in the data:

data<- readRDS("data/Final_data_for_HARL.rds")
data$Date<- as.Date(data$Date)
data$month<- month(data$Date)
summer<- data[which(data$month %in% 5:9),] # excluding April and October
summer$Holiday<- as.numeric((summer$holiday == 1) |
                              (summer$dow %in% c("Saturday", "Sunday")))

my_quant<- function(df, region_var, split_var #, probs)
  ){
  regions<- unique(df[, region_var]) # state or county?
  
  q<- rep(0,dim(df)[1])
  
  for(r in regions){
    pos<- which(df[, region_var] == r)
    # r_quants<- quantile(df[pos, split_var], probs)
    # q[pos]<- as.numeric(cut(df[pos, split_var], r_quants))
    percentile<- ecdf(df[pos, split_var])
    q[pos]<- percentile(df[pos, split_var])
  }
  return(q)
}

summer$quant_HI<- my_quant(summer, "state", "HImaxF_PopW")
summer$quant_HI_yest<- my_quant(summer, "state", "HI_lag1")
summer$quant_HI_3d<- my_quant(summer, "state", "HI_3days")

summer$quant_HI_county<- my_quant(summer, "GEOID", "HImaxF_PopW")
summer$quant_HI_yest_county<- my_quant(summer, "GEOID", "HI_lag1")
summer$quant_HI_3d_county<- my_quant(summer, "GEOID", "HI_3days")

summer$failed_alert_abs<- as.numeric(summer$alert & summer$HImaxF_PopW < 90) # absolute 
summer$failed_alert_rel<- as.numeric(summer$alert & summer$quant_HI < 0.8) # relative
summer$failed_alert_rel_county<- as.numeric(summer$alert & summer$quant_HI_county < 0.8) # relative


## Define states, actions, rewards

States<- summer[, c( "HI_lag1", "HI_3days", 
                    # "quant_HI_yest_county", "quant_HI_3d_county",
               "Pop_density", # "Holiday,
               "Med.HH.Income", "alerts_2wks")]
States.1<- summer[, c( "HI_lag1", "HI_3days", 
                      # "quant_HI_yest_county", "quant_HI_3d_county",
                 "Pop_density", # "Holiday,
                 "Med.HH.Income", "alerts_2wks")]
States<- States[-seq(153, nrow(summer), 153),] # there are 153 days each summer
States.1<- States.1[-seq(1, nrow(summer), 153),]

Actions<- summer[-seq(1, nrow(summer), 153),"alert"]

Rewards<- (-1*(summer$N*100000/summer$Population + 
           1*summer$failed_alert_rel))[-seq(1, nrow(summer), 153)]# weight differently?

## Normalize state variables:
S<- scale(States)
S.1<- scale(States.1)

# For now, let's say the discount is:
discount<- 0.999

## Get indices of train (2006-2014) and test (2015-2016) years:
train<- which(summer$Date[-seq(1, nrow(summer), 153)] < "2015-01-01")
test<- which(summer$Date[-seq(1, nrow(summer), 153)] >= "2015-01-01")

save.image("data/HARL_prelim_image.RData")

##################################################

load("data/HARL_prelim_image.RData")

## LSTDQ:

lstdq<- function(S.1, discount, Phi, b, policy){
  # Phi is the matrix of covariates / basis functions
  # b is Phi^T %*% R 
  # policy is a vector of length |S|
  
  P.Pi.Phi<- matrix(0, ncol = ncol(Phi), nrow = nrow(Phi))
  colnames(P.Pi.Phi)<- colnames(Phi)
  P.Pi.Phi[which(policy == 0),1:ncol(S.1)]<- S.1[which(policy == 0),]
  P.Pi.Phi[which(policy == 1),(ncol(S.1) + 1):(2*ncol(S.1))]<- S.1[which(policy == 1),]
  
  A_mat<- t(Phi)%*%(Phi - discount*P.Pi.Phi)
  
  # Invert A_mat if it's non-singular:
  test_svd<- svd(A_mat)
  if(sum(test_svd$d < 1)){
    break
  }else{
    return(solve(A_mat)%*%b) # w
  }
}

lspi<- function(S, A, R, S.1, discount, tol){
  # A and R are vectors with the actions and rewards respectively 
  # S and S.1 are matrices with one "state" in each row
  # discount is a scalar between 0 and 1
  # tol is convergence tolerance
  
  Phi<- matrix(0, ncol = 2*ncol(S), nrow = nrow(S))
  colnames(Phi)<- rep(colnames(S),2)
  Phi[which(A == 0),1:ncol(S)]<- S[which(A == 0),]
  Phi[which(A == 1),(ncol(S) + 1):(2*ncol(S))]<- S[which(A == 1),]
  
  b<- t(Phi) %*% R
  
  # Initialize policy:
  policy<- rep(c(0,1), nrow(Phi)/2)
  
  w_old<- rep(1, ncol(Phi))
  
  # Apply LSTDQ:
  w_new<- lstdq(S.1, discount, Phi, b, policy)
  
  i<- 1
  while(rmse(w_old, w_new) > tol){
  # while(i < 4){
    # print(data.frame(No_alert=w_new[1:ncol(S)], Alert = w_new[(ncol(S) + 1):(2*ncol(S))]))
    
    # Update the policy:
    Q0<- S.1%*%w_new[1:ncol(S.1)]
    Q1<- S.1%*%w_new[(ncol(S) + 1):(2*ncol(S))]
    Q<- cbind(Q0, Q1)
    policy<- max.col(Q) - 1
    
    # Re-run LSTDQ:
    w_old<- w_new
    w_new<- lstdq(S.1, discount, Phi, b, policy)
    i<- i+1
  }
  
  return(list(w_new, policy, i))
}

##### Getting preliminary results:

for(i in c(1, 10, 100, 1000, 10000)){
  Rewards<- (-1*(summer$N*100000/summer$Population + 
                   i*summer$failed_alert_rel_county))[-seq(1, nrow(summer), 153)]# weight differently?
  
  results<- lspi(S[train,], Actions[train], Rewards[train], S.1[train,], 
                 discount, tol = 0.001)
  
  w<- results[[1]]
  policy<- results[[2]]
  iter<- results[[3]]
  # iter
  
  ## Inspect:
  # sum(policy)
  # Results<- data.frame(Policy = policy, States[train,])
  
  W<- data.frame(No_alert=w[1:ncol(S)], Alert = w[(ncol(S) + 1):(2*ncol(S))])
  row.names(W)<- row.names(w)[1:ncol(S)]
  
  print(i)
  print(W)
  print(sum(policy))
}

