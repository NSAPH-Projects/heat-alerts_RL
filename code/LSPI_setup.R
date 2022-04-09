library(lubridate)


## Setup:
setwd("/n/dominici_nsaph_l3/projects/heat-alerts_mortality_RL")

## Read in the data:

data<- readRDS("data/Data_for_HARL.rds")
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

summer$failed_alert_abs<- as.numeric(summer$alert & summer$HImaxF_PopW < 90) # absolute 
summer$failed_alert_rel<- as.numeric(summer$alert & summer$quant_HI < 0.8) # relative

## Define states, actions, rewards

S<- summer[, c("Med.HH.Income", "Pop_density",
                             "Holiday", "HI_lag1" # or "quant_HI_yest"
                             # "alerts_2wks", "HI_3days"
                             )]
S.1<- summer[, c("Med.HH.Income", "Pop_density",
                            "Holiday", "HI_lag1" # or "quant_HI_yest"
                            # "alerts_2wks", "HI_3days"
                            )]
S<- S[-seq(153, nrow(summer), 153),] # there are 153 days each summer
S.1<- S.1[-seq(1, nrow(summer), 153),]

A<- summer[-seq(1, nrow(summer), 153),"alert"]

R<- (-1*(summer$N*100000/summer$Population + 
           summer$failed_alert_abs))[-seq(1, nrow(summer), 153)]# weight differently?

## Normalize state variables:
norm_S<- scale(S)
norm_S.1<- scale(S.1)

# For now, let's say the discount is:
discount<- 0.999

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
  
  # Apply LSTDQ:
  w<- lstdq(S.1, discount, Phi, b, policy)
  
  ## Implement while loop with tol...
  
  # Update the policy:
  Q0<- S%*%w[1:ncol(S)]
  Q1<- S%*%w[(ncol(S) + 1):(2*ncol(S))]
  Q<- cbind(Q0, Q1)
  policy<- max.col(Q) - 1
  
  return(list(w, policy))
}


lspi(S, A, R, S.1, Phi, discount, tol)

