
library(dplyr)

setwd("/n/dominici_nsaph_l3/projects/heat-alerts_mortality_RL")

load("data/HARL_prelim_image.RData")

## Getting the weights from per-decision importance sampling:

pi_b<- function(S_test, A_test){
  
  S_test<- data.frame(round(S_test,1), M = rep(1:(N/n), each = n))
  df<- aggregate(P_one ~ ., data.frame(S_test, P_one = A_test), mean)
  DF<- inner_join(data.frame(S_test, A_test), df)
  
  probs<- rep(0, N)
  probs[which(A_test == 0)]<- 1 - DF$P_one[which(A_test == 0)]
  probs[which(A_test == 1)]<- DF$P_one[which(A_test == 1)]
  return(probs)
}

pi_g<- function(S_test, A_test, w, q_tol){
  
  ## Get the actions under the policy:
  s<- ncol(S_test)
  Q0<- S_test%*%w[1:s]
  Q1<- S_test%*%w[(s + 1):(2*s)]
  q_scale<- sd(c(Q0, Q1))
  Q<- cbind(Q0, Q1, Q1-Q0)/q_scale
  policy<- rep(0, N)
  policy[which(Q[,3] > 0 & Q[,3]/abs(Q[,1]) > q_tol)]<- 1
  
  ## Get the probabilities:
  
  S_test<- data.frame(round(S_test,1), M = rep(1:(N/n), each = n))
  df<- aggregate(P_one ~ ., data.frame(S_test, P_one = policy), mean)
  DF<- inner_join(data.frame(S_test, A_test), df)
  
  probs<- rep(0, N)
  probs[which(A_test == 0)]<- 1 - DF$P_one[which(A_test == 0)]
  probs[which(A_test == 1)]<- DF$P_one[which(A_test == 1)]
  return(probs)
}

## Performing off-policy evaluation:

OPE<- function(S_test, A_test, w, y, q_tol){
  R_test<- Rewards[test] - y*failed_alerts
  
  pi_ratio<- pi_g(S_test, A_test, w, q_tol) / Pi_b
  
  weights<- as.vector(sapply(seq(1, N/n), function(i){cumprod(pi_ratio[i:(i+n-1)])}))
  
  return((1/M)*sum(weights*discount_vec*R_test))
}

## Set up test set:
N<- length(test)
n<- 152 # number of days in each summer (episode)
M<- N/n # number of episodes

S_test<- S[test,]
A_test<- Actions[test]
discount_vec<- rep(cumprod(rep(discount, n))/discount, M)
failed_alerts<- (A_test == 1) & (Not_Hot[test] == 1)
Rewards<- (-1*(summer$N*100000/summer$Population))[-seq(153, nrow(summer), 153)]

Pi_b<- pi_b(S_test, A_test)


#### Test for y = 10: 

OPE(S_test, A_test, w = c(),
    y = 10, q_tol = 0.00175)

OPE(S_test, A_test, w = c(),
    y = 10, q_tol = 0.001)

#### Test for y = 100: 

OPE(S_test, A_test, w = c(4.9002988, 0.9561502, 122.8304463, -467.6929355, 
                          -626.8763427, 5.3314790, 0.5666687, 125.3147654, 
                          -467.0453364, -616.1291645),
    y = 100, q_tol = 0.0025) # -91.86163

OPE(S_test, A_test, w = c(3.227026, -1.057144, 157.743203, -397.232099,
                          -915.638060, 2.3019162, -0.1524963, 158.0077266,
                          -396.6269728, -903.3962598),
    y = 100, q_tol = 0.002) # -87.37957

OPE(S_test, A_test, w = c(2.0424366, -2.5993480, 241.4489724, -630.5807619,
                          -1430.6157089, 0.6076141, -0.6964970, 241.2610769,
                          -629.8293962, -1415.5634076),
    y = 100, q_tol = 0.00175) # -87.43654

OPE(S_test, A_test, w = c(1.0884783, -4.2786934, 342.4826609, -827.0148787,
                          -1696.9958081, -0.7456498, -1.4345054, 342.1957501,
                          -826.1844939, -1679.2944142),
    y = 100, q_tol = 0.0015) # -86.28342

OPE(S_test, A_test, w = c(0.5592941, -5.3593660, 477.8782490, -1710.4075695,
                          -635.3351123, -1.0490838, -1.7841658, 477.4670577,
                          -1708.6196682,  -615.0188748),
    y = 100, q_tol = 0.001) # -89.0691

OPE(S_test, A_test, w = c(4.357196, -5.755611, 603.492995, -2671.301377, 251.558483,
                          4.136001, -1.775227, 602.964898, -2667.975966, 274.604229),
    y = 100, q_tol = 0.0005) # -90.25862

