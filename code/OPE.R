
library(dplyr)
library(pracma)

# setwd("/n/dominici_nsaph_l3/projects/heat-alerts_mortality_RL")

### Get "propensity scores":

pi_b1<- function(a_model, data, ML = TRUE){
  Data<- data.frame(scale(data[,vars<- c("HImaxF_PopW", "quant_HI_county", 
                                            "quant_HI_yest_county",
                                            "quant_HI_3d_county", 
                                            "quant_HI_fwd_avg_county",
                                            "Pop_density", "Med.HH.Income",
                                            "year", "dos",
                                            "alert_sum")]), 
                    alert = factor(data$alert),
                    dow = factor(data$dow), 
                    holiday = factor(data$holiday),
                    Zone = factor(data$BA_zone))
  if(ML == FALSE){
    pb<- predict(a_model, Data)
    return(sigmoid(pb))
  }else{
    pb<- predict(a_model, Data, type = "prob")[,2] # grabs probs for a=1
    return(pb)
  }
}

m<- 2.504325 # From training set
s<- 5.194564

new_policy<- function(Q_model, data, Budget){
  
  # df0<- model.matrix(Q_model)
  # df0[,"A"]<- 0
  # df0[,20:36]<- 0
  # 
  # df1<- model.matrix(Q_model)
  # df1[,"A"]<- 1
  # df1[,20:36]<- df1[,3:19]
  
  df0<- S_full_0
  df1<- S_full_1
  
  new_alerts<- rep(0, nrow(df0))
  policy<- rep(0, nrow(df0))
  
  ID<- rep(1:(nrow(df0)/(n_days-1)), each = (n_days-1))#[positive]
    
  for(i in 1:max(ID)){
    pos<- which(ID == i)
    
    d<- 1
    while(d < 153 & new_alerts[pos[d]] < Budget[pos[d]]){
      new_scaled<- (new_alerts[pos[d]] - m)/s
      v0<- df0[pos[d],]
      v0["alert_sum"]<- new_scaled
      # q0<- coef(Q_model) %*% v0
      q0<- Q_c %*% v0
      v1<- df1[pos[d],]
      v1["alert_sum"]<- new_scaled
      v1["A:alert_sum"]<- new_scaled
      # q1<- coef(Q_model) %*% v1
      q1<- Q_c %*% v1
      if(q1 > q0){
        policy[pos[d]]<- 1
        new_alerts[pos[d:(n_days-1)]]<- new_alerts[pos[d:(n_days-1)]] + 1
      }
      d<- d+1
    }
  }
  
  return(policy)
}

set.seed(321)

random_policy<- function(data, budget){ 
  
  policy<- rep(0, nrow(data))
  ID<- rep(1:(nrow(data)/(n_days-1)), each = (n_days-1))
  
  for(i in 1:max(ID)){
    pos<- which(ID == i)
    
    policy[pos]<- budget[i]/(n_days-1)
    
    # if(budget[i] > 0){
    #   policy[sample(pos, budget[i], replace=FALSE)]<- 1
    # }
  }
  
  return(policy)
}

### Perform off-policy evaluation:

OPE<- function(a_model, Q_model, Data, R, discount, Budget){
  
  A<- Data$alert
  
  pb1<- pi_b1(a_model, Data)
  pb<- pb1
  pb[which(A == 0)]<- 1 - pb1[which(A == 0)]
  
  # Regularize:
  pb[which(pb > 0.99)]<- 0.99 
  pb[which(pb < 0.01)]<- 0.01
  
  # pg1<- random_policy(Data, budget[which(budget > 0)])
  # pg<- pg1
  # pg[which(A == 0)]<- 1 - pg1[which(A == 0)]
  pol<- new_policy(Q_model, Data, Budget)
  pg<- rep(0,length(pb))
  pg[which(A == pol)]<- 1
  
  ## Following notation from Levine et al. 2020:
  n<- nrow(Data)/(n_days-1)
  H<- n_days-1
  w<- rep(0, n*H)
  eps<- 0.01
  for(i in 1:n){
    for(t in 1:H){
      ep_start<- (i-1)*H
      if(0 %in% pg[(ep_start+1):(ep_start+t)]){
        w[ep_start+t]<- 0
      }else{
        w[ep_start+t]<- exp(sum(log(pg[ep_start:(ep_start+t)]))) / 
          (eps + exp(sum(log(pb[ep_start:(ep_start+t)]))))
        # w[ep_start+t]<- exp( sum(log(pg[ep_start:(ep_start+t)]))
        #   - sum(log(pb[ep_start:(ep_start+t)])) )
        # w[ep_start+t]<- 1/prod(pb[ep_start:(ep_start+t)])
      }
    }
  }
  
  # discount_vec<- rep(cumprod(rep(discount, H))/discount, n)
  discount_vec<- 1
  
  return((1/n)*sum(w*discount_vec*R))
}

### Run this ^^^

load("data/Train-Valid-Test.RData")

n_counties<- length(unique(Train$GEOID))
n_years<- 11
n_days<- 153

a_model<- readRDS("Aug_results/a_RF_9-5_50pct.rds")
# OR:
a_model<- readRDS("Aug_results/a_glm_8-30.rds")

Q_model<- readRDS("Aug_results/Lm_8-2_full.rds")
# OR: 
Coefs<- readRDS("Aug_results/Q-coefficients_9-7.rds")
Q_c<- Coefs[nrow(Coefs),]

data<- Train

Data<- data[-seq(n_days, nrow(data), n_days),]
R<- -1*(Train$N*100000/Train$Pop.65)[-seq(n_days, nrow(data), n_days)]
# discount<- 0.999
discount<- 1

budget<- data[which(data$dos == 153), "alert_sum"]
Budget<- rep(budget, each = (n_days - 1))
nonzero<- which(Budget > 0)

## Make states the same as in Q-learning:

States<- Train[, c("HImaxF_PopW", "quant_HI_county", "quant_HI_yest_county",
                     "quant_HI_3d_county", "quant_HI_fwd_avg_county",
                     "BA_zone", "Pop_density", "Med.HH.Income",
                     "year", "dos", "holiday", # "Holiday", 
                     "dow", "alert_sum")] # same variables as in model for alerts

States<- States[-seq(n_days, nrow(Train), n_days),]

## Scale and one-hot encode:
S<- States %>% mutate_if(is.numeric, scale)
S_full_0<- model.matrix(~ A*., data.frame(A=0,S))
S_full_1<- model.matrix(~ A*., data.frame(A=1,S))

## Only look at counties with at least one heat alert?

OPE(a_model, Q_model, Data[nonzero,], R[nonzero], discount, Budget[nonzero])


#######################################################################3

## Go back and get the mean, sd from the training Q-values:

q_tols<- c(0.0025, 0.002, 0.00175, 0.0015, 0.001, 0.0005)
Q_scales<- c()

for(q in q_tols){
  results<- readRDS(paste0("new_results/Q-scale_policy_q-tol_", q, ".rds"))
  Q_scales<- append(Q_scales, results[[1]])
}

## Getting the weights from per-decision importance sampling:

pi_b<- function(S_test, A_test){
  
  # S_test<- data.frame(round(S_test), M = rep(1:(N/n), each = n))
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
  Q<- cbind(Q0, Q1, Q1-Q0)/Q_scales[which(q_tols == q_tol)]
  policy<- rep(0, N)
  policy[which(Q[,3] > 0 & Q[,3]/abs(Q[,1]) > q_tol)]<- 1
  
  ## Get the probabilities:
  
  # S_test<- data.frame(round(S_test), M = rep(1:(N/n), each = n))
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

#### Test for y = 100:

OPE(S_test, A_test, w = c(4.9592316, 0.9695083, 127.1199305, 
                          -467.6479547, -644.3341325, 5.4035366, 
                          0.5878685, 129.6059521, -466.9971469,
                          -633.4854926),
    y = 100, q_tol = 0.0025) # -91.28094

OPE(S_test, A_test, w = c(),
    y = 100, q_tol = 0.002) #

OPE(S_test, A_test, w = c(),
    y = 100, q_tol = 0.00175) #

OPE(S_test, A_test, w = c(),
    y = 100, q_tol = 0.0015) #

OPE(S_test, A_test, w = c(),
    y = 100, q_tol = 0.001) #

OPE(S_test, A_test, w = c(),
    y = 100, q_tol = 0.0005) #

1     4.069279     3.776929
2    -6.060571    -1.927672
3   586.094591   585.587001
4 -2538.162927 -2535.023654
5     4.527331    28.002250


############################### First round:

#### Test for y = 100: rounding with 1 decimal, then with 0, then with 2

OPE(S_test, A_test, w = c(4.9002988, 0.9561502, 122.8304463, -467.6929355, 
                          -626.8763427, 5.3314790, 0.5666687, 125.3147654, 
                          -467.0453364, -616.1291645),
    y = 100, q_tol = 0.0025) # -91.86163, -117.8713, -90.16397

OPE(S_test, A_test, w = c(3.227026, -1.057144, 157.743203, -397.232099,
                          -915.638060, 2.3019162, -0.1524963, 158.0077266,
                          -396.6269728, -903.3962598),
    y = 100, q_tol = 0.002) # -87.37957, -89.28409, -85.68192

OPE(S_test, A_test, w = c(2.0424366, -2.5993480, 241.4489724, -630.5807619,
                          -1430.6157089, 0.6076141, -0.6964970, 241.2610769,
                          -629.8293962, -1415.5634076),
    y = 100, q_tol = 0.00175) # -87.43654, -89.12207, -85.7404

OPE(S_test, A_test, w = c(1.0884783, -4.2786934, 342.4826609, -827.0148787,
                          -1696.9958081, -0.7456498, -1.4345054, 342.1957501,
                          -826.1844939, -1679.2944142),
    y = 100, q_tol = 0.0015) # -86.28342, -84.8756, -84.58577

OPE(S_test, A_test, w = c(0.5592941, -5.3593660, 477.8782490, -1710.4075695,
                          -635.3351123, -1.0490838, -1.7841658, 477.4670577,
                          -1708.6196682,  -615.0188748),
    y = 100, q_tol = 0.001) # -89.0691, -89.86042, -87.37144

OPE(S_test, A_test, w = c(4.357196, -5.755611, 603.492995, -2671.301377, 251.558483,
                          4.136001, -1.775227, 602.964898, -2667.975966, 274.604229),
    y = 100, q_tol = 0.0005) # -90.25862, -90.33333, -88.56096


#### Test for y = 50? 

OPE(S_test, A_test, w = c(),
    y = 50, q_tol = 0.002)



