
setwd("/n/dominici_nsaph_l3/projects/heat-alerts_mortality_RL")

load("data/HARL_prelim_image.RData")

#### Off-Policy Evaluation:

## Importance sampling:



## Doubly-robust estimator:



#### Test: 

## Results from LSPI with i = 100, Q = 0.0025:

w<- c(1.8691083, -0.3793147, 248.2657825, -235.3186229, -783.4913561,
      1.14443322, 0.06340306, 248.08788236, -234.84976634, -772.86677101)

Q0<- S[test,]%*%w[1:ncol(S)]
Q1<- S[test,]%*%w[(ncol(S) + 1):(2*ncol(S))]
q_scale<- sd(c(Q0, Q1))
Q<- cbind(Q0, Q1, Q1-Q0)/q_scale
policy<- rep(0, length(test))
policy[which(Q[,3] > 0 & Q[,3]/abs(Q[,1]) > 0.0025)]<- 1

failed_alerts<- policy == 1 & Not_Hot[test] == 1

R<- (-1*(summer$N*100000/summer$Population))[-seq(153, nrow(summer), 153)][test] - 
  100*failed_alerts


