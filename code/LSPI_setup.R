
## Setup:
setwd("/n/dominici_nsaph_l3/projects/heat-alerts_mortality_RL")

## Read in the data:

data<- readRDS("data/Merged_with_lags.rds")

## Create some additional variables:

# Explore alerts and quantiles:
states<- unique(data$state)
alert_stats<- matrix(0, ncol = 4, nrow = length(states))

for(s in states){
  i<- which(states == s)
  df<- data[which(data$state == s),]
  alert_stats[i,1]<- min(df[which(df$alert == 1), "HImaxF_PopW"])
  alert_stats[i,2]<- max(df[which(df$alert == 0), "HImaxF_PopW"])
}

# aggregate(HImaxF_PopW ~ state, data, function(x) quantile(x, probs = c(0.9, 0.97, 1)))

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

data$quant_HI<- my_quant(data, "state", "HImaxF_PopW")

data$failed_alert<- as.numeric(data$alert & data$HImaxF_PopW < 90) # adjust threshold?

## Define states, actions, rewards

A<- data[,"alert"]

R<- -1*(data$N*100000/data$CENSUS2010POP + data$failed_alert) 

# Include in state: heat alerts within last 10 days, forecasted quantile of HI, 
  # actual quantile of heat HI on previous day, 
  # % of HHs in that county with income below the poverty line,
  # % of county land area classified as urban (or avg pop density?)
# Possibly include: holiday / weekend?

S<- data[1:(dim(data)[1]-1),c("quant_HI", "time")] # adjust variables and indexing later
S.1<- data[2:dim(data)[1],c("quant_HI", "time")] # adjust variables and indexing later

# For now, let's say the discount is:
discount<- 0.999

## Define basis functions:
phi<- list(function(x){x}, function(x){x^2})
# Example:
# unlist(lapply(phi, function(f) sapply(1:3, f)))

## LSTDQ:

lstdq<- function(S, A, R, S.1, phi_func, discount, Phi, b, policy){
  # Phi is a matrix of phi_func applied to the samples (s,a)
  # b is Phi^T %*% R 
  # policy is a vector of length |S|
  
  P.Pi.Phi<- t(sapply(1:1000, function(i){ # fix number of rows!
    s<- as.vector(unlist(lapply(phi, function(f) sapply(S.1[i,],f))))
    if(policy[i] == 1){
      c(s, rep(0,2*dim(S)[2]))
    }else{
      c(rep(0,2*dim(S)[2]), s)
    }
  }))
  
  A<- t(Phi)%*%(Phi - discount*P.Pi.Phi)
  
  # Then: 
  # Invert A
  # w = (A.inv)%*%b
  
}

lspi<- function(S, A, R, S.1, phi_func, discount){
  # A and R are vectors with the actions and rewards respectively 
  # S and S.1 are matrices with one "state" in each row
  # phi_func is a list of the basis functions
  # discount is a scalar between 0 and 1
  
  Phi<- t(sapply(1:1000, function(i){ # fix number of rows!
    s<- as.vector(unlist(lapply(phi, function(f) sapply(S[i,],f))))
    if(A[i] == 1){
      c(s, rep(0,2*dim(S)[2]))
    }else{
      c(rep(0,2*dim(S)[2]), s)
    }
  })) # Might actually just want to run this once per model and save it?
      # Maybe could use for bootstrapping if I keep track of row indices?
  
  b<- t(Phi) %*% R[1:1000]
  
  # Then apply lstdq and update the policy...
}



