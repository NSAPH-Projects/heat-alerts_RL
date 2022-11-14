
library(caret)

## Read in data:

load("data/Small_S-A-R_prepped.RData")

args<- commandArgs(trailing = TRUE) # outcome (string), date (string), constrained (bool)

outcome<- args[1]
date<- args[2]
constrained<- args[3]

## Visitation ratios:

if(constrained == TRUE){
  ranger_model<- readRDS(paste0("Fall_results/VisRat_", date, "_", # 11-8
                                outcome, "_constrained_75pct.rds"))
  new_pol<- read.csv(paste0("Fall_results/DQN_", date, "_", # 11-5
                            outcome, "_constrained_policy.csv"))$policy
}else{
  ranger_model<- readRDS(paste0("Fall_results/VisRat_", date, "_", 
                                outcome, "_75pct.rds"))
  new_pol<- read.csv(paste0("Fall_results/DQN_", date, "_", 
                            outcome, "_policy.csv"))$policy
}

VR_probs<- predict(ranger_model, DF, type = "prob")
VR_preds<- VR_probs$New/VR_probs$Behavior


## Policies:

behav<- readRDS("Fall_results/BART_preds_mean_NEW.rds")
behav<- behav[-seq(n_days, length(behav), n_days)]

pb<- behav
pb[which(A == 0)]<- 1-behav[which(A == 0)]


pg<- as.numeric(new_pol == A)

R<- R_hosps[,1]

H<- n_days-1

## OPE, no uncertainty:

if(constrained == TRUE){
  sink(paste0("new_results/VisRat-OPE_", date, "_", outcome, "_constrained.txt"))
}else{
  sink(paste0("new_results/VisRat-OPE_", date, "_", outcome, ".txt"))
}


summary(pg * VR_preds * R / pb)
hist(log(-pg * VR_preds * R / pb))

H*mean(R)
H*mean(pg * VR_preds * R / pb)

# eps<- 0.00001
# 
# n_days*mean(pg * VR_preds * R / (pb+eps))
# 
# w<- exp(rowSums(cbind(log(pg+eps), -log(pb+eps), log(VR_preds+eps))))
# n_days*mean(w*R)


## Bootstrap to account for uncertainty in R:

n_boots<- 1000

Res_behav<- rep(0, n_boots)
Res_new<- rep(0, n_boots)
N<- length(R)

set.seed(321)

for(i in 1:n_boots){
  inds<- sample(1:N, N, replace=TRUE)
  Res_behav[i]<- H*mean(R[inds])
  Res_new[i]<- H*mean(pg[inds] * VR_preds[inds] * R[inds] / pb[inds])
}

quantile(Res_behav, probs=c(0.05, 0.95))
quantile(Res_new, probs=c(0.05, 0.95))


## Accounting for uncertainty in pb:

behav<- readRDS("Fall_results/BART_preds_all_NEW.rds")
behav<- behav[,-seq(n_days, length(behav), n_days)]

pb<- behav
pb[,which(A == 0)]<- 1-behav[,which(A == 0)]

n<- nrow(pb)
shuffled_pb<- apply(pb, MARGIN=2, function(y){sample(y,n)})

Res_samps_pb<- apply(shuffled_pb, MARGIN=1, 
                     function(x) H*mean(pg * VR_preds * R / x))
quantile(Res_samps_pb, probs=c(0.05, 0.95))

n2_boots<- 500

Res_samps_pb_R<- matrix(0, nrow = n, ncol = n2_boots)

for(j in max(c(1,sum(Res_samps_pb_R[1,]!=0))):n2_boots){
  inds<- sample(1:N, N, replace=TRUE)
  Res_samps_pb_R[,j]<- apply(shuffled_pb[,inds], MARGIN=1, 
                       function(x) H*mean(pg[inds] * VR_preds[inds] * R[inds] / x))
  print(j)
}

# quantile(Res_samps_pb_R[,1:sum(Res_samps_pb_R[1,]!=0)], probs=c(0.05, 0.95))
quantile(Res_samps_pb_R, probs=c(0.05, 0.95))

sink()
