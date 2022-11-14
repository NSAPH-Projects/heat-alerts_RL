
library(caret)

## Read in data:

load("data/Small_S-A-R_prepped.RData")

## Visitation ratios:

ranger_model<- readRDS("Fall_results/VisRat_11-8_hosps_constrained_75pct.rds")

VR_probs<- predict(ranger_model, DF, type = "prob")
VR_preds<- VR_probs$New/VR_probs$Behavior


## Policies:

behav<- readRDS("Fall_results/BART_preds_mean_NEW.rds")
behav<- behav[-seq(n_days, length(behav), n_days)]

pb<- behav
pb[which(A == 0)]<- 1-behav[which(A == 0)]

new_pol<- read.csv("Fall_results/DQN_11-5_hosps_constrained_policy.csv")$policy

pg<- as.numeric(new_pol == A)

R<- R_hosps[,1]

H<- n_days-1

## OPE, no uncertainty:

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

Res_samps_pb_R<- matrix(0, nrow = n, ncol = n_boots)

for(j in max(c(1,sum(Res_samps_pb_R[1,]!=0))):n_boots){
  inds<- sample(1:N, N, replace=TRUE)
  Res_samps_pb_R[,j]<- apply(shuffled_pb[,inds], MARGIN=1, 
                       function(x) H*mean(pg[inds] * VR_preds[inds] * R[inds] / x))
  print(j)
}

quantile(Res_samps_pb_R[,1:sum(Res_samps_pb_R[1,]!=0)], probs=c(0.05, 0.95))

