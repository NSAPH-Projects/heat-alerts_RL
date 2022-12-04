
library(caret)

## Read in data:

load("data/Small_S-A-R_prepped.RData")

args<- commandArgs(trailing = TRUE) # outcome (string), date (string), constrained (bool)

outcome<- args[1]
date<- args[2]
constrained<- args[3]

## Visitation ratios:

if(constrained == TRUE){
  ranger_model<- readRDS(paste0("Fall_results/DensRat_", date, "_", # 11-8
                                outcome, "_constrained_75pct.rds"))
  new_pol<- read.csv(paste0("Fall_results/DQN_", date, "_", # 11-5
                            outcome, "_constrained_policy.csv"))$policy
}else{
  ranger_model<- readRDS(paste0("Fall_results/DensRat_", date, "_", 
                                outcome, "_75pct.rds"))
  new_pol<- read.csv(paste0("Fall_results/DQN_", date, "_", 
                            outcome, "_policy.csv"))$policy
}

VR_probs<- predict(ranger_model, DF, type = "prob")
VR_preds<- VR_probs$New/VR_probs$Behavior

VR_normed<- VR_preds/sum(VR_preds)


new_alert_sum<- rep(0,nrow(DF))
for(i in which(DF$dos == min(DF$dos))){
  new_alert_sum[i:(i+n_days-2)]<- cumsum(new_pol[i:(i+n_days-2)])
}

S_Budget<- rep(budget, each = n_days-1)
new_More_alerts<- S_Budget - new_alert_sum

new_DF<- DF
new_DF["alert_sum"]<- new_alert_sum
new_DF["More_alerts"]<- new_More_alerts
new_DF[,c("alert_sum", "More_alerts")]<- scale(new_DF[,c("alert_sum", "More_alerts")])
new_DF$alert<- new_pol

## Policies:

behav<- readRDS("Fall_results/BART_preds_mean_11-20.rds")
behav<- behav[-seq(n_days, length(behav), n_days)]

pb<- behav
pb[which(A == 0)]<- 1-behav[which(A == 0)]


pg<- as.numeric(new_pol == A)

if(outcome == "hosps"){
  R<- R_other_hosps[,1]
  # R<- R_hosps[,1]
  modeled_R<- read.csv("Fall_results/R_11-27_other-hosps.csv")*1000
  R_b<- modeled_R$X0
  R_b[A == 1]<- modeled_R$X1[A == 1]
  # R_model<- readRDS("Fall_results/Rewards_VarImp.rds")
}else{
  R<- R_deaths[,1]
}

new_R<- predict(R_model, new_DF)


H<- n_days-1

## OPE, no uncertainty:

if(constrained == TRUE){
  sink(paste0("new_results/VisRat-OPE_", date, "_", outcome, "_constrained.txt"))
}else{
  sink(paste0("new_results/VisRat-OPE_", date, "_", outcome, ".txt"))
}

H*mean(R)
H*mean(new_R)

# summary(VR_normed * R)
# hist(log(-VR_normed*R))
# 
# H*mean(R)
# H*sum(VR_normed*R)
# 
# H*mean(R_b)
# H*sum(VR_normed*R_b)

# p.a<- pg/pb
# p.a_normed<- p.a/sum(p.a)
# H*length(p.a)*sum(p.a_normed * VR_normed * R)

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
