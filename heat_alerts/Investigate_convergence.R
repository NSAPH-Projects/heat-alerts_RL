library(dplyr)
library(ggplot2)
library(cowplot)

setwd("/n/dominici_nsaph_l3/Lab/projects/heat-alerts_mortality_RL")

#### Using d3rlpy:
folder<- "d3rlpy_logs/vanilla_DQN_20230227173321"
folder<- "d3rlpy_logs/vanilla_DQN_modeled-R_20230228145703"
folder<- "d3rlpy_logs/vanilla_DQN_modeled-R_rand-effs_not-forcing_20230304153017"
Loss_a<- read.csv(paste0(folder,"/loss.csv"), header = FALSE)
folder<- "d3rlpy_logs/vanilla_DQN_modeled-R_rand-effs_not-forcing_more-epochs_20230303122304"
Loss_b<- read.csv(paste0(folder,"/loss.csv"), header = FALSE)
Loss_b$V1<- Loss_b$V1 + nrow(Loss_a)
Loss<- rbind(Loss_a, Loss_b)

folder<- "d3rlpy_logs/vanilla_DQN_modeled-R_rand-effs_not-forcing_constrained_90pct_20230304224723" 
folder<- "d3rlpy_logs/vanilla_DQN_default-params_20230320163928"
folder<- "d3rlpy_logs/vanilla_DQN_default-params_modeled-R_rand-effs_not-forcing_20230321143836"
folder<- "d3rlpy_logs/vanilla_DQN_default-params_constrained_90pct_20230321175057"
folder<- "d3rlpy_logs/vanilla_DQN_default-params_modeled-R_rand-effs_not-forcing_constrained_90pct_20230321192541"
folder<- "d3rlpy_logs/vanilla_DQN_lr-003_20230322190102"
folder<- "d3rlpy_logs/vanilla_DQN_lr-0005_20230322192458"
folder<- "d3rlpy_logs/vanilla_DQN_lr-0001_20230322195401"

folder<- "d3rlpy_logs/vanilla_DQN_lr-0001_long-run_20230324140132"
folder<- "d3rlpy_logs/vanilla_DQN_lr-0001_long-run_constrained_90pct_20230324140128"
folder<- "d3rlpy_logs/vanilla_DQN_lr-0001_long-run_modeled-R_rand-effs_not-forcing_20230324140130"
folder<- "d3rlpy_logs/vanilla_DQN_lr-0001_long-run_modeled-R_rand-effs_not-forcing_constrained_90pct_20230324140126"

folder<- "d3rlpy_logs/vanilla_DQN_lr-0001_long-run_20230325095107"
folder<- "d3rlpy_logs/vanilla_DQN_lr-0001_long-run_modeled-R_rand-effs_not-forcing_20230325095126"
folder<- "d3rlpy_logs/vanilla_DQN_lr-0001_long-run_constrained_90pct_20230326055014"
folder<- "d3rlpy_logs/vanilla_DQN_lr-0001_long-run_modeled-R_rand-effs_not-forcing_constrained_90pct_20230326055944"

folder<- "d3rlpy_logs/vanilla_DQN_nonnegative-R_20230404170322/"
folder<- "d3rlpy_logs/Double_DQN_20230401143034/"
folder<- "d3rlpy_logs/CPQ_observed-alerts_20230404173637"

folder<- "d3rlpy_logs/vanilla_DQN_centered-R_20230405112316/"
folder<- "d3rlpy_logs/Double_DQN_centered-R_20230405112316/"
folder<- "d3rlpy_logs/CPQ_observed-alerts_centered-R_20230405112316/"

folder<- "d3rlpy_logs/Double_DQN_w-index_20230405172205/"
folder<- "d3rlpy_logs/CPQ_observed-alerts_w-index_20230405172205/"

folder<- "d3rlpy_logs/vanilla_DQN_sync-10_20230406123838/"
folder<- "d3rlpy_logs/Double_DQN_sync-10_20230406123836/"
folder<- "d3rlpy_logs/CPQ_observed-alerts_sync-10_20230406231919/"

folder<- "d3rlpy_logs/vanilla_DQN_lr3e-3sr3_20230407163616/"
folder<- "d3rlpy_logs/Double_DQN_lr3e-3sr3_20230407163612/"
folder<- "d3rlpy_logs/CPQ_observed-alerts_lr3e-3sr3_20230408030457/"

folder<- "d3rlpy_logs/vanilla_DQN_lr1e-3sr10_20230408094450/" # intentionally cut off early 

folder<- "d3rlpy_logs/vanilla_DQN_lr1e-3sr15_modeled-R_20230408160515/"
folder<- "d3rlpy_logs/vanilla_DQN_lr1e-3sr10_modeled-R_20230408124037/"
Loss_a<- read.csv(paste0(folder,"/loss.csv"), header = FALSE)
folder<- "d3rlpy_logs/vanilla_DQN_lr1e-3sr10_modeled-R_part-2_20230410085922/"
Loss_b<- read.csv(paste0(folder,"/loss.csv"), header = FALSE)
Loss_b$V1<- Loss_b$V1 + nrow(Loss_a)
Loss<- rbind(Loss_a, Loss_b)

folder<- "d3rlpy_logs/vanilla_DQN_lr1e-3sr10_w-state_20230409102256/" 
folder<- "d3rlpy_logs/CPQ_observed-alerts_lr1e-3sr10_w-state_20230409232657/"

folder<- "d3rlpy_logs/vanilla_DQN_lr1e-3sr10_gamma-99_20230418110923/"
folder<- "d3rlpy_logs/vanilla_DQN_lr1e-2_20230418110922/"

Loss<- read.csv(paste0(folder,"/loss.csv"), header = FALSE)
Loss_compare<- read.csv(paste0("d3rlpy_logs/vanilla_DQN_lr1e-3sr10_20230408094450/","/loss.csv"), header = FALSE)
Loss_compare<- read.csv(paste0("d3rlpy_logs/vanilla_DQN_lr3e-3sr3_20230407163616/","/loss.csv"), header = FALSE)
n<- min(nrow(Loss), nrow(Loss_compare))
DF<- data.frame(Epoch=Loss$V1[1:n], Loss=Loss$V3[1:n], Loss_compare=Loss_compare$V3[1:n])

ggplot(DF, aes(x=Epoch, y=log(Loss))) + geom_line() + 
  xlab("Epochs") + ylab("Log of Loss") + ggtitle("DQN Model") +
  geom_line(aes(y=log(Loss_compare), col="orange"))


folder<- "d3rlpy_logs/vanilla_DQN_tiny-S_20230419090929/"
folder<- "d3rlpy_logs/CPQ_observed-alerts_tiny-S_20230419090923/"

folder<- "d3rlpy_logs/vanilla_DQN_symlog-R_20230420095217/"
folder<- "d3rlpy_logs/vanilla_DQN_symlog-R_lr1e-2_20230420095231/"

folder<- "d3rlpy_logs/vanilla_DQN_pca-1_20230423115829/"
folder<- "d3rlpy_logs/vanilla_DQN_pca-05_20230423115833/"

folder<- "d3rlpy_logs/vanilla_DQN_small-S_20230423140350/"
folder<- "d3rlpy_logs/vanilla_DQN_medium-S_20230423140352/"

folder<- "d3rlpy_logs/vanilla_DQN_tiny-S_b_20230423165905/"
folder<- "d3rlpy_logs/vanilla_DQN_pca-0_/" # not sure what happened to this?

folder<- "d3rlpy_logs/vanilla_DQN_small-S_all-data_20230424085551/"
folder<- "d3rlpy_logs/vanilla_DQN_small-S_lr1e-2_20230424085551/"
# folder<- "d3rlpy_logs/vanilla_DQN_small-S_lr1e-3_20230424085558/"

folder<- "d3rlpy_logs/vanilla_DQN_small-S_lr5e-2_20230425180555/"
folder<- "d3rlpy_logs/Double_DQN_small-S_lr1e-2_20230425180552/"
folder<- "d3rlpy_logs/CPQ_observed-alerts_small-S_lr1e-2_20230425180553/"

folder<- "d3rlpy_logs/vanilla_DQN_small-S_lr1e-2_20230424085551/"
Loss_a<- read.csv(paste0(folder,"/loss.csv"), header = FALSE)
TDE_a<- read.csv(paste0(folder,"/td_error.csv"), header = FALSE)
folder<- "d3rlpy_logs/vanilla_DQN_small-S_lr1e-2_B_20230426131736/"
Loss_b<- read.csv(paste0(folder,"/loss.csv"), header = FALSE)
TDE_b<- read.csv(paste0(folder,"/td_error.csv"), header = FALSE)
Loss_b$V1<- Loss_b$V1 + nrow(Loss_a)
TDE_b$V1<- TDE_b$V1 + nrow(TDE_a)
Loss<- rbind(Loss_a, Loss_b)
TDE<- rbind(TDE_a, TDE_b)

folder<- "d3rlpy_logs/Double_DQN_small-S_lr5e-3_20230426131739/"
folder<- "d3rlpy_logs/CPQ_observed-alerts_small-S_lr5e-3_20230426131737/"

folder<- "d3rlpy_logs/Double_DQN_small-S_lr1e-3_20230428091612/"
folder<- "d3rlpy_logs/CPQ_observed-alerts_small-S_lr1e-3_20230428091615/"
folder<- "d3rlpy_logs/Double_DQN_medium-S_lr5e-3_20230428091614/"

Loss<- read.csv(paste0(folder,"/loss.csv"), header = FALSE)
DF<- data.frame(Epoch=Loss$V1, Loss=Loss$V3)
ggplot(DF[5:nrow(DF),], aes(x=Epoch, y=log(Loss))) + geom_line() + 
  xlab("Epochs") + ylab("Log of Loss") # + ggtitle("DQN Model")

TDE<- read.csv(paste0(folder,"/td_error.csv"), header = FALSE)
DF_tde<- data.frame(Epoch=TDE$V1, TD_Error=TDE$V3)
ggplot(DF_tde, aes(x=Epoch, y=TD_Error)) + geom_line() + 
  xlab("Epochs") + ylab("TD Error") # + ggtitle("DQN Model")

AVS<- read.csv(paste0(folder,"/value_scale.csv"), header = FALSE)
DF_avs<- data.frame(Epoch=AVS$V1, Value_Scale=AVS$V3)
ggplot(DF_avs, aes(x=Epoch, y=Value_Scale)) + geom_line() + 
  xlab("Epochs") + ylab("Average Value Scale") # + ggtitle("DQN Model")

## With different seeds:
models<- c("Double_DQN", "CPQ_observed-alerts")
seeds<- c(1,2,4:10)

for(m in models){
  for(s in seeds){
    folder<- list.files(path = "d3rlpy_logs/", pattern = paste0(m, "_small-S_lr5e-3_seed-", s, "_"))
    Loss<- read.csv(paste0("d3rlpy_logs/", folder,"/loss.csv"), header = FALSE)
    DF<- data.frame(Epoch=Loss$V1, Loss=Loss$V3)
    p<- ggplot(DF[5:nrow(DF),], aes(x=Epoch, y=log(Loss))) + geom_line() + 
      xlab("Epochs") + ylab("Log of Loss") + ggtitle(m)
    plot(p)
  }
}


###########################################################

Alerts<- read.csv("Fall_results/Total_alerts_vanilla_DQN_default-params.csv")[,1]
Alerts<- read.csv("Fall_results/Total_alerts_vanilla_DQN_default-params_modeled-R_rand-effs_not-forcing.csv")[,1]
Alerts<- read.csv("Fall_results/Total_alerts_vanilla_DQN_default-params_constrained_90pct.csv")[,1]
Alerts<- read.csv("Fall_results/Total_alerts_vanilla_DQN_default-params_modeled-R_rand-effs_not-forcing_constrained_90pct.csv")[,1]
Alerts<- read.csv("Fall_results/Total_alerts_vanilla_DQN_lr-003.csv")[,1]
Alerts<- read.csv("Fall_results/Total_alerts_vanilla_DQN_lr-0005.csv")[,1]
Alerts<- read.csv("Fall_results/Total_alerts_vanilla_DQN_lr-0001.csv")[,1]

Alerts<- read.csv("Fall_results/Total_alerts_vanilla_DQN_lr-0001_long-run.csv")[,1]
Alerts<- read.csv("Fall_results/Total_alerts_vanilla_DQN_lr-0001_long-run_modeled-R_rand-effs_not-forcing.csv")[,1]
Alerts<- read.csv("Fall_results/Total_alerts_vanilla_DQN_lr-0001_long-run_constrained_90pct.csv")[,1]
Alerts<- read.csv("Fall_results/Total_alerts_vanilla_DQN_lr-0001_long-run_modeled-R_rand-effs_not-forcing_constrained_90pct.csv")[,1]

Alerts<- read.csv("Fall_results/Total_alerts_vanilla_DQN_nonnegative-R.csv")[,1]
Alerts<- read.csv("Fall_results/Total_alerts_Double_DQN.csv")[,1]
Alerts<- read.csv("Fall_results/Total_alerts_CPQ_observed-alerts.csv")[,1]

Alerts<- read.csv("Fall_results/Total_alerts_Double_DQN_w-index.csv")[,1]
Alerts<- read.csv("Fall_results/Total_alerts_CPQ_observed-alerts_w-index.csv")[,1]

Alerts<- read.csv("Fall_results/Total_alerts_vanilla_DQN_sync-10.csv")[,1]
Alerts<- read.csv("Fall_results/Total_alerts_Double_DQN_sync-10.csv")[,1]
Alerts<- read.csv("Fall_results/Total_alerts_CPQ_observed-alerts_sync-10.csv")[,1]

Alerts<- read.csv("Fall_results/Total_alerts_vanilla_DQN_lr3e-3sr3.csv")[,1]
Alerts<- read.csv("Fall_results/Total_alerts_Double_DQN_lr3e-3sr3.csv")[,1]
Alerts<- read.csv("Fall_results/Total_alerts_CPQ_observed-alerts_lr3e-3sr3.csv")[,1]

Alerts<- read.csv("Fall_results/Total_alerts_vanilla_DQN_lr1e-3sr10_modeled-R.csv")[,1]
Alerts<- read.csv("Fall_results/Total_alerts_vanilla_DQN_lr1e-3sr15_modeled-R.csv")[,1]

Alerts<- read.csv("Fall_results/Total_alerts_vanilla_DQN_lr1e-3sr10_w-state.csv")[,1]
Alerts<- read.csv("Fall_results/Total_alerts_CPQ_observed-alerts_lr1e-3sr10_w-state.csv")[,1]

Alerts_a<- read.csv("Fall_results/Total_alerts_vanilla_DQN_lr1e-3sr10_modeled-R.csv")[,1]
Alerts_b<- read.csv("Fall_results/Total_alerts_vanilla_DQN_lr1e-3sr10_modeled-R_part-2.csv")[,1]
Alerts<- append(Alerts_a, Alerts_b)

Alerts<- read.csv("Fall_results/Total_alerts_vanilla_DQN_lr1e-3sr10_gamma-99.csv")[,1]
Alerts<- read.csv("Fall_results/Total_alerts_vanilla_DQN_lr1e-2.csv")[,1]

Alerts<- read.csv("Fall_results/Total_alerts_vanilla_DQN_tiny-S.csv")[,1]
Alerts<- read.csv("Fall_results/Total_alerts_CPQ_observed-alerts_tiny-S.csv")[,1]

Alerts<- read.csv("Fall_results/Total_alerts_vanilla_DQN_symlog-R.csv")[,1]
Alerts<- read.csv("Fall_results/Total_alerts_vanilla_DQN_symlog-R_lr1e-2.csv")[,1]

Alerts<- read.csv("Fall_results/Total_alerts_vanilla_DQN_small-S.csv")[,1]
Alerts<- read.csv("Fall_results/Total_alerts_vanilla_DQN_medium-S.csv")[,1]
Alerts<- read.csv("Fall_results/Total_alerts_vanilla_DQN_tiny-S_b.csv")[,1]

Alerts_a<- read.csv("Fall_results/Total_alerts_vanilla_DQN_small-S_lr1e-2.csv")[,1]
Alerts_b<- read.csv("Fall_results/Total_alerts_vanilla_DQN_small-S_lr1e-2_B.csv")[,1]
Alerts<- append(Alerts_a, Alerts_b)

Alerts<- read.csv("Fall_results/Total_alerts_vanilla_DQN_small-S_lr5e-2.csv")[,1]
Alerts<- read.csv("Fall_results/Total_alerts_Double_DQN_small-S_lr1e-2.csv")[,1]
Alerts<- read.csv("Fall_results/Total_alerts_CPQ_observed-alerts_small-S_lr1e-2.csv")[,1]

Alerts<- read.csv("Fall_results/Total_alerts_Double_DQN_small-S_lr5e-3.csv")[,1]
Alerts<- read.csv("Fall_results/Total_alerts_CPQ_observed-alerts_small-S_lr5e-3.csv")[,1]

Alerts<- read.csv("Fall_results/Total_alerts_Double_DQN_medium-S_lr5e-3.csv")[,1]
Alerts<- read.csv("Fall_results/Total_alerts_Double_DQN_small-S_lr1e-3.csv")[,1]
Alerts<- read.csv("Fall_results/Total_alerts_CPQ_observed-alerts_small-S_lr1e-3.csv")[,1]

a_DF<-  data.frame(Epoch=1:length(Alerts), Alerts)
ggplot(a_DF, aes(x=Epoch, y=Alerts)) + geom_point() + 
  xlab("Epochs") + ylab("Days with Alerts") + geom_smooth() # + ggtitle("Vanilla DQN")

## Make figure for "F31" proposal:

Alerts<- read.csv("Fall_results/Total_alerts_vanilla_DQN.csv")[,1]/(153*11*596)
a_DF<-  data.frame(Epoch=1:length(Alerts), Alerts)
p1<- ggplot(a_DF, aes(x=Epoch, y=Alerts)) + geom_point() + 
  xlab("Epochs") + ylab("Days with Alerts") + ggtitle("Vanilla DQN") + geom_smooth()

Alerts_a<- read.csv("Fall_results/Total_alerts_vanilla_DQN_modeled-R_rand-effs_not-forcing.csv")[,1]
Alerts_b<- read.csv("Fall_results/Total_alerts_vanilla_DQN_modeled-R_rand-effs_not-forcing_more-epochs.csv")[,1]
Alerts<- append(Alerts_a, Alerts_b)/(153*11*596)
a_DF<-  data.frame(Epoch=1:length(Alerts), Alerts)
p2<- ggplot(a_DF, aes(x=Epoch, y=Alerts)) + geom_point() + 
  xlab("Epochs") + ylab("Days with Alerts") + ggtitle("DQN with Modeled Rewards (MR)") + geom_smooth() +
  ylim(0,1)

Alerts<- read.csv("Fall_results/Total_alerts_vanilla_DQN_constrained_90pct.csv")[,1]/sum(read.csv("data/Pct_90_eligible.csv"))
a_DF<-  data.frame(Epoch=1:length(Alerts), Alerts)
p3<- ggplot(a_DF, aes(x=Epoch, y=Alerts)) + geom_point() + 
  xlab("Epochs") + ylab("Days with Alerts") + ggtitle("DQN with Eligibility Constraint (EC)") + geom_smooth() +
  ylim(0,1)

Alerts<- read.csv("Fall_results/Total_alerts_vanilla_DQN_modeled-R_rand-effs_not-forcing_constrained_90pct.csv")[,1]/sum(read.csv("data/Pct_90_eligible.csv"))
a_DF<-  data.frame(Epoch=1:length(Alerts), Alerts)
p4<- ggplot(a_DF, aes(x=Epoch, y=Alerts)) + geom_point() + 
  xlab("Epochs") + ylab("Days with Alerts") + ggtitle("DQN with EC + MR") + geom_smooth() +
  ylim(0,1)

plot_grid(p1,p2,p3,p4, nrow=2)

#### Pytorch convergence in terms of loss:

# DQN<- read.csv("Fall_results/DQN_10-18_epoch-losses.csv")
# DQN<- read.csv("lightning_logs/constr_deaths_adam_huber/version_4/metrics.csv")
# DQN<- read.csv("lightning_logs/feb_constr_hosps_adam_huber_REs/version_5/metrics.csv") # not separating preds_1 and RS in softplus
DQN_a<- read.csv("lightning_logs/feb_constr_hosps_adam_huber_REs/version_6/metrics.csv") # additive RS
DQN_b<- read.csv("lightning_logs/feb_constr_hosps_adam_huber_REs/version_8/metrics.csv")
DQN_c<- read.csv("lightning_logs/feb_constr_hosps_adam_huber_REs/version_10/metrics.csv")
DQN_b$epoch<- 1:nrow(DQN_b) + 1999
DQN<- rbind(DQN_a, DQN_b)
DQN_c$epoch<- 1:nrow(DQN_c) + 3499
DQN<- rbind(DQN, DQN_c)
DQN_a<- read.csv("lightning_logs/feb_constr_hosps_adam_huber_REs/version_7/metrics.csv") # additive RS with lr=0.001
DQN_b<- read.csv("lightning_logs/feb_constr_hosps_adam_huber_REs/version_9/metrics.csv")
DQN_c<- read.csv("lightning_logs/feb_constr_hosps_adam_huber_REs/version_11/metrics.csv")
# DQN_a<- read.csv("lightning_logs/constr_hosps_adam_huber_REs/version_0/metrics.csv")
# DQN_b<- read.csv("lightning_logs/constr_hosps_adam_huber_REs/version_1/metrics.csv")
DQN_b$epoch<- 1:nrow(DQN_b) + 1999
DQN<- rbind(DQN_a, DQN_b)
DQN_c$epoch<- 1:nrow(DQN_c) + 4999
DQN<- rbind(DQN, DQN_c)
# DQN<- read.csv("lightning_logs/constr_all-hosps_adam_huber/version_0/metrics.csv")
DQN_a<- read.csv("lightning_logs/feb_constr_hosps_adam_huber_REs/version_12/metrics.csv")
DQN_b<- read.csv("lightning_logs/feb_constr_hosps_adam_huber_REs/version_14/metrics.csv")
DQN_b$epoch<- 1:nrow(DQN_b) + 1499
DQN<- rbind(DQN_a, DQN_b)
# DQN<- read.csv("lightning_logs/feb_constr_hosps_adam_huber_REs/version_13/metrics.csv")
DQN<- read.csv("lightning_logs/feb_constr_hosps_adam_huber_REs/version_18/metrics.csv")

ggplot(DQN, aes(x=epoch, y=log(epoch_loss))) + geom_line() + 
  xlab("Epochs") + ylab("Log of Huber Loss") + ggtitle("DQN Model")

ggplot(DQN, aes(x=epoch, y=Q1.Q0)) + geom_line() + 
  xlab("Epochs") + ylab("Number of Days Q1 > Q0") + ggtitle("DQN Model")

ggplot(DQN, aes(x=epoch, y=n_alerts)) + geom_line() + 
  xlab("Epochs") + ylab("Number of Alerts Issued") + ggtitle("DQN Model")

ggplot(DQN, aes(x=epoch, y=log(n_alerts/Q1.Q0))) + geom_line() + 
  xlab("Epochs") + ylab("Log of {Number of Alerts Issued / Number of Days Q1 > Q0}") + ggtitle("DQN Model")

### Validation vs Training, Rewards and Alerts Models

# DF<- read.csv("lightning_logs/R_no-past_deaths_adam_mse/version_0/metrics.csv")
DF<- read.csv("lightning_logs/feb_R_hosps_adam_mse/version_5/metrics.csv")
DF<- read.csv("lightning_logs/feb_R_other-hosps_adam_mse/version_5/metrics.csv")

Val_Loss<- DF[seq(1,nrow(DF),2),1]
Train_Loss<- DF[seq(2,nrow(DF),2),5]
Epoch<- 1:length(Val_Loss)

# DF<- read.csv("lightning_logs/test_NN_alerts/version_13/metrics.csv")
# 
# Val_Loss<- DF[seq(1,nrow(DF),2),1]
# Train_Loss<- DF[seq(2,nrow(DF),2),4]
# Epoch<- 1:length(Val_Loss)


Plot_df<- data.frame(Epoch, Train_Loss, Val_Loss)[-1,]

ggplot(Plot_df[1:nrow(Plot_df),], aes(x=Epoch)) + geom_line(aes(y=Train_Loss)) + 
  geom_line(aes(y=Val_Loss), col = "blue") +
  ylab("Loss") + ggtitle("Model Convergence")

#### Torch lm:

Coefs1<- readRDS("Aug_results/Q-coefficients_9-7.rds")
Coefs2<- readRDS("Aug_results/Q-coefficients_9-8.rds")
Coefs<- rbind(Coefs1, Coefs2)

## Looking at coefs all together:

c0<- Coefs[1:(nrow(Coefs)-1),]
c1<- Coefs[2:nrow(Coefs),]

res<- apply(cbind(c0, c1), MARGIN = 1, function(x) sqrt(mean((x[49:96] - x[1:48])^2)))

plot(1:length(res), res)

## Now looking at the individual coefs:

cols<- readRDS("data/Model_colnames_9-8.rds")
colnames(Coefs)<- cols

par(mfrow=c(3,4))
for(j in 1:ncol(Coefs)){
  plot(1:nrow(Coefs), Coefs[,j], main = cols[j])
}

## Looking RMSE of Q-values over time:

load("data/Train-Test.RData")

n_counties<- length(unique(Train$GEOID))
n_years<- 11
n_days<- 153

States<- Train[, c("HImaxF_PopW", "quant_HI_county", "quant_HI_yest_county",
                   "quant_HI_3d_county", "quant_HI_fwd_avg_county",
                   "BA_zone", "Pop_density", "Med.HH.Income",
                   "year", "dos", "holiday", # "Holiday", 
                   "dow", "alert_sum")] # same variables as in model for alerts

States<- States[-seq(n_days, nrow(Train), n_days),]
S<- States %>% mutate_if(is.numeric, scale)

S_full_0<- model.matrix(~ A*., data.frame(A=0,S))
S_full_1<- model.matrix(~ A*., data.frame(A=1,S))

rm("Train")
rm("Test")
rm("Coefs1")
rm("Coefs2")
rm("States")
gc()

df0<- S_full_0 %*% Coefs[nrow(Coefs),]
df1<- S_full_1 %*% Coefs[nrow(Coefs),]

## Just look at a subsequence:
ss<- seq(1, nrow(Coefs), 20)

DF0<- S_full_0 %*% t(Coefs[ss,])
DF1<- S_full_1 %*% t(Coefs[ss,])

q_diff_0<- rep(0,length(ss))
q_diff_1<- rep(0,length(ss))

for(j in 2:length(ss)){
  q_diff_0[j]<- sqrt(mean((DF0[,j] - DF0[,j-1])^2))
  q_diff_1[j]<- sqrt(mean((DF1[,j] - DF1[,j-1])^2))
}

plot(1:length(ss), q_diff_0)
plot(1:length(ss), q_diff_1) # does actually look like we've converged?


##########################################################

## Read in text file:
res<- read.table("Aug_results/First_SGD.txt", fill=TRUE)

## Spread in coefficients:
diff_coefs<- res[seq(2,nrow(res),28),2]
plot(1:999, diff_coefs, type = "l")

## Value of intervention, all else equal:
A_coef<- res[seq(6,nrow(res), 28),2]
plot(1:999, A_coef)

## Effect of alert_sum:
alert_sum<- res[seq(18,nrow(res), 28),1]
plot(1:999, alert_sum)

## Effect of Med HH Income:
med_HH<- res[seq(14,nrow(res), 28),3]
plot(1:999, med_HH)
