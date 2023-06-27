library(dplyr)
library(ggplot2)
library(cowplot, lib.loc = "~/apps/R_4.2.2")

folder<- "d3rlpy_logs/SC_DoubleDQN_Elig-90pct_MR-True_LR-1e-04_NH-2-16_B-32_fips-4013_seed-321_20230626113643/"

Loss<- read.csv(paste0(folder,"/loss.csv"), header = FALSE)
DF<- data.frame(Epoch=Loss$V1, Loss=Loss$V3)
ggplot(DF[1:nrow(DF),], aes(x=Epoch, y=log(Loss))) + geom_line() + 
  xlab("Epochs") + ylab("Log of Loss") 

f<- "Summer_results/SC_DoubleDQN_Elig-90pct_MR-True_LR-1e-04_NH-2-16_B-32_fips-4013_seed-321_MA_50_total_alerts_.csv"
Alerts<- read.csv(f)[,1]
a_DF<-  data.frame(Epoch=1:length(Alerts), Alerts)
ggplot(a_DF, aes(x=Epoch, y=Alerts)) + geom_point() + 
  xlab("Epochs") + ylab("Days with Alerts") + geom_smooth()

##########

CPQ_MR<- list.files("d3rlpy_logs", pattern = "Try-2_SC_CPQ_Elig-90pct_MR-T")
CPQ_OR<- list.files("d3rlpy_logs", pattern = "Try-2_SC_CPQ_Elig-90pct_MR-F")
DD_MR<- list.files("d3rlpy_logs", pattern = "Try-2_SC_DoubleDQN_Elig-90pct_MR-T")
DD_OR<- list.files("d3rlpy_logs", pattern = "Try-2_SC_DoubleDQN_Elig-90pct_MR-F")

for(f in c(CPQ_MR, DD_MR, CPQ_OR, DD_OR)){
  Loss<- read.csv(paste0("d3rlpy_logs/",f,"/loss.csv"), header = FALSE)
  DF<- data.frame(Epoch=Loss$V1, Loss=Loss$V3)
  p<- ggplot(DF[1:nrow(DF),], aes(x=Epoch, y=log(Loss))) + geom_line() + 
    xlab("Epochs") + ylab("Log of Loss") + ggtitle(f)
  print(p)
}

CPQ_MR_alerts<- list.files("Summer_results/", pattern = "Try-2_SC_CPQ_Elig-90pct_MR-T_")
CPQ_OR_alerts<- list.files("Summer_results/", pattern = "Try-2_SC_CPQ_Elig-90pct_MR-F_")
DD_MR_alerts<- list.files("Summer_results/", pattern = "Try-2_SC_DoubleDQN_Elig-90pct_MR-T_")
DD_OR_alerts<- list.files("Summer_results/", pattern = "Try-2_SC_DoubleDQN_Elig-90pct_MR-F_")

CPQ_MR_pol<- list.files("Policies/", pattern = "Policy_Try-2_SC_CPQ_Elig-90pct_MR-T_")
CPQ_OR_pol<- list.files("Policies/", pattern = "Policy_Try-2_SC_CPQ_Elig-90pct_MR-F_")
DD_MR_pol<- list.files("Policies/", pattern = "Policy_Try-2_SC_DoubleDQN_Elig-90pct_MR-T_")
DD_OR_pol<- list.files("Policies/", pattern = "Policy_Try-2_SC_DoubleDQN_Elig-90pct_MR-F_")

plot_files<- c(CPQ_MR_alerts, DD_MR_alerts, CPQ_OR_alerts, DD_OR_alerts)
pol_files<- c(CPQ_MR_pol, DD_MR_pol, CPQ_OR_pol, DD_OR_pol)

alerts<- c()
for(i in 1:length(plot_files)){
  Alerts<- read.csv(paste0("Summer_results/", plot_files[i]))[,1]
  a_DF<-  data.frame(Epoch=1:length(Alerts), Alerts)
  p<- ggplot(a_DF, aes(x=Epoch, y=Alerts)) + geom_point() +
    xlab("Epochs") + ylab("Days with Alerts") + geom_smooth() +
    ggtitle(plot_files[i]) + ylim(0,200)
  print(p)
  alerts<- append(alerts, sum(read.csv(paste0("Policies/",pol_files[i]))[,2]))
}
alerts

NH<- rep(c(128, 16, 32, 64), 6)
NL<- rep(c(2,2,2,2,3,3,3,3),3)
LR<- rep(c(0.001, 0.01, 0.0001), each = 8)
plot(NH[which(NL==2)], alerts[which(NL==2)])

##########

CPQ_folders<- list.files("d3rlpy_logs", pattern = "CPQ_observed-alerts_fips-6085")
DD_folders<- list.files("d3rlpy_logs", pattern = "Double_DQN_fips-6085")

for(f in c(CPQ_folders, DD_folders)){
  Loss<- read.csv(paste0("d3rlpy_logs/",f,"/loss.csv"), header = FALSE)
  DF<- data.frame(Epoch=Loss$V1, Loss=Loss$V3)
  p<- ggplot(DF[1:nrow(DF),], aes(x=Epoch, y=log(Loss))) + geom_line() + 
    xlab("Epochs") + ylab("Log of Loss") + ggtitle(f)
  print(p)
}

alerts_CPQ_folders<- list.files("Summer_results", pattern = "total_alerts_CPQ_observed-alerts_fips-6085")
alerts_DD_folders<- list.files("Summer_results", pattern = "total_alerts_Double_DQN_fips-6085")

for(f in c(alerts_CPQ_folders, alerts_DD_folders)){
  Alerts<- read.csv(paste0("Summer_results/",f))[,1]
  a_DF<-  data.frame(Epoch=1:length(Alerts), Alerts)
  p<- ggplot(a_DF, aes(x=Epoch, y=Alerts)) + geom_point() + 
    xlab("Epochs") + ylab("Days with Alerts") + geom_smooth() + ggtitle(f)
  print(p)
}

