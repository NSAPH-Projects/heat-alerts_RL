library(dplyr)
library(ggplot2)
library(cowplot, lib.loc = "~/apps/R_4.2.2")

CPQ_folders<- list.files("d3rlpy_logs", pattern = "CPQ_observed-alerts_fips-6085_lr05")
DD_folders<- list.files("d3rlpy_logs", pattern = "Double_DQN_fips-6085_lr05")

for(f in c(CPQ_folders, DD_folders)){
  Loss<- read.csv(paste0("d3rlpy_logs/",f,"/loss.csv"), header = FALSE)
  DF<- data.frame(Epoch=Loss$V1, Loss=Loss$V3)
  p<- ggplot(DF[1:nrow(DF),], aes(x=Epoch, y=log(Loss))) + geom_line() + 
    xlab("Epochs") + ylab("Log of Loss") + ggtitle(f)
  print(p)
}


Alerts<- read.csv("Summer_results/MA_200_total_alerts_Double_DQN_logMR_small-S_lr03_seed-3.csv")[,1]
Alerts<- read.csv("Summer_results/MA_200_total_alerts_CPQ_observed-alerts_logMR_small-S_lr03_seed-3.csv")[,1]
Alerts<- read.csv("Summer_results/MA_200_total_alerts_Double_DQN_logMR_small-S_lr03_seed-1.csv")[,1]
Alerts<- read.csv("Summer_results/MA_200_total_alerts_CPQ_observed-alerts_logMR_small-S_lr03_seed-1.csv")[,1]
Alerts<- read.csv("Summer_results/MA_200_total_alerts_Double_DQN_logMR_small-S_lr03_seed-2.csv")[,1]
Alerts<- read.csv("Summer_results/MA_200_total_alerts_CPQ_observed-alerts_logMR_small-S_lr03_seed-2.csv")[,1]

a_DF<-  data.frame(Epoch=1:length(Alerts), Alerts)
ggplot(a_DF, aes(x=Epoch, y=Alerts)) + geom_point() + 
  xlab("Epochs") + ylab("Days with Alerts") + geom_smooth() # + ggtitle("Vanilla DQN")
