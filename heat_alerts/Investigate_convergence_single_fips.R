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

alerts_CPQ_folders<- list.files("Summer_results", pattern = "total_alerts_CPQ_observed-alerts_fips-6085_lr03")
alerts_DD_folders<- list.files("Summer_results", pattern = "total_alerts_Double_DQN_fips-6085_lr03")

for(f in c(alerts_CPQ_folders, alerts_DD_folders)){
  Alerts<- read.csv(paste0("Summer_results/",f))[,1]
  a_DF<-  data.frame(Epoch=1:length(Alerts), Alerts)
  p<- ggplot(a_DF, aes(x=Epoch, y=Alerts)) + geom_point() + 
    xlab("Epochs") + ylab("Days with Alerts") + geom_smooth() + ggtitle(f)
  print(p)
}

