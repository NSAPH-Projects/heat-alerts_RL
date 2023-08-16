library(ggplot2)


Alerts<- read.csv("Fall_results/MA_200_total_alerts_vanilla_DQN_small-S_lr1e-2.csv")[,1]

a_DF<-  data.frame(Epoch=1:length(Alerts), Alerts)
ggplot(a_DF, aes(x=Epoch, y=Alerts)) + geom_point() + 
  xlab("Epochs") + ylab("Days with Alerts") + geom_smooth() # + ggtitle("Vanilla DQN")

## With different seeds:

Alerts<- read.csv("Fall_results/MA_200_total_alerts_Double_DQN_small-S_lr5e-3.csv")[,1]

a_DF<-  data.frame(Epoch=1:length(Alerts), Alerts)
ggplot(a_DF, aes(x=Epoch, y=Alerts)) + geom_point() + 
  xlab("Epochs") + ylab("Days with Alerts") + geom_smooth() + ggtitle("Double_DQN")


models<- c("Double_DQN", "CPQ_observed-alerts")
seeds<- c(1,2,4:10)

for(m in models){
  for(s in seeds){
    Alerts<- read.csv(paste0("Fall_results/MA_200_total_alerts_", m, "_small-S_lr5e-3_seed-", s, ".csv"))[,1]
    a_DF<-  data.frame(Epoch=1:length(Alerts), Alerts)
    p<- ggplot(a_DF, aes(x=Epoch, y=Alerts)) + geom_point() + 
      xlab("Epochs") + ylab("Days with Alerts") + geom_smooth() + ggtitle(m)
    print(p)
  }
}

Alerts<- read.csv("Fall_results/MA_200_total_alerts_CPQ_observed-alerts_small-S_lr5e-3.csv")[,1]

a_DF<-  data.frame(Epoch=1:length(Alerts), Alerts)
ggplot(a_DF, aes(x=Epoch, y=Alerts)) + geom_point() + 
  xlab("Epochs") + ylab("Days with Alerts") + geom_smooth() + ggtitle("CPQ_observed-alerts")


