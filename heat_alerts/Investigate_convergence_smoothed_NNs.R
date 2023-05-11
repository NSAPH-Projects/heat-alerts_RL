library(ggplot2)


Alerts<- read.csv("Fall_results/MA_200_total_alerts_vanilla_DQN_small-S_lr1e-2.csv")[,1]

a_DF<-  data.frame(Epoch=1:length(Alerts), Alerts)
ggplot(a_DF, aes(x=Epoch, y=Alerts)) + geom_point() + 
  xlab("Epochs") + ylab("Days with Alerts") + geom_smooth() # + ggtitle("Vanilla DQN")
