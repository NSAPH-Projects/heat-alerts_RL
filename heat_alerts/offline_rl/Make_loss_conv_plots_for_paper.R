
library(dplyr)
library(ggplot2)
library(cowplot)

v0<- read.csv("d3rlpy_logs/vanilla_DQN_lr3e-3sr3_20230407163616/loss.csv", header = FALSE)
v1<- read.csv("d3rlpy_logs/vanilla_DQN_lr1e-2_20230418110922/loss.csv", header = FALSE)
v2<- read.csv("d3rlpy_logs/vanilla_DQN_symlog-R_lr1e-2_20230420095231/loss.csv", header = FALSE)
DF.1<- data.frame(Epoch=rep(1:10000, 3), Loss=c(v0$V3, c(v1$V3, rep(NA, 5000)), v2$V3), 
                  Setup=c(rep("LR=0.003, log(R)", 10000), 
                                 rep("LR=0.01, log(R)", 10000),
                                 rep("LR=0.01, symlog(R)", 10000)))
ggplot(DF.1, aes(x=Epoch, y=log(Loss), color=Setup)) + geom_line() + 
  xlab("Epochs") + ylab("Log of Loss") + 
  ggtitle("Vanilla DQN with 90% Heat Index Days and Large # Covariates")


# folder<- "d3rlpy_logs/vanilla_DQN_small-S_lr1e-2_20230424085551/"
# Loss_a<- read.csv(paste0(folder,"/loss.csv"), header = FALSE)
# folder<- "d3rlpy_logs/vanilla_DQN_small-S_lr1e-2_B_20230426131736/"
# Loss_b<- read.csv(paste0(folder,"/loss.csv"), header = FALSE)
# Loss_b$V1<- Loss_b$V1 + nrow(Loss_a)
# v3<- rbind(Loss_a, Loss_b)

v4<- read.csv("d3rlpy_logs/vanilla_DQN_small-S_20230423140350/loss.csv", header = FALSE)
v5<- read.csv("d3rlpy_logs/vanilla_DQN_small-S_all-data_20230424085551/loss.csv", header = FALSE)
n<- nrow(v5)
DF.3<- data.frame(Epoch=c(1:n, 1:n), Loss=c(v4$V3[1:n], v5$V3), 
                  Eligible=c(rep("90% heat index", n), rep("All days", n)))
ggplot(DF.3, aes(x=Epoch, y=log(Loss), color=Eligible)) + geom_line() + 
  xlab("Epochs") + ylab("Log of Loss") + 
  ggtitle("Vanilla DQN with Small # Covariates, LR = 0.003, and symlog(R)")

v3<- read.csv("d3rlpy_logs/vanilla_DQN_small-S_lr1e-2_20230424085551/loss.csv", header = FALSE)
v6<- read.csv("d3rlpy_logs/Double_DQN_small-S_lr1e-2_20230425180552/loss.csv", header = FALSE)
v7<- read.csv("d3rlpy_logs/CPQ_observed-alerts_small-S_lr1e-2_20230425180553/loss.csv", header = FALSE)
DF.4<- data.frame(Epoch=c(1:10000, 1:10000, 1:10000), Loss=c(v3$V3, v6$V3, v7$V3), 
                  Model=c(rep("vanilla DQN", 10000), rep("Double DQN", 10000), rep("CPQ with budget = obs", 10000)))
ggplot(DF.4, aes(x=Epoch, y=log(Loss), color=Model)) + geom_line() + 
  xlab("Epochs") + ylab("Log of Loss") + 
  ggtitle("Models with 90% Heat Index Days, Small # Covariates, 
                    LR = 0.01, and symlog(R)")


