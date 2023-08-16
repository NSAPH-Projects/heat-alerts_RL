
library(dplyr)
library(ggplot2)
library(cowplot, lib.loc = "~/apps/R_4.2.2")

pattern<- "SC_Pyro-1"
pattern<- "SC_Pyro-1_CPQ"

## Inspect loss:

loss_files<- list.files("d3rlpy_logs", pattern = pattern)
# loss_files<- loss_files[c(1:6, 13:18)]
# loss_files<- loss_files[c(19:30)]

for(f in loss_files){
  Loss<- read.csv(paste0("d3rlpy_logs/",f,"/loss.csv"), header = FALSE)
  DF<- data.frame(Epoch=Loss$V1, Loss=Loss$V3)
  p<- ggplot(DF[1:nrow(DF),], aes(x=Epoch, y=log(Loss))) + geom_line() + 
    xlab("Epochs") + ylab("Log of Loss") + ggtitle(f)
  print(p)
}

## Inspect alerts issued:

# pos<- 19:30 # 7:18
pos<- 1:12
plot_files<- list.files("Summer_results/", pattern = pattern)[pos]
pol_files<- list.files("Policies/", pattern = paste0("Policy_",pattern))[pos]

Days<- rep(1:153,10)
alerts<- c()
for(i in 1:length(plot_files)){
  Alerts<- read.csv(paste0("Summer_results/", plot_files[i]))[,1]
  a_DF<-  data.frame(Epoch=1:length(Alerts), Alerts)
  p<- ggplot(a_DF, aes(x=Epoch, y=Alerts)) + geom_point() +
    xlab("Epochs") + ylab("Days with Alerts") + geom_smooth() +
    ggtitle(plot_files[i]) + ylim(0,1600)
  print(p)
  A<- read.csv(paste0("Policies/",pol_files[i]))[,2]
  alerts<- append(alerts, sum(A))
  print(sum(A))
  print(Days[A==1])
}
alerts


## Evaluate DOS and streak length:


pol_stats<- t(sapply(1:length(pol_files), function(i){
  A<- read.csv(paste0("Policies/",pol_files[i]))[,2]
  if(sum(A) > 0){
    D<- Days[A==1]
    summary_dos<- summary(D)
    diffs<- D[2:length(D)] - D[1:(length(D)-1)]
    L<- rle(diffs)
    streaks<- L$lengths[which(L$values == 1)]
    num_streaks<- length(streaks)
    avg_streak_length<- mean(streaks + 1)
    return(c(as.vector(summary_dos), num_streaks, avg_streak_length))
  }else{
    return(rep(NA, 8))
  }
}))

Pol_stats<- data.frame(pol_stats)
names(Pol_stats)<- c("Min", "Q1", "Median", "Mean", "Q3", "Max", "N. streaks", "Avg. streak length")
Pol_stats

