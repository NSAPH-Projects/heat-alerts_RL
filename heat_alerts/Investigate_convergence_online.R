
library(dplyr)
library(ggplot2)
library(cowplot, lib.loc = "~/apps/R_4.2.2")
library(stringr)

model<- "Online-1_DoubleDQN"
model<- "Online-1_SAC"

i<- 5
these_plots<- seq(i, 30, 5)

########### Final evaluations:

## Benchmarks:
df<- read.csv("Summer_results/ORL_eval_NWS.csv")[,-1]
df<- read.csv("Summer_results/ORL_eval_zero.csv")[,-1]

DF<- aggregate(. ~ County + Year
               , df, sum)
fips<- c(36005, 41067, 28035, 6071, 4013)
eval_years<- c(2015, 2011, 2007)

eval<- matrix(0, ncol=3, nrow=length(fips))
for(i in 1:length(fips)){
  # print(DF[which(DF$County == fips[i] & DF$Year %in% eval_years),])
  eval[i,]<- as.vector(unlist(aggregate(. ~ County, 
            DF[which(DF$County == fips[i] & 
                       DF$Year %in% eval_years),
               c("County", "Actions", "Rewards")], sum)))
}
Eval<- as.data.frame(eval)
names(Eval)<- c("County", "Actions", "Rewards")
Eval

# ggplot(DF, aes(x=County,y=Rewards
#                    # ,color=as.factor(Year)
#                )) + geom_line() + geom_smooth() + ggtitle("NWS")

## Actual models:
train_files<- list.files("Summer_results", 
                         pattern = paste0("ORL_training_", model))[these_plots]
eval_files<- list.files("Summer_results", 
                         pattern = paste0("ORL_eval_", model))[these_plots]
# Order = 28035, 36005, 4013, 41067, 6071
eval_NWS<- c(-422.7742, -487.1440, -480.2201, -445.2530, -498.1305)[i]

for(f in train_files){
  df<- read.csv(paste0("Summer_results/",f))[,-1]
  DF<- aggregate(. ~ Model, df, sum)
  r<- ggplot(DF, aes(x=Model,y=Rewards
                     # ,color=as.factor(Year)
             )) + geom_line() + geom_smooth() + ggtitle(f)
  print(r)
}

for(f in eval_files){
  df<- read.csv(paste0("Summer_results/",f))[,-1]
  DF<- aggregate(. ~ Model, df, sum)
  r<- ggplot(DF, aes(x=Model,y=Rewards
                     # ,color=as.factor(Year)
  )) + geom_line() + geom_smooth() + ggtitle(f) +
    geom_hline(yintercept=eval_NWS, 
               color="orange", lwd=1, lty=2) + ylim(-560, -440)
  print(r)
}

########### Getting insight into training:

plot_metric<- function(df, metric, title, meanline=FALSE){
  DF<- data.frame(df[,1], df[,3])
  names(DF)<- c("Epoch", "M")
  if(!meanline){
    p<- ggplot(DF, aes(x=Epoch, y=M)) + geom_line() +
      ylab(metric) + ggtitle(title)
  }else{
    p<- ggplot(DF, aes(x=Epoch, y=M)) + geom_line() +
      ylab(metric) + ggtitle(title) + geom_smooth()
  }
  return(p)
}

folders<- list.files("d3rlpy_logs", pattern = model)[these_plots]

for(f in folders){
  loss<- read.csv(paste0("d3rlpy_logs/",f,"/loss.csv"), header = FALSE)
  print(plot_metric(loss, "Loss", f))
}

## For SAC:
for(f in folders){
  actor_loss<- read.csv(paste0("d3rlpy_logs/",f,"/actor_loss.csv"), header = FALSE)
  print(plot_metric(actor_loss, "Actor Loss", f))
  critic_loss<- read.csv(paste0("d3rlpy_logs/",f,"/critic_loss.csv"), header = FALSE)
  print(plot_metric(critic_loss, "Critic Loss", f))
  temp_loss<- read.csv(paste0("d3rlpy_logs/",f,"/temp_loss.csv"), header = FALSE)
  print(plot_metric(temp_loss, "Temp. Loss", f))
}

for(f in folders){
  Rollout<- read.csv(paste0("d3rlpy_logs/",f,"/rollout_return.csv"), header = FALSE)
  rollout<- Rollout
  rollout$V3<- sapply(Rollout$V3, function(x){
    x2<- gsub("tensor\\(\\[\\[","",x)
    x3<- as.numeric(gsub("\\]\\]\\)","",x2))
  })
  print(plot_metric(rollout, "Rollout Return", f))
}

# for(f in folders){
#   eval<- read.csv(paste0("d3rlpy_logs/",f,"/evaluation.csv"), header = FALSE)
#   plot_metric(loss, "Evaluation")
# }

for(f in folders){
  custom<- read.csv(paste0("d3rlpy_logs/",f,"/custom_metrics.csv"))
  # print(sum(custom$Alert_sum))
  custom$budget_frac<- custom$Alert_sum/custom$Budget
  print(plot_metric(custom[,c("X", "Alert_sum", "budget_frac")], "Fraction of Budget Used", f))
}

for(f in folders){
  custom<- read.csv(paste0("d3rlpy_logs/",f,"/custom_metrics.csv"))
  print(plot_metric(custom[,c("X", "Alert_sum", "Avg_DOS")],
                    "Average Day of Summer", f, meanline=TRUE))
}

for(f in folders){
  custom<- read.csv(paste0("d3rlpy_logs/",f,"/custom_metrics.csv"))
  print(plot_metric(custom[,c("X", "Alert_sum", "Avg_StrkLn")], 
                    "Average Alert Streak Length", f, meanline=TRUE))
  # print(summary(custom$Avg_StrkLn))
}

for(f in folders){
  custom<- read.csv(paste0("d3rlpy_logs/",f,"/custom_metrics.csv"))
  custom$strk_frac<- custom$Avg_StrkLn/custom$Budget
  print(plot_metric(custom[,c("X", "Alert_sum", "strk_frac")], 
                    "Average Alert Streak Length / Budget", f, meanline=TRUE))
}



