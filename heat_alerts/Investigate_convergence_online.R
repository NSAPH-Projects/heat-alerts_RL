
library(dplyr)
library(ggplot2)
library(cowplot, lib.loc = "~/apps/R_4.2.2")
library(stringr)

model<- "Online-0_DoubleDQN"
model<- "Online-0_SAC"

i<- 1
these_plots<- seq(i, 60, 5)

########### Final evaluations:

## Benchmarks:
df<- read.csv("Summer_results/ORL_eval_NWS.csv")[,-1]
df<- read.csv("Summer_results/ORL_eval_zero.csv")[,-1]

DF<- aggregate(. ~ County # + Year
               , df, sum)
fips<- 4013
fips<- 36061
DF[which(DF$County == fips),]

# ggplot(DF, aes(x=County,y=Rewards
#                    # ,color=as.factor(Year)
#                )) + geom_line() + geom_smooth() + ggtitle("NWS")

## Actual models:
files<- list.files("Summer_results", pattern = model)
for(f in files){
  df<- read.csv(paste0("Summer_results/",f))[,-1]
  DF<- aggregate(. ~ Model, df, sum)
  r<- ggplot(DF, aes(x=Model,y=Rewards
                     # ,color=as.factor(Year)
             )) + geom_line() + geom_smooth() + ggtitle(f)
  print(r)
}

########### Getting insight into training:

plot_metric<- function(df, metric, title){
  DF<- data.frame(df[,1], df[,3])
  names(DF)<- c("Epoch", "M")
  p<- ggplot(DF, aes(x=Epoch, y=M)) + geom_line() +
    ylab(metric) + ggtitle(title)
  return(p)
}

folders<- list.files("d3rlpy_logs", pattern = model)

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

for(f in folders[these_plots]){
  custom<- read.csv(paste0("d3rlpy_logs/",f,"/custom_metrics.csv"))
  custom$budget_frac<- custom$Alert_sum/custom$Budget
  print(plot_metric(custom[,c("X", "Alert_sum", "budget_frac")], "Fraction of Budget Used", f))
}

for(f in folders[these_plots]){
  custom<- read.csv(paste0("d3rlpy_logs/",f,"/custom_metrics.csv"))
  print(plot_metric(custom[,c("X", "Alert_sum", "Avg_DOS")], "Average Day of Summer", f))
}

for(f in folders[these_plots]){
  custom<- read.csv(paste0("d3rlpy_logs/",f,"/custom_metrics.csv"))
  print(plot_metric(custom[,c("X", "Alert_sum", "Avg_StrkLn")], "Average Alert Streak Length", f))
}


