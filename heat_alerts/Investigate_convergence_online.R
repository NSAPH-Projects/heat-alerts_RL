
library(dplyr)
library(ggplot2)
library(cowplot, lib.loc = "~/apps/R_4.2.2")

model<- "test_4"

plot_metric<- function(df, metric){
  DF<- data.frame(df$V1, df$V3)
  names(DF)<- c("Epoch", "M")
  p<- ggplot(DF, aes(x=Epoch, y=M)) + geom_line() +
    ylab(metric)
  return(p)
}

# eval<- read.csv(paste0("d3rlpy_logs/",model,"/evaluation.csv"), header = FALSE)
rollout<- read.csv(paste0("d3rlpy_logs/",model,"/rollout_return.csv"), header = FALSE)
loss<- read.csv(paste0("d3rlpy_logs/",model,"/loss.csv"), header = FALSE)


plot_metric(loss, "Loss")
plot_metric(rollout, "Rollout Return")
# plot_metric(eval, "Evaluation")

