
library(ggplot2)
library(cowplot, lib.loc = "~/apps/R_4.2.2")

model<- "trpo_0"
model<- "ppo_0"
model<- "qrdqn_0"
model<- "dqn_0"

df<- read.csv(paste0("logs/SB/", model, "/training_metrics/progress.csv"))

my_plot<- function(var, title){
  Y<- df[,var][!is.na(df[,var])]
  DF<- data.frame(Y, Iterations=1:length(Y))
  p<- ggplot(DF, aes(x = Iterations, y = Y)) + ylab(title) +
    geom_line() + 
    geom_smooth(span=0.3) +
      ggtitle(title)
  return(p)
}

emr<- my_plot("eval.mean_reward", "Eval Mean Reward")
tr<- my_plot("custom.training_rewards", "Training Rewards")
a_freq<- my_plot("custom.alerts_freq", "Alert Frequency")
over_b<- my_plot( "custom.over_budget_freq", "Over Budget Frequency")
a_t<- my_plot( "custom.average_t_alerts", "Avg. Day of Summer of Alerts")
std_a_t<- my_plot( "custom.stdev_t_alerts", "StDev of Day of Summer of Alerts")
strk_len<- my_plot("custom.average_streak", "Avg. Streak Length")
std_strk<- my_plot("custom.stdev_streak", "Alert Streak StDev")

plot_grid(emr, tr, a_freq, over_b, nrow=2)
plot_grid(a_t, std_a_t, strk_len, std_strk, nrow=2)


plot_grid(emr, tr)
plot_grid(a_freq, over_b)
plot_grid(a_t, std_a_t)
plot_grid(strk_len, std_strk)




