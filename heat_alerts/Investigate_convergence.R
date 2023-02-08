library(dplyr)
library(ggplot2)
library(cowplot)

#### Pytorch convergence in terms of loss:

# DQN<- read.csv("Fall_results/DQN_10-18_epoch-losses.csv")
# DQN<- read.csv("lightning_logs/constr_deaths_adam_huber/version_4/metrics.csv")
DQN<- read.csv("lightning_logs/feb_constr_hosps_adam_huber_REs/version_3/metrics.csv")
# DQN_a<- read.csv("lightning_logs/constr_hosps_adam_huber_REs/version_0/metrics.csv")
# DQN_b<- read.csv("lightning_logs/constr_hosps_adam_huber_REs/version_1/metrics.csv")
# DQN_b$epoch<- 1:nrow(DQN_b) + 999
# DQN<- rbind(DQN_a, DQN_b)
# DQN<- read.csv("lightning_logs/constr_all-hosps_adam_huber/version_0/metrics.csv")

ggplot(DQN, aes(x=epoch, y=log(epoch_loss))) + geom_line() + 
  xlab("Epochs") + ylab("Log of Huber Loss") + ggtitle("DQN Model")

ggplot(DQN, aes(x=epoch, y=Q1.Q0)) + geom_line() + 
  xlab("Epochs") + ylab("Number of Days Q1 > Q0") + ggtitle("DQN Model")

ggplot(DQN, aes(x=epoch, y=n_alerts)) + geom_line() + 
  xlab("Epochs") + ylab("Number of Alerts Issued") + ggtitle("DQN Model")

ggplot(DQN, aes(x=epoch, y=log(n_alerts/Q1.Q0))) + geom_line() + 
  xlab("Epochs") + ylab("Log of {Number of Alerts Issued / Number of Days Q1 > Q0}") + ggtitle("DQN Model")

### Validation vs Training, Rewards and Alerts Models

# DF<- read.csv("lightning_logs/R_no-past_deaths_adam_mse/version_0/metrics.csv")
DF<- read.csv("lightning_logs/feb_R_hosps_adam_mse/version_0/metrics.csv")
DF<- read.csv("lightning_logs/feb_R_other-hosps_adam_mse/version_0/metrics.csv")

Val_Loss<- DF[seq(1,nrow(DF),2),1]
Train_Loss<- DF[seq(2,nrow(DF),2),5]
Epoch<- 1:length(Val_Loss)

# DF<- read.csv("lightning_logs/test_NN_alerts/version_11/metrics.csv")
# 
# Val_Loss<- DF[seq(1,nrow(DF),2),1]
# Train_Loss<- DF[seq(2,nrow(DF),2),4]
# Epoch<- 1:length(Val_Loss)


Plot_df<- data.frame(Epoch, Train_Loss, Val_Loss)[-1,]

ggplot(Plot_df[1:nrow(Plot_df),], aes(x=Epoch)) + geom_line(aes(y=Train_Loss)) + 
  geom_line(aes(y=Val_Loss), col = "blue") +
  ylab("Loss") + ggtitle("Model Convergence")

#### Torch lm:

Coefs1<- readRDS("Aug_results/Q-coefficients_9-7.rds")
Coefs2<- readRDS("Aug_results/Q-coefficients_9-8.rds")
Coefs<- rbind(Coefs1, Coefs2)

## Looking at coefs all together:

c0<- Coefs[1:(nrow(Coefs)-1),]
c1<- Coefs[2:nrow(Coefs),]

res<- apply(cbind(c0, c1), MARGIN = 1, function(x) sqrt(mean((x[49:96] - x[1:48])^2)))

plot(1:length(res), res)

## Now looking at the individual coefs:

cols<- readRDS("data/Model_colnames_9-8.rds")
colnames(Coefs)<- cols

par(mfrow=c(3,4))
for(j in 1:ncol(Coefs)){
  plot(1:nrow(Coefs), Coefs[,j], main = cols[j])
}

## Looking RMSE of Q-values over time:

load("data/Train-Test.RData")

n_counties<- length(unique(Train$GEOID))
n_years<- 11
n_days<- 153

States<- Train[, c("HImaxF_PopW", "quant_HI_county", "quant_HI_yest_county",
                   "quant_HI_3d_county", "quant_HI_fwd_avg_county",
                   "BA_zone", "Pop_density", "Med.HH.Income",
                   "year", "dos", "holiday", # "Holiday", 
                   "dow", "alert_sum")] # same variables as in model for alerts

States<- States[-seq(n_days, nrow(Train), n_days),]
S<- States %>% mutate_if(is.numeric, scale)

S_full_0<- model.matrix(~ A*., data.frame(A=0,S))
S_full_1<- model.matrix(~ A*., data.frame(A=1,S))

rm("Train")
rm("Test")
rm("Coefs1")
rm("Coefs2")
rm("States")
gc()

df0<- S_full_0 %*% Coefs[nrow(Coefs),]
df1<- S_full_1 %*% Coefs[nrow(Coefs),]

## Just look at a subsequence:
ss<- seq(1, nrow(Coefs), 20)

DF0<- S_full_0 %*% t(Coefs[ss,])
DF1<- S_full_1 %*% t(Coefs[ss,])

q_diff_0<- rep(0,length(ss))
q_diff_1<- rep(0,length(ss))

for(j in 2:length(ss)){
  q_diff_0[j]<- sqrt(mean((DF0[,j] - DF0[,j-1])^2))
  q_diff_1[j]<- sqrt(mean((DF1[,j] - DF1[,j-1])^2))
}

plot(1:length(ss), q_diff_0)
plot(1:length(ss), q_diff_1) # does actually look like we've converged?


##########################################################

## Read in text file:
res<- read.table("Aug_results/First_SGD.txt", fill=TRUE)

## Spread in coefficients:
diff_coefs<- res[seq(2,nrow(res),28),2]
plot(1:999, diff_coefs, type = "l")

## Value of intervention, all else equal:
A_coef<- res[seq(6,nrow(res), 28),2]
plot(1:999, A_coef)

## Effect of alert_sum:
alert_sum<- res[seq(18,nrow(res), 28),1]
plot(1:999, alert_sum)

## Effect of Med HH Income:
med_HH<- res[seq(14,nrow(res), 28),3]
plot(1:999, med_HH)
