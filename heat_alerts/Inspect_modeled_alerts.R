
library(ggplot2)
library(dplyr)
library(viridis)

load("data/Small_S-A-R_prepped.RData")

val<- read.csv("data/Python_val_set.csv")[,2]
Validation<- rep(1,length(A))
Validation[val]<- 2

# ahat<- read.csv("Fall_results/Alerts_model_1-23.csv")[,2]
ahat<- read.csv("Fall_results/Alerts_model_2-21.csv")[,2]

summary(ahat)

# tab<- table(data.frame(Obs=A, Preds=round(ahat)))
tab<- table(data.frame(Obs=A[-val], Preds=ahat[-val] >= 0.01))
tab
# sensitivity:
tab[2,2]/sum(tab[2,])
# specificity:
tab[1,1]/sum(tab[1,])
# PPV:
tab[2,2]/sum(tab[,2])
# NPV:
tab[1,1]/sum(tab[,1])

## Just in validation set:
tab<- table(data.frame(Obs=A[val], Preds=round(ahat[val])))
tab<- table(data.frame(Obs=A[val], Preds=ahat[val] >= 0.01))
tab
# sensitivity:
tab[2,2]/sum(tab[2,])
# specificity:
tab[1,1]/sum(tab[1,])
# PPV:
tab[2,2]/sum(tab[,2])
# NPV:
tab[1,1]/sum(tab[,1])

### Looking at probability constraint:

allow<- ahat >= 0.001
mean(allow) 
Tab<- table(data.frame(Obs=A, Preds=allow))
Tab

Prob_threshold<- c(0.00001, 0.0001, 0.001, 0.01, seq(0.02, 0.1, 0.02))
Allowed_alert_rate<- sapply(Prob_threshold, function(p){mean(ahat >= p)})
Allowed_days<- Allowed_alert_rate*152

data.frame(Prob_threshold, Allowed_alert_rate, Allowed_days)
plot(log10(Prob_threshold), Allowed_days)

data<- read.csv("data/Train_smaller-for-Python.csv")

# budget<- data[which(data$dos == 153), "alert_sum"]
# Budget<- rep(budget, each = n_days)
# data$More_alerts<- Budget - data$alert_sum
Data<- data[-seq(n_days, nrow(data), n_days),]

Data$allow<- allow

tot_allow<- aggregate(allow ~ year + fips, Data, sum)

tot_episode<- aggregate(alert ~ year + fips, Data, sum)


# plot(tot_episode$alert, tot_allow$allow, col = Validation[seq(1, length(A), 152)])
# abline(0,1)

counties<- distinct(Data[,c("fips", "BA_zone")])

plot_DF<- data.frame(Observed = tot_episode$alert, Allowed = tot_allow$allow, 
                     fips = tot_allow$fips)
Plot_DF<- inner_join(plot_DF, counties)
names(Plot_DF)[4]<- "Climate Zone"
Plot_DF$`Climate Zone`<- factor(Plot_DF$`Climate Zone`, 
                                   levels = c("Very Cold", "Cold",
                                              "Mixed-Dry", "Mixed-Humid",
                                              "Marine", "Hot-Dry", "Hot-Humid"))

ggplot(Plot_DF, aes(x=Observed, y=Allowed, col = `Climate Zone`)) + geom_point() +
  ggtitle("Number of Alerts per County-Summer: Model vs Reality") + 
  geom_abline(slope=1, intercept = 0) + scale_color_viridis(discrete = TRUE)

Over<- data.frame(Budget = tot_episode$alert, Allowed = tot_allow$allow)[which(tot_episode$alert > tot_allow$allow),]
Over$Diff<- Over$Budget - Over$Allowed
Over[order(Over$Diff, decreasing=TRUE),]
