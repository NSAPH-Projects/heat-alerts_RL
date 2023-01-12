
load("data/Small_S-A-R_prepped.RData")

ahat<- read.csv("Fall_results/Alerts_model_1-11.csv")[,2]

summary(ahat)

tab<- table(data.frame(Obs=A, Preds=round(ahat)))
# sensitivity:
tab[2,2]/sum(tab[2,])
# specificity:
tab[1,1]/sum(tab[1,])
# PPV:
tab[2,2]/sum(tab[,2])
# NPV:
tab[1,1]/sum(tab[,1])

### Looking at probability constraint:

allow<- ahat >= 0.01
mean(allow) 

data<- read.csv("data/Train_smaller-for-Python.csv")

budget<- data[which(data$dos == 153), "alert_sum"]
Budget<- rep(budget, each = n_days)
data$More_alerts<- Budget - data$alert_sum
Data<- data[-seq(n_days, nrow(data), n_days),]

Data$allow<- allow

tot_allow<- aggregate(allow ~ year + fips, Data, sum)

val<- read.csv("data/Python_val_set.csv")[,2]
Validation<- rep(1,length(A))
Validation[val]<- 2

plot(budget, tot_allow$allow, col = Validation[seq(1, length(A), 152)])
abline(0,1)

Over<- data.frame(Budget = budget, Allowed = tot_allow$allow)[which(budget > tot_allow$allow),]


