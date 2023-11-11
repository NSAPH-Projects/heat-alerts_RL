library(BART)
library(profvis)
library(ggplot2)
library(parallel)
detectCores()

## Read in data and make dummy vars:

n_days<- 153

data<- read.csv("data/Train_smaller-for-Python.csv")

budget<- data[which(data$dos == 153), "alert_sum"]
Budget<- rep(budget, each = n_days)
data$More_alerts<- Budget - data$alert_sum
Data<- data[-seq(n_days, nrow(data), n_days),]

DF<- data.frame(scale(Data[,vars<- c("HImaxF_PopW", "quant_HI_county", 
                                     "quant_HI_yest_county",
                                     "quant_HI_3d_county", 
                                     "quant_HI_fwd_avg_county",
                                     "l.Pop_density", "l.Med.HH.Income",
                                     "year", "dos",
                                     "alert_sum", "More_alerts")]), 
                # alert = Data$alert,
                dow = Data$dow, 
                holiday = Data$holiday,
                Zone = Data$BA_zone)

dummy_vars<- with(DF, data.frame(model.matrix(~dow+0), model.matrix(~Zone+0)))
DF<- DF[,-which(names(DF) %in% c("dow", "Zone"))]
DF<- data.frame(DF, dummy_vars)

## Create new_DF:
new_pol<- read.csv("Fall_results/DQN_11-5_hosps_constrained_policy.csv")$policy

new_alert_sum<- rep(0,nrow(Data))
for(i in which(Data$dos == 1)){
  new_alert_sum[i:(i+n_days-2)]<- cumsum(new_pol[i:(i+n_days-2)])
}

S_Budget<- rep(budget, each = n_days-1)
new_More_alerts<- S_Budget - new_alert_sum

new_DF<- DF
new_DF["alert_sum"]<- new_alert_sum
new_DF["More_alerts"]<- new_More_alerts
new_DF[,c("alert_sum", "More_alerts")]<- scale(new_DF[,c("alert_sum", "More_alerts")])


## Set up train and "test" sets:

set.seed(321)
train_inds<- sample(1:nrow(Data), 0.75*nrow(Data), replace = FALSE)

x.train<- rbind(DF[train_inds,], new_DF[train_inds,])
y.train<- c(rep(0, length(train_inds)), rep(1, length(train_inds)))

test_inds<- setdiff(1:nrow(Data), train_inds)

x.test<- rbind(DF[test_inds,], new_DF[test_inds,])
y.test<- c(rep(0, length(test_inds)), rep(1, length(test_inds)))


## Set up BART:

start<- Sys.time()
post<- mc.pbart(x.train, y.train, printevery = 10, ndpost = 100,
                nskip=5000, keepevery = 1000, mc.cores = 5, seed = 321)

p<- list(treedraws = post$treedraws, binaryOffset = post$binaryOffset)
class(p)<- "pbart"
saveRDS(p, "Fall_results/VisRat_BART-model_11-5_hosps_constrained.rds")

end<- Sys.time()
end - start


## Getting predictions later:
p<- readRDS("Fall_results/VisRat_BART-model_11-5_hosps_constrained.rds")

a1_train<- which(y.train == 1)
a1_test<- which(y.test == 1)

preds.a1_train<- predict(p, x.train[a1_train,])$prob.test
preds.a1_test<- predict(p, x.test[a1_test,])$prob.test

a0_train<- which(y.train == 0)
a0_test<- which(y.test == 0)

preds.a0_train<- predict(p, x.train[a0_train,])$prob.test
preds.a0_test<- predict(p, x.test[a0_test,])$prob.test






