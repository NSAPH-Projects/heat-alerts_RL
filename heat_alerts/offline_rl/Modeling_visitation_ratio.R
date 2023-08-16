library(caret)

## Read in data and make dummy vars:

# n_days<- 153
# 
# data<- read.csv("data/Train_smaller-for-Python.csv")
# 
# budget<- data[which(data$dos == 153), "alert_sum"]
# Budget<- rep(budget, each = n_days)
# data$More_alerts<- Budget - data$alert_sum
# Data<- data[-seq(n_days, nrow(data), n_days),]
# 
# DF<- data.frame(scale(Data[,vars<- c("HImaxF_PopW", "quant_HI_county",
#                                      "quant_HI_yest_county",
#                                      "quant_HI_3d_county",
#                                      "quant_HI_fwd_avg_county",
#                                      "l.Pop_density", "l.Med.HH.Income",
#                                      "year", "dos",
#                                      "alert_sum", "More_alerts")]),
#                 # alert = Data$alert,
#                 dow = Data$dow,
#                 holiday = Data$holiday,
#                 Zone = Data$BA_zone)
# 
# dummy_vars<- with(DF, data.frame(model.matrix(~dow+0), model.matrix(~Zone+0)))
# DF<- DF[,-which(names(DF) %in% c("dow", "Zone"))]
# DF<- data.frame(DF, dummy_vars)
# 
# A<- Data$alert
# 
# R_hosps<- -1000*(Data["all_hosps"]/Data["total_count"])
# R_deaths<- -1000*(Data["N"]/Data["Pop.65"])
# 
# rm(list= ls()[!(ls() %in% c('DF','n_days', 'A',
#                             'R_hosps', 'R_deaths', 'budget'))])
# save.image("data/Small_S-A-R_prepped.RData")

load("data/Small_S-A-R_prepped.RData")

## Create new_DF:
# new_pol<- read.csv("Fall_results/DQN_11-5_hosps_constrained_policy.csv")$policy
new_pol<- read.csv("Fall_results/DQN_11-5_deaths_constrained_policy.csv")$policy

new_alert_sum<- rep(0,nrow(DF))
for(i in which(DF$dos == min(DF$dos))){
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
train_inds<- sample(1:nrow(DF), 0.75*nrow(DF), replace = FALSE)

x.train<- rbind(DF[train_inds,], new_DF[train_inds,])
y.train<- c(rep("Behavior", length(train_inds)), rep("New", length(train_inds)))
levels(y.train)<- c("Behavior", "New") # so caret can predict probabilities

test_inds<- setdiff(1:nrow(DF), train_inds)

x.test<- rbind(DF[test_inds,], new_DF[test_inds,])
y.test<- c(rep("Behavior", length(test_inds)), rep("New", length(test_inds)))
levels(y.test)<- c("Behavior", "New")

Train<- data.frame(Policy=y.train, x.train)
row.names(Train)<- NULL
Test<- data.frame(Policy=y.test, x.test)
row.names(Test)<- NULL

## Train model:

myControl<- trainControl(classProbs = TRUE, savePredictions = "final",
  verboseIter = TRUE, allowParallel = FALSE) #TRUE

tgrid<- expand.grid( .mtry = 15, .splitrule = "extratrees", .min.node.size = 1)

## Run model:

s<- Sys.time()

ranger_model<- train(Policy ~ ., data = Train, 
                     method = "ranger",
                     trControl = myControl, # importance = "permutation",
                     tuneGrid = tgrid) 

e<- Sys.time()
e-s

saveRDS(ranger_model, "Fall_results/VisRat_11-8_deaths_constrained_75pct.rds")


#### Assess predictions:

ranger_model<- readRDS("Fall_results/VisRat_11-8_hosps_constrained_75pct.rds")

## Training:

tab<- table(data.frame(Obs=ranger_model$pred$obs, Pred=ranger_model$pred$pred))
# sensitivity:
tab[2,2]/sum(tab[2,])
# specificity:
tab[1,1]/sum(tab[1,])
# PPV:
tab[2,2]/sum(tab[,2])
# NPV:
tab[1,1]/sum(tab[,1])


## Testing:

preds<- predict(ranger_model, x.test) # to get probabilities (for OPE), use type = "prob"

tab<- table(data.frame(Obs=y.test, Pred=preds))
# sensitivity:
tab[2,2]/sum(tab[2,])
# specificity:
tab[1,1]/sum(tab[1,])
# PPV:
tab[2,2]/sum(tab[,2])
# NPV:
tab[1,1]/sum(tab[,1])


