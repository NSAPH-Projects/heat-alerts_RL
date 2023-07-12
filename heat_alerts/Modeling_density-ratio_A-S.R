library(caret)

load("data/Small_S-A-R_prepped.RData")

new_pol<- read.csv("Fall_results/DQN_12-16_hosps_constrained_policy.csv")$policy
# new_pol<- read.csv("Fall_results/DQN_12-16_deaths_constrained_policy.csv")$policy

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

## Include alert:
DF$A<- A
new_DF$A<- new_pol

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

myControl<- trainControl(method = "none",
                        classProbs = TRUE, savePredictions = "final",
                         verboseIter = TRUE, allowParallel = FALSE) #TRUE

tgrid<- expand.grid( .mtry = 15, .splitrule = "extratrees", .min.node.size = 1)

## Run model:

s<- Sys.time()

ranger_model<- train(Policy ~ ., data = Train, 
                     method = "ranger", max.depth = 3,
                     trControl = myControl, # importance = "permutation",
                     tuneGrid = tgrid) 

e<- Sys.time()
e-s

saveRDS(ranger_model, "Fall_results/DensRat_12-16_hosps_constrained_75pct.rds")
# saveRDS(ranger_model, "Fall_results/DensRat_12-16_deaths_constrained_75pct.rds")


#### Assess predictions:

# ranger_model<- readRDS("Fall_results/DensRat_11-29_hosps_constrained_75pct.rds")
ranger_model<- readRDS("Fall_results/DensRat_11-29_deaths_constrained_75pct.rds")

## Training:

# tab<- table(data.frame(Obs=ranger_model$pred$obs, Pred=ranger_model$pred$pred))
Pred<- max.col(ranger_model$finalModel$predictions)-1
Obs<- rep(0, nrow(Train))
Obs[which(Train$Policy == "New")]<- 1
tab<- table(data.frame(Obs, Pred))

# sensitivity:
tab[2,2]/sum(tab[2,])
# specificity:
tab[1,1]/sum(tab[1,])
# PPV:
tab[2,2]/sum(tab[,2])
# NPV:
tab[1,1]/sum(tab[,1])


