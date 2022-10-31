library(dplyr)
library(ggplot2)
library(caret)
# library(parallel)
# library(doParallel)

load("data/Train-Test.RData")

n_counties<- length(unique(Train$GEOID))
n_years<- 11
n_days<- 153

n_cv<- 2

set.seed(321)
# eda_set<- Train[sample(1:nrow(Train), 0.75*nrow(Train), replace = FALSE),]
eda_set<- Train
Eda_set<- data.frame(scale(eda_set[,vars<- c("HImaxF_PopW", "quant_HI_county", 
                                             "quant_HI_yest_county",
                                             "quant_HI_3d_county", 
                                             "quant_HI_fwd_avg_county",
                                             "Pop_density", "Med.HH.Income",
                                             "year", "dos",
                                             "alert_sum")]), 
                     alert = factor(eda_set$alert),
                     dow = factor(eda_set$dow), 
                     holiday = factor(eda_set$holiday),
                     Zone = factor(eda_set$BA_zone))

levels(Eda_set$alert)<- c("none", "alert") # so caret can predict probabilities

Eda_set$R<- sqrt((eda_set$N / eda_set$Pop.65)*10000)

IND<- createFolds(Eda_set$R, n_cv, returnTrain = TRUE)

myControl<- trainControl(#method = "cv", number = n_cv, 
  index = IND,
  savePredictions = "final",
  verboseIter = TRUE, allowParallel = FALSE) #TRUE

tgrid<- expand.grid( .mtry = 15, .splitrule = "extratrees", .min.node.size = 1)

## Set up parallelization:

# cluster<- makeCluster(10)
# registerDoParallel(cluster)

## Run model:

s<- Sys.time()

ranger_model<- train(R ~ ., data = Eda_set, method = "ranger",
                     trControl = myControl, tuneGrid = tgrid,
                     importance = "permutation")

e<- Sys.time()
e-s
# stopCluster(cluster)

# len<- 1000
# pred_list<- vector(mode = "list", length = len)
# my_seq<- round(seq(1, nrow(Eda_set), length.out = len+1))
# for(i in 2:len){
#   pred_list[[i]]<- predict(ranger_model, Eda_set[my_seq[i-1]:my_seq[i],])
#   print(i)
# }

# preds<- predict(ranger_model, Eda_set)
preds<- ranger_model$pred$pred
saveRDS(preds, "Fall_results/Rewards_preds-RF.rds")
obs<- ranger_model$pred$obs

sink("Fall_results/Rewards_RF.txt")

## R2 = 3.4% on 75% of the training data

## R2 on full training data:
cor(preds, obs)^2

## R2 on original scale:

cor(preds^2, obs^2)^2

sink()

## Check:

preds<- readRDS("Fall_results/Rewards_preds-RF.rds")

## Get correct indices:
inds<- c(IND$Fold2, IND$Fold1)
obs<- Eda_set$R[inds]
cor(preds, obs)^2 # correct

1 - sum((obs - preds)^2)/sum((obs - mean(obs))^2) # 0.024
1 - sum((obs^2 - preds^2)^2)/sum((obs^2 - mean(obs^2))^2) # -0.142

#### Investigate accuracy in more detail:

P<- data.frame(Predictions=preds, Observations=obs)

diffs<- abs(preds-obs)
d<- which(diffs > 1)

ggplot(P[d,], aes(x=Observations, y=Predictions)) + geom_point()


## Characteristics of points with large differences?
  ## Same: alert, alert_sum, 
  ## Substantially harder to predict: higher Pop_density, 
  ## A bit harder to predict: smaller dos, larger MedHHInc, earlier years, Fri + Saturdays, cold zone
  ## A bit easier to predict: Sundays, hot-humid + mixed-humid, higher HImax / quant 3d / fwd_avg

df<- Train[inds,]

summary(df$quant_HI_fwd_avg_county)
summary(df[d,"quant_HI_fwd_avg_county"])

table(df$BA_zone)/nrow(df)
table(df[d,"BA_zone"])/nrow(df[d,])
