
library(caret)

load("data/Small_S-A-R_prepped.RData")

DF$alert<- A

DF$R<- R_hosps[,1]

myControl<- trainControl(method = "none", savePredictions = "final",
                         verboseIter = TRUE, allowParallel = FALSE)

tgrid<- expand.grid( .mtry = 15, .splitrule = "extratrees", .min.node.size = 1)


## Run model:

s<- Sys.time()

ranger_model<- train(R ~ ., data = DF, method = "ranger",
                     trControl = myControl, tuneGrid = tgrid,
                     importance = "permutation")

e<- Sys.time()
e-s

saveRDS(ranger_model, "Fall_results/Rewards_VarImp.rds")

# preds<- ranger_model$pred$pred
# saveRDS(preds, "Fall_results/Rewards_preds-RF.rds")
# obs<- ranger_model$pred$obs

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
