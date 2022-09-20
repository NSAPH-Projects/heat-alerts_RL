library(dplyr)
library(caret)
# library(parallel)
# library(doParallel)

load("data/Train-Test.RData")

n_counties<- length(unique(Train$GEOID))
n_years<- 11
n_days<- 153

n_cv<- 5

set.seed(321)
eda_set<- Train[sample(1:nrow(Train), 0.5*nrow(Train), replace = FALSE),]
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

IND<- createFolds(Eda_set$alert, n_cv, returnTrain = TRUE)

myControl<- trainControl(#method = "cv", number = n_cv, 
                          classProbs = TRUE,
                          search = "grid", index = IND,
                          savePredictions = "final",
                          verboseIter = TRUE, allowParallel = FALSE) #TRUE

tgrid<- expand.grid( .mtry = 15, .splitrule = "extratrees", .min.node.size = 1)

## Set up parallelization:

# cluster<- makeCluster(10)
# registerDoParallel(cluster)

## Run model:

s<- Sys.time()

ranger_model<- train(alert ~ ., data = Eda_set, method = "ranger",
                      trControl = myControl, importance = "permutation",
                     tuneGrid = tgrid) 

e<- Sys.time()
e-s
# stopCluster(cluster)

saveRDS(ranger_model, "Aug_results/a_RF_9-5_50pct.rds")


#### Assess predictions:

ranger_model<- readRDS("Aug_results/a_RF_9-5_50pct.rds")

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

## Validation: (on the rest of the training data)

valid<- Train[which(! as.numeric(row.names(Train)) %in% as.numeric(row.names(eda_set))),]
Valid<- data.frame(scale(valid[,vars<- c("HImaxF_PopW", "quant_HI_county", 
                                           "quant_HI_yest_county",
                                           "quant_HI_3d_county", 
                                           "quant_HI_fwd_avg_county",
                                           "Pop_density", "Med.HH.Income",
                                           "year", "dos",
                                           "alert_sum")]), 
                   alert = factor(valid$alert),
                   dow = factor(valid$dow), 
                   holiday = factor(valid$holiday),
                   Zone = factor(valid$BA_zone))
levels(Valid$alert)<- c("none", "alert")

preds<- predict(ranger_model, Valid) # to get probabilities (for OPE), use type = "prob"

tab<- table(data.frame(Obs=Valid$alert, Pred=preds))
# sensitivity:
tab[2,2]/sum(tab[2,])
# specificity:
tab[1,1]/sum(tab[1,])
# PPV:
tab[2,2]/sum(tab[,2])
# NPV:
tab[1,1]/sum(tab[,1])

## Testing: (on the test data set)

Testing<- data.frame(scale(Test[,vars<- c("HImaxF_PopW", "quant_HI_county", 
                                         "quant_HI_yest_county",
                                         "quant_HI_3d_county", 
                                         "quant_HI_fwd_avg_county",
                                         "Pop_density", "Med.HH.Income",
                                         "year", "dos",
                                         "alert_sum")]), 
                   alert = factor(Test$alert),
                   dow = factor(Test$dow), 
                   holiday = factor(Test$holiday),
                   Zone = factor(Test$BA_zone))
levels(Testing$alert)<- c("none", "alert")

preds<- predict(ranger_model, Testing) # to get probabilities (for OPE), use type = "prob"

tab<- table(data.frame(Obs=Testing$alert, Pred=preds))
# sensitivity:
tab[2,2]/sum(tab[2,])
# specificity:
tab[1,1]/sum(tab[1,])
# PPV:
tab[2,2]/sum(tab[,2])
# NPV:
tab[1,1]/sum(tab[,1])

