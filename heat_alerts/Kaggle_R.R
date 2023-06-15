
library(caret, lib.loc = "~/apps/R_4.2.2")
library(foreach, lib.loc = "~/apps/R_4.2.2")
library(iterators, lib.loc = "~/apps/R_4.2.2")
library(doParallel, lib.loc = "~/apps/R_4.2.2")
library(dplyr)

library(ranger, lib.loc = "~/apps/R_4.2.2")
library(RSNNS, lib.loc = "~/apps/R_4.2.2")
# library(xgboost, lib.loc = "~/apps/R_4.2.2")
library(Cubist, lib.loc = "~/apps/R_4.2.2")


# Read in data:
load("data/Small_S-A-R_prepped.RData")
DF$weekend<- DF$dowSaturday | DF$dowSunday

Large_S<- DF[,c("HImaxF_PopW", "quant_HI_county", "quant_HI_yest_county",
                "quant_HI_3d_county", "HI_mean",
                "l.Pop_density", "l.Med.HH.Income",
                "year", "dos", "holiday", "weekend",
                "alert_lag1", "alert_lag2", "alerts_2wks", "T_since_alert", "alert_sum",
                "death_mean_rate", "all_hosp_mean_rate", "heat_hosp_mean_rate",
                "all_hosp_2wkMA_rate", "heat_hosp_2wkMA_rate", "all_hosp_3dMA_rate",     
               "heat_hosp_3dMA_rate", "age_65_74_rate", "age_75_84_rate", "dual_rate",       
                "broadband.usage", "Democrat", "Republican", "pm25",
                "ZoneCold", "ZoneHot.Dry", "ZoneHot.Humid", "ZoneMarine",
                "ZoneMixed.Dry", "ZoneMixed.Humid", "ZoneVery.Cold")]

Medium_S<- DF[,c("quant_HI_county", "quant_HI_yest_county", "quant_HI_3d_county", "HI_mean",
                 "l.Pop_density", "l.Med.HH.Income",
                 "year", "dos", "weekend",
                 "T_since_alert", "alert_sum",
                 "all_hosp_mean_rate", "all_hosp_2wkMA_rate", "all_hosp_3dMA_rate",
                 "Republican", "pm25", "age_65_74_rate", "age_75_84_rate", "dual_rate",
                 "ZoneCold", "ZoneHot.Dry", "ZoneHot.Humid", "ZoneMarine",
                 "ZoneMixed.Dry", "ZoneMixed.Humid", "ZoneVery.Cold")]

Small_S<- DF[,c("quant_HI_county", "HI_mean", "l.Pop_density", "l.Med.HH.Income",
                "year", "dos", "weekend", 
                "T_since_alert", "alert_sum", "all_hosp_mean_rate", 
                "all_hosp_2wkMA_rate", "all_hosp_3dMA_rate")]

#### Tune several models:

# Train<- data.frame(Medium_S)
Train<- data.frame(Large_S)
Train$Y<- R_other_hosps[,1]

N<- 100724 # size of top 90th perc. heat index days
n_cv<- 5
set.seed(321)

algos<- c('ranger', 'cubist', 'mlpWeightDecayML') # 'xgbTree' -- takes a long time, has a lot of tuning params

grids<- list(#expand.grid( .mtry = 7, .splitrule = "variance", .min.node.size = 200 ),
              expand.grid( .mtry = seq(5, 10), .splitrule = "variance",
                           .min.node.size = seq(100, 400, 100) ),
             # expand.grid( .committees = c(5), .neighbors = 0 ),
             expand.grid( .committees = c(3, 5, 7, 10, 20), .neighbors = 0 ),
             # expand.grid(.layer1 = c(100), .layer2 = c(100),
             #             .layer3 = c(100), .decay = c(0.0)),
             expand.grid(.layer1 = c(10,55,100), .layer2 = c(10,55,100),
                         .layer3 = c(10,55,100), .decay = c(0.0, 1e-4)))

# cluster<- makeCluster(detectCores() - 2)
cluster<- makeCluster(20)
registerDoParallel(cluster)
clusterEvalQ(cluster, .libPaths("~/apps/R_4.2.2"))

sink("Summer_results/Kaggle_tuning_6-14_large_b.txt")
for(a in 1:length(algos)){
  
  for(subset in c("all", "pct90")){ 
    
    if(subset == "all"){
      dataset<- sample_n(Train, N)
    }else{
      dataset<- Train[which((Train$quant_HI_county*qhic_sd + qhic_mean) >= 0.9),]
    }
    
    #Set up control object:
    myControl<- trainControl(method = "repeatedcv", number = n_cv, repeats = 3, search = "grid", 
                              index = createFolds(dataset$Y, n_cv),
                              verboseIter = FALSE, allowParallel = TRUE, savePredictions = FALSE)
    
    # PID<- Sys.getpid()
    model_start<- Sys.time()
    model<- caret::train(Y ~ ., data = dataset, method = algos[a], 
                         trControl = myControl, tuneGrid = grids[[a]])
    model_end<- Sys.time()
    print(paste0(algos[a], ", subset = ", subset, ": ", round(model_end - model_start,2), " minutes"))
    print(model$results)
    print(model$bestTune)
    # Mem_peak<- system(paste0('grep VmPeak /proc/', PID, '/status'), intern = TRUE)
    # print(paste("Process", PID, "memory peak =", Mem_peak))
  }
  sink() # remove later
  
}
sink()
stopCluster(cluster)


############### Now take best models from above: writing down best params = R^2

# Ranger, medium S, all data: 6, 400 = 0.068
# Ranger, medium S, 90pct: 8, 400 = 0.075
# Ranger, large S, all data: 10, 400 = 0.068
# Ranger, large S, 90pct: 9, 400 = 0.075

# Cubist, medium S, all data: 10, 0 = 0.067
# Cubist, medium S, 90pct: 10, 0 = 0.077
# Cubist, large S, all data: 3, 0 = 0.068
# Cubist, large S, 90pct: 5, 0 = 0.079

# MLP, medium S, all data: 55, 10, 10, 0 = 0.069
# MLP, medium S, 90pct: 100, 55, 55, 0 = 0.078
# MLP, large S, all data: 100, 100, 100, 0 = 0.068
# MLP, large S, 90pct: 100, 100, 55, 0 = 0.077

n_cv<- 10
set.seed(321)

cluster<- makeCluster(5) # 10?
registerDoParallel(cluster)
clusterEvalQ(cluster, .libPaths("~/apps/R_4.2.2"))

#### Cubist + large S + 90pct:

Train.a<- data.frame(Large_S)
Train.a$Y<- R_other_hosps[,1]
Train.a<- Train.a[which((Train.a$quant_HI_county*qhic_sd + qhic_mean) >= 0.9),]

myControl.a<- trainControl(method = "repeatedcv", number = n_cv, repeats = 1, search = "grid", 
                         index = createFolds(Train.a$Y, n_cv, returnTrain = TRUE),
                         verboseIter = TRUE, allowParallel = TRUE, savePredictions = TRUE)

model.a<- caret::train(Y ~ ., data = Train.a, method = "cubist", 
                     trControl = myControl.a, tuneGrid = 
                       expand.grid(.committees = 5, .neighbors = 0))

preds.a<- model.a$pred[,c("pred", "obs")]
saveRDS(preds.a, "Summer_results/Kaggle_preds_a.rds")

#### MLP + medium S + all:

Train.b<- data.frame(Medium_S)
Train.b$Y<- R_other_hosps[,1]

myControl.b<- trainControl(method = "repeatedcv", number = n_cv, repeats = 3, search = "grid", 
                         index = createFolds(Train.b$Y, n_cv, returnTrain = TRUE),
                         verboseIter = TRUE, allowParallel = TRUE, savePredictions = TRUE)

model.b<- caret::train(Y ~ ., data = Train.b, method = "mlpWeightDecayML", 
                       trControl = myControl.b, tuneGrid = 
                         expand.grid(.layer1 = 55, .layer2 = 10,
                                                 .layer3 = 10, .decay = 0.0))

preds.b<- model.b$pred$pred[which((Train.b$quant_HI_county*qhic_sd + qhic_mean) >= 0.9)]
saveRDS(preds.b, "Summer_results/Kaggle_preds_b.rds")

#### MLP + large S + all:

Train.c<- data.frame(Large_S)
Train.c$Y<- R_other_hosps[,1]

myControl.c<- trainControl(method = "repeatedcv", number = n_cv, repeats = 3, search = "grid", 
                         index = createFolds(Train.c$Y, n_cv, returnTrain = TRUE),
                         verboseIter = TRUE, allowParallel = TRUE, savePredictions = TRUE)

model.c<- caret::train(Y ~ ., data = Train.c, method = "mlpWeightDecayML", 
                       trControl = myControl.c, tuneGrid = 
                         expand.grid(.layer1 = 100, .layer2 = 100,
                                     .layer3 = 100, .decay = 0.0))

preds.c<- model.c$pred$pred[which((Train.c$quant_HI_county*qhic_sd + qhic_mean) >= 0.9)]
saveRDS(preds.c, "Summer_results/Kaggle_preds_c.rds")

#### MLP + large S + 90pct:

Train.d<- data.frame(Large_S)
Train.d$Y<- R_other_hosps[,1]
Train.d<- Train.d[which((Train.d$quant_HI_county*qhic_sd + qhic_mean) >= 0.9),]

myControl.d<- trainControl(method = "repeatedcv", number = n_cv, repeats = 3, search = "grid", 
                         index = createFolds(Train.d$Y, n_cv, returnTrain = TRUE),
                         verboseIter = TRUE, allowParallel = TRUE, savePredictions = TRUE)

model.d<- caret::train(Y ~ ., data = Train.d, method = "mlpWeightDecayML", 
                       trControl = myControl.d, tuneGrid = 
                         expand.grid(.layer1 = 100, .layer2 = 100,
                                     .layer3 = 55, .decay = 0.0))

preds.d<- model.d$pred$pred
saveRDS(preds.d, "Summer_results/Kaggle_preds_d.rds")

stopCluster(cluster)

#### Put it all together:

Results<- data.frame(obs = model.d$pred$obs, preds.a, preds.b, preds.c, preds.d)

saveRDS(Results, "Summer_results/Kaggle_predictions.rds")


