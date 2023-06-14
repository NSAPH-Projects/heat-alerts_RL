
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

Train<- data.frame(Medium_S)
Train$Y<- R_other_hosps[,1]

N<- 10000
n_cv<- 5
set.seed(321)

algos<- c('ranger', 'cubist', 'mlpWeightDecayML') # 'xgbTree' -- takes a long time, has a lot of tuning params

grids<- list(expand.grid( .mtry = 7, .splitrule = "variance", .min.node.size = 200 ),
              # expand.grid( .mtry = seq(5, 10), 
              #             .splitrule = "variance", .min.node.size = seq(10, 210, 50) ),
             expand.grid( .committees = c(5), .neighbors = 0 ),
             # expand.grid( .committees = c(5, 10, 20), .neighbors = 0 ),
             # expand.grid(.layer1 = c(10,100), .layer2 = c(10,100),
             #             .layer3 = c(10,100), .decay = c(0.0, 1e-4)),
             expand.grid(.layer1 = c(100), .layer2 = c(100),
                         .layer3 = c(100), .decay = c(0.0)))

cluster<- makeCluster(detectCores() - 2) # cluster<- makeCluster(20)
registerDoParallel(cluster)
clusterEvalQ(cluster, .libPaths("~/apps/R_4.2.2"))

sink("Summer_results/Kaggle_preliminaries.txt")
for(a in 1:length(algos)){
  
  for(subset in c("all", "pct90")){ 
    
    
    dataset<- sample_n(Train, N)
    
    #Set up control object:
    myControl<- trainControl(method = "repeatedcv", number = n_cv, repeats = 3, search = "grid", 
                              index = createFolds(dataset$Y, n_cv),
                              verboseIter = FALSE, allowParallel = TRUE, savePredictions = FALSE)
    
    PID<- Sys.getpid()
    model_start<- Sys.time()
    model<- caret::train(Y ~ ., data = dataset, method = algos[a], 
                         trControl = myControl)
                         # , tuneGrid = grids[[a]])
    model_end<- Sys.time()
    print(model$results)
    print(model$bestTune)
    Mem_peak<- system(paste0('grep VmPeak /proc/', PID, '/status'), intern = TRUE)
    print(paste0(algos[a], ": ", model_end - model_start, " minutes"))
    print(paste("Process", PID, "memory peak =", Mem_peak))
  }
  
}
sink()
stopCluster(cluster)
