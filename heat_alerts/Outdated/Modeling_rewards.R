
library(stringi, lib.loc = "~/apps/R_4.2.2")
library(caret, lib.loc = "~/apps/R_4.2.2")

load("data/Small_S-A-R_prepped.RData")

DF$alert<- A

# DF$R<- R_deaths[,1]
# DF$R<- R_all_hosps[,1]
# DF$R<- R_heat_hosps[,1]
DF$R<- - R_other_hosps[,1]*1000

myControl<- trainControl(method = "none", savePredictions = "final",
                         verboseIter = TRUE, allowParallel = FALSE)

tgrid<- expand.grid( .mtry = 15, .splitrule = "extratrees", 
                     .min.node.size = 5)


## Run model:

# outcomes<- c("deaths", "all_hosps", "heat_hosps", "other_hosps")
# outcomes<- c("deaths", "other_hosps")
# 
# i<- 2

sink("Fall_results/Rewards_VarImp_3-13-23.txt")

# for(o in list(R_deaths, R_all_hosps, R_heat_hosps, R_other_hosps)){
# for(o in list(R_deaths, R_other_hosps)){
  
  # DF$R<- o[,1]
  
  s<- Sys.time()
  
  ranger_model<- train(R ~ ., data = DF, method = "ranger",
                       trControl = myControl, tuneGrid = tgrid,
                       importance = "permutation"
  )
  
  e<- Sys.time()
  e-s
  
  # saveRDS(ranger_model, paste0("Fall_results/Rewards_", outcomes[i], ".rds"))
  # print(outcomes[i])
  # print(varImp(ranger_model))
  
  # i<- i+1
  
# }

VI<- varImp(ranger_model)
VI_df<- data.frame(Variable=row.names(VI$importance), Importance=VI$importance$Overall)
VI_df[order(VI_df$Importance, decreasing = TRUE),]

sink()

## Make marginal plots

ranger_model<- readRDS("Fall_results/Rewards_deaths.rds")

n_pts<- 10

marg_template<- matrix(rep(colMeans(DF[,1:15]), each=n_pts*2), nrow = n_pts*2)
marg_template<- data.frame(marg_template, 
                           alert_lag1 = 0, alert_lag2 = 0, holiday = 0,
                           dowFriday = 0, dowMonday = 0, dowSaturday = 0,
                           dowSunday = 0, dowThursday = 1, dowTuesday = 0, 
                           dowWednesday = 0, ZoneCold = 0, ZoneHot.Dry = 0, 
                           ZoneHot.Humid = 0, ZoneMarine = 0, ZoneMixed.Dry = 0,
                           ZoneMixed.Humid = 0, ZoneVery.Cold = 0, 
                           alert = c(rep(0,n_pts), rep(1,n_pts)))
marg_template$holiday<- as.factor(marg_template$holiday)
names(marg_template)<- names(DF)

zones<- names(DF)[26:32]

for(z in zones){ # climate zones
  print(z)
  this_marg<- marg_template
  this_marg[,z]<- 1
  this_marg$HImaxF_PopW = rep(seq(0.5, 5.2, length.out = n_pts), 2) # 90 to 145 F
  this_marg$R<- predict(ranger_model, this_marg)
  
  this_marg$alert<- as.factor(this_marg$alert)
  this_marg$HImaxF_PopW<- this_marg$HImaxF_PopW*11.71149 + 83.97772 # from S, Train_smaller-for-Python.csv
  ggplot(this_marg, aes(x = HImaxF_PopW, y = R, color = alert)) + 
    geom_line() + ggtitle(z)
}




##############################

preds<- ranger_model$finalModel$predictions
# obs<- DF$R
obs<- R_heat_hosps[,1]

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
