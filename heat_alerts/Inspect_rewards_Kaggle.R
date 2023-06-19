
library(ggplot2)
# install.packages("viridis")
library(viridis)
library(cowplot, lib.loc = "~/apps/R_4.2.2")

# Read in data:
load("data/Small_S-A-R_prepped.RData")
DF$weekend<- DF$dowSaturday | DF$dowSunday
DF$alert<- A

Large_S<- DF[,c("alert","HImaxF_PopW", "quant_HI_county", "quant_HI_yest_county",
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

Medium_S<- DF[,c("alert", "quant_HI_county", "quant_HI_yest_county", "quant_HI_3d_county", "HI_mean",
                 "l.Pop_density", "l.Med.HH.Income",
                 "year", "dos", "weekend",
                 "T_since_alert", "alert_sum",
                 "all_hosp_mean_rate", "all_hosp_2wkMA_rate", "all_hosp_3dMA_rate",
                 "Republican", "pm25", "age_65_74_rate", "age_75_84_rate", "dual_rate",
                 "ZoneCold", "ZoneHot.Dry", "ZoneHot.Humid", "ZoneMarine",
                 "ZoneMixed.Dry", "ZoneMixed.Humid", "ZoneVery.Cold")]

## Model NN:
Train.nn<- data.frame(Medium_S)
Train.nn$Y<- R_other_hosps[,1]
Train.nn<- Train.nn[which((Train.nn$quant_HI_county*qhic_sd + qhic_mean) >= 0.9),]

# preds.nn<- read.csv("Summer_results/R_6-19_lr-00063_90pct.csv")
preds.nn<- read.csv("Summer_results/R_6-19_forced_lr-00063_90pct.csv")

DF.nn<- Train.nn
DF.nn$pred_R0<- preds.nn[,"X0"]
DF.nn$pred_R1<- preds.nn[,"X1"]
DF.nn$pred_Y<- sapply(1:nrow(DF.nn), function(i){preds.nn[i,DF.nn[i,"alert"] + 2]})  

summary(DF.nn$pred_R1 - DF.nn$pred_R0)

#############

## Model a:

Train.a<- data.frame(Large_S)
Train.a$alert<- A
Train.a$Y<- R_other_hosps[,1]
Train.a<- Train.a[which((Train.a$quant_HI_county*qhic_sd + qhic_mean) >= 0.9),]

preds.a<- readRDS("Summer_results/Kaggle_preds_a.rds")

DF.a<- Train.a[preds.a$rowIndex,]
DF.a$pred_Y<- preds.a$pred

## Model b:

Train.b<- data.frame(Large_S)
Train.b$alert<- A
Train.b$Y<- R_other_hosps[,1]

preds.b<- readRDS("Summer_results/Kaggle_preds_b-cubist.rds")

DF.b<- Train.b[preds.b$rowIndex,]
DF.b$pred_Y<- preds.b$pred
DF.b<- DF.b[which((DF.b$quant_HI_county*qhic_sd + qhic_mean) >= 0.9),]


#### Summarize accuracy and Make plots:

Data<- DF.nn
N<- nrow(Data)

cor(Data$Y, Data$pred_Y)^2
1 - (sum((Data$Y-Data$pred_Y)^2)/(((N-1)*var(Data$Y))))

set.seed(321)
samp<- sample(1:N, round(0.1*N))

plot(Data[samp,"Y"], Data[samp,"pred_Y"], 
     col = alpha(Data$alert + 1, 0.5), pch=16,
     main = "NOHR")
abline(0,1)

plot_DF<- Data[samp,]

ggplot(plot_DF, 
       aes(x=quant_HI_county, y=pred_Y, col = as.factor(alert), alpha=0.5)) +
  geom_point() + geom_smooth(data=subset(plot_DF, alert == 0), col = "purple")

ggplot(plot_DF, 
       aes(x=quant_HI_county, y=pred_Y, col = all_hosp_mean_rate, alpha=0.5)) +
  geom_point() + scale_color_viridis()

ggplot(plot_DF, 
       aes(x=quant_HI_county, y=pred_Y, col = all_hosp_2wkMA_rate, alpha=0.5)) +
  geom_point() + scale_color_viridis()

ggplot(plot_DF, 
       aes(x=quant_HI_county, y=pred_Y, col = weekend, alpha=0.5)) +
  geom_point() 

ggplot(plot_DF, 
       aes(x=quant_HI_county, y=pred_Y, col = dual_rate, alpha=0.5)) +
  geom_point() + scale_color_viridis()

ggplot(plot_DF, 
       aes(x=quant_HI_county, y=pred_Y, col = l.Pop_density, alpha=0.5)) +
  geom_point() + scale_color_viridis()

ggplot(plot_DF, 
       aes(x=quant_HI_county, y=pred_Y, col = l.Med.HH.Income, alpha=0.5)) +
  geom_point() + scale_color_viridis()

ggplot(plot_DF, 
       aes(x=quant_HI_county, y=pred_Y, col = age_65_74_rate, alpha=0.5)) +
  geom_point() + scale_color_viridis()

ggplot(plot_DF, 
       aes(x=quant_HI_county, y=pred_Y, col = Republican, alpha=0.5)) +
  geom_point() + scale_color_viridis()

