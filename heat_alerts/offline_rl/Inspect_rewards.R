
library(ggplot2)
# install.packages("viridis")
library(viridis)
library(cowplot)

## Average effect of heat alerts vs health history:

ests<- read.csv("Fall_results/All-Hosp_preds_vs_History.csv")
Ests<- data.frame(t(ests)[-1,])
names(Ests)<- c("Health History", "Average Effect", "Average Effect when A=1")

ggplot(Ests, aes(x = `Health History`, y = `Average Effect`)) + geom_line() +
  geom_line(aes(y = `Average Effect when A=1`), col = "green")

## Compare to actual / covariates:

load("data/Small_S-A-R_prepped.RData")
DF$alert<- A
Data<- DF

## 1-12 was MSE with softplus, 1-13 was MSE with relu

pred_deaths<- read.csv("Fall_results/R_1-23_deaths.csv")#*1000
Pred_deaths<- sapply(1:length(A), function(i) pred_deaths[i,A[i]+2])
# Pred_deaths[which(Pred_deaths > 0)]<- 0
pred_OH<- read.csv("Fall_results/R_3-3_other-hosps_all.csv")#*1000
pred_OH<- pred_OH[-seq(153, nrow(pred_OH), 153),]
Pred_OH<- sapply(1:length(A), function(i) pred_OH[i,A[i]+2])
pred_hosps<- read.csv("Fall_results/R_3-3_all-hosps_all.csv")#*1000
pred_hosps<- pred_hosps[-seq(153, nrow(pred_hosps), 153),]
Pred_hosps<- sapply(1:length(A), function(i) pred_hosps[i,A[i]+2])

#### Summary stats:

summary(R_deaths[,1])
summary(Pred_deaths)
summary(R_all_hosps[,1])
summary(Pred_hosps)
summary(R_other_hosps[,1])
summary(Pred_OH)

cor(R_deaths[,1], Pred_deaths)^2
cor(R_all_hosps[,1], Pred_hosps)^2
cor(R_other_hosps[,1], Pred_OH)^2

sqrt(mean((R_deaths[,1] - Pred_deaths)^2)) / sd(R_deaths[,1])
sqrt(mean((R_all_hosps[,1] - Pred_hosps)^2)) / sd(R_all_hosps[,1])
sqrt(mean((R_other_hosps[,1] - Pred_OH)^2)) / sd(R_other_hosps[,1])

mean((R_deaths[,1] - Pred_deaths)^2) / sd(R_deaths[,1])
mean((R_all_hosps[,1] - Pred_hosps)^2) / sd(R_all_hosps[,1])
mean((R_other_hosps[,1] - Pred_OH)^2) / sd(R_other_hosps[,1])

## Comparing effect of alerts:

summary(pred_deaths$X1 - pred_deaths$X0)
summary(pred_hosps$X1 - pred_hosps$X0)
summary(pred_OH$X1 - pred_OH$X0)

summary(pred_deaths$X1[A==1] - pred_deaths$X0[A==1])
summary(pred_hosps$X1[A==1] - pred_hosps$X0[A==1])
summary(pred_OH$X1[A==1] - pred_OH$X0[A==1])

# summary(pred_deaths$X1[A==1] - pred_deaths$X0[A==1])
# summary(pred_hosps$X1[A==1] - pred_hosps$X0[A==1])
# summary(pred_OH$X1[A==1] - pred_OH$X0[A==1])

#### Plots:

set.seed(321)
samp<- sample(1:length(A), round(0.05*length(A)))

plot(R_deaths[samp,1], Pred_deaths[samp],  main = "Deaths",
     col = alpha(A+1, 0.5), pch=16)
abline(0,1)

plot(R_all_hosps[samp,1], Pred_hosps[samp], col = alpha(A+1, 0.5), pch=16, 
     main = "All Hospitalizations")
abline(0,1)

plot(R_other_hosps[samp,1], Pred_OH[samp], col = alpha(A+1, 0.5), pch=16,
     main = "Other Hospitalizations")
abline(0,1)


## With covariates:

plot_DF<- data.frame(Data, Pred_deaths)[samp,]
ggplot(plot_DF, 
       aes(x=quant_HI_county, y=Pred_deaths, col = alert, alpha=0.5)) +
  geom_point() + geom_smooth(data=subset(plot_DF, alert == 0), col = "red") +
  scale_color_viridis()

ggplot(data.frame(Data, Pred_deaths)[samp,], 
       aes(x=quant_HI_county, y=Pred_deaths, col = quant_HI_yest_county, alpha=0.5)) +
  geom_point() + scale_color_viridis()

ggplot(data.frame(Data, Pred_deaths)[samp,], 
       aes(x=quant_HI_county, y=Pred_deaths, col = quant_HI_3d_county, alpha=0.5)) +
  geom_point() + scale_color_viridis()

ggplot(data.frame(Data, Pred_deaths)[samp,], 
       aes(x=quant_HI_county, y=Pred_deaths, col = dos, alpha=0.5)) +
  geom_point() + scale_color_viridis()

ggplot(data.frame(Data, Pred_deaths)[samp,], 
       aes(x=quant_HI_county, y=Pred_deaths, col = as.numeric(holiday), alpha=0.5)) +
  geom_point() + scale_color_viridis()

ggplot(data.frame(Data, Pred_deaths)[samp,], 
       aes(x=quant_HI_county, y=Pred_deaths, col = l.Pop_density, alpha=0.5)) +
  geom_point() + scale_color_viridis()

ggplot(data.frame(Data, Pred_deaths)[samp,], 
       aes(x=quant_HI_county, y=Pred_deaths, col = l.Med.HH.Income, alpha=0.5)) +
  geom_point() + scale_color_viridis()

ggplot(data.frame(Data, Pred_deaths)[samp,], 
       aes(x=quant_HI_county, y=Pred_deaths, col = broadband.usage, alpha=0.5)) +
  geom_point() + scale_color_viridis()

ggplot(data.frame(Data, Pred_deaths)[samp,], 
       aes(x=quant_HI_county, y=Pred_deaths, col = Democrat, alpha=0.5)) +
  geom_point() + scale_color_viridis()

ggplot(data.frame(Data, Pred_deaths)[samp,], 
       aes(x=quant_HI_county, y=Pred_deaths, col = Republican, alpha=0.5)) +
  geom_point() + scale_color_viridis()

ggplot(data.frame(Data, Pred_deaths)[samp,], 
       aes(x=quant_HI_county, y=Pred_deaths, col = death_mean_rate, alpha=0.5)) +
  geom_point() + scale_color_viridis()

ggplot(data.frame(Data, Pred_deaths)[samp,], 
       aes(x=quant_HI_county, y=Pred_deaths, col = all_hosp_mean_rate, alpha=0.5)) +
  geom_point() + scale_color_viridis()


plot_DF<- data.frame(Data, Pred_hosps)[samp,]
ggplot(plot_DF, 
       aes(x=quant_HI_county, y=Pred_hosps, col = alert, alpha=0.5)) +
  geom_point() + geom_smooth(data=subset(plot_DF, alert == 0), col = "red") +
  scale_color_viridis()

ggplot(data.frame(Data, Pred_hosps)[samp,], 
       aes(x=quant_HI_county, y=Pred_hosps, col = quant_HI_yest_county, alpha=0.5)) +
  geom_point() + scale_color_viridis()

ggplot(data.frame(Data, Pred_hosps)[samp,], 
       aes(x=quant_HI_county, y=Pred_hosps, col = quant_HI_3d_county, alpha=0.5)) +
  geom_point() + scale_color_viridis()

ggplot(data.frame(Data, Pred_hosps)[samp,], 
       aes(x=quant_HI_county, y=Pred_hosps, col = dos, alpha=0.5)) +
  geom_point() + scale_color_viridis()

ggplot(data.frame(Data, Pred_hosps)[samp,], 
       aes(x=quant_HI_county, y=Pred_hosps, col = as.numeric(holiday), alpha=0.5)) +
  geom_point() + scale_color_viridis()

ggplot(data.frame(Data, Pred_hosps)[samp,], 
       aes(x=quant_HI_county, y=Pred_hosps, col = l.Pop_density, alpha=0.5)) +
  geom_point() + scale_color_viridis()

ggplot(data.frame(Data, Pred_hosps)[samp,], 
       aes(x=quant_HI_county, y=Pred_hosps, col = l.Med.HH.Income, alpha=0.5)) +
  geom_point() + scale_color_viridis()

ggplot(data.frame(Data, Pred_hosps)[samp,], 
       aes(x=quant_HI_county, y=Pred_hosps, col = broadband.usage, alpha=0.5)) +
  geom_point() + scale_color_viridis()

ggplot(data.frame(Data, Pred_hosps)[samp,], 
       aes(x=quant_HI_county, y=Pred_hosps, col = Democrat, alpha=0.5)) +
  geom_point() + scale_color_viridis()

ggplot(data.frame(Data, Pred_hosps)[samp,], 
       aes(x=quant_HI_county, y=Pred_hosps, col = Republican, alpha=0.5)) +
  geom_point() + scale_color_viridis()

ggplot(data.frame(Data, Pred_hosps)[samp,], 
       aes(x=quant_HI_county, y=Pred_hosps, col = all_hosp_mean_rate, alpha=0.5)) +
  geom_point() + scale_color_viridis()

ggplot(data.frame(Data, Pred_hosps)[samp,], 
       aes(x=quant_HI_county, y=Pred_hosps, col = death_mean_rate, alpha=0.5)) +
  geom_point() + scale_color_viridis()



plot_DF<- data.frame(Data, Pred_OH)[samp,]
ggplot(plot_DF, 
       aes(x=quant_HI_county, y=Pred_OH, col = alert, alpha=0.5)) +
  geom_point() + geom_smooth(data=subset(plot_DF, alert == 0), col = "red") +
  scale_color_viridis()

ggplot(data.frame(Data, Pred_OH)[samp,], 
       aes(x=quant_HI_county, y=Pred_OH, col = quant_HI_yest_county, alpha=0.5)) +
  geom_point() + scale_color_viridis()

ggplot(data.frame(Data, Pred_OH)[samp,], 
       aes(x=quant_HI_county, y=Pred_OH, col = quant_HI_3d_county, alpha=0.5)) +
  geom_point() + scale_color_viridis()

ggplot(data.frame(Data, Pred_OH)[samp,], 
       aes(x=quant_HI_county, y=Pred_OH, col = dos, alpha=0.5)) +
  geom_point() + scale_color_viridis()

ggplot(data.frame(Data, Pred_OH)[samp,], 
       aes(x=quant_HI_county, y=Pred_OH, col = as.numeric(holiday), alpha=0.5)) +
  geom_point() + scale_color_viridis()

ggplot(data.frame(Data, Pred_OH)[samp,], 
       aes(x=quant_HI_county, y=Pred_OH, col = l.Pop_density, alpha=0.5)) +
  geom_point() + scale_color_viridis()

ggplot(data.frame(Data, Pred_OH)[samp,], 
       aes(x=quant_HI_county, y=Pred_OH, col = l.Med.HH.Income, alpha=0.5)) +
  geom_point() + scale_color_viridis()

ggplot(data.frame(Data, Pred_OH)[samp,], 
       aes(x=quant_HI_county, y=Pred_OH, col = broadband.usage, alpha=0.5)) +
  geom_point() + scale_color_viridis()

ggplot(data.frame(Data, Pred_OH)[samp,], 
       aes(x=quant_HI_county, y=Pred_OH, col = Democrat, alpha=0.5)) +
  geom_point() + scale_color_viridis()

ggplot(data.frame(Data, Pred_OH)[samp,], 
       aes(x=quant_HI_county, y=Pred_OH, col = Republican, alpha=0.5)) +
  geom_point() + scale_color_viridis()

ggplot(data.frame(Data, Pred_OH)[samp,], 
       aes(x=quant_HI_county, y=Pred_OH, col = all_hosp_mean_rate, alpha=0.5)) +
  geom_point() + scale_color_viridis()

ggplot(data.frame(Data, Pred_OH)[samp,], 
       aes(x=quant_HI_county, y=Pred_OH, col = death_mean_rate, alpha=0.5)) +
  geom_point() + scale_color_viridis()


############### Making figure(s) for paper:

plot_DF_deaths<- data.frame(Data, Pred_deaths)[samp,]
plot_DF_deaths[,"No Alert"] = as.factor(!plot_DF_deaths$alert)
p4<- ggplot(plot_DF_deaths, 
       aes(x=quant_HI_county, y=-Pred_deaths, col = `No Alert`)) +
  geom_point(alpha=0.3) + geom_smooth(data=subset(plot_DF_deaths, alert == 0), col = "black") +
  xlab("County Quantile of Heat Index") + ylab("Predicted Deaths")

plot_DF_hosps<- data.frame(Data, Pred_hosps)[samp,]
plot_DF_hosps[,"No Alert"] = as.factor(!plot_DF_hosps$alert)
p3<- ggplot(plot_DF_hosps, 
            aes(x=quant_HI_county, y=-Pred_hosps, col = `No Alert`)) +
  geom_point(alpha=0.3) + geom_smooth(data=subset(plot_DF_hosps, alert == 0), col = "black") +
  xlab("County Quantile of Heat Index") + ylab("Predicted Hosps (All HR)")

plot_DF_OH<- data.frame(Data, Pred_OH)[samp,]
plot_DF_OH[,"No Alert"] = as.factor(!plot_DF_OH$alert)
p1<- ggplot(plot_DF_OH, 
            aes(x=quant_HI_county, y=-Pred_OH, col = `No Alert`)) +
  geom_point(alpha=0.3) + geom_smooth(data=subset(plot_DF_OH, alert == 0), col = "black") +
  xlab("County Quantile of Heat Index") + ylab("Predicted Hosps (NOHR)")

p2<- ggplot(data.frame(Data, Pred_OH)[samp,], 
            aes(x=quant_HI_county, y=-Pred_OH, col = all_hosp_mean_rate)) +
  geom_point(alpha = 0.4) + scale_color_viridis() +
  xlab("County Quantile of Heat Index") + ylab("Predicted Hosps (NOHR)") +
  guides(color=guide_legend(title="Hosps Mean"))


plot_grid(p1,
          p2,
          p3,
          p4,
          nrow = 2)
