
library(ggplot2)

load("data/Small_S-A-R_prepped.RData")
DF$alert<- A

pred_deaths<- read.csv("Fall_results/R_1-2_deaths.csv")
Pred_deaths<- sapply(1:length(A), function(i) pred_deaths[i,A[i]+2])
# Pred_deaths[which(Pred_deaths > 0)]<- 0
pred_OH<- read.csv("Fall_results/R_1-2_other-hosps.csv")
Pred_OH<- sapply(1:length(A), function(i) pred_OH[i,A[i]+2])
pred_hosps<- read.csv("Fall_results/R_1-2_all-hosps.csv")
Pred_hosps<- sapply(1:length(A), function(i) pred_hosps[i,A[i]+2])

#### Summary stats:

summary(R_deaths[,1])
summary(Pred_deaths*1000)
summary(R_all_hosps[,1])
summary(Pred_hosps*1000)
summary(R_other_hosps[,1])
summary(Pred_OH*1000)


cor(R_deaths[,1], Pred_deaths)^2
cor(R_all_hosps[,1], Pred_hosps)^2
cor(R_other_hosps[,1], Pred_OH)^2

## Comparing effect of alerts:

summary(pred_deaths$X1 - pred_deaths$X0)
summary(pred_hosps$X1 - pred_hosps$X0)
summary(pred_OH$X1 - pred_OH$X0)

summary(pred_deaths$X1[A==1] - pred_deaths$X0[A==1])
summary(pred_hosps$X1[A==1] - pred_hosps$X0[A==1])
summary(pred_OH$X1[A==1] - pred_OH$X0[A==1])

#### Plots:

set.seed(321)
samp<- sample(1:length(A), round(0.05*length(A)))

plot(R_deaths[samp,1], Pred_deaths[samp], col = A+1, main = "Deaths",
     ylim=c(-16, 0))
abline(0,1)

plot(R_all_hosps[samp,1], Pred_hosps[samp], col = A+1, 
     main = "All Hospitalizations")
abline(0,1)

plot(R_other_hosps[samp,1], Pred_OH[samp], col = A+1,
     main = "Other Hospitalizations")
abline(0,1)




