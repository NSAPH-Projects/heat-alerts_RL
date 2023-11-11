
library(ggplot2)

cate<- read.csv("Summer_results/CATEs_6-11.csv", header=FALSE)
names(cate)<- c("cate", "LCI", "UCI", "A")
S<- read.csv("Summer_results/X_small.csv", header=FALSE)
names(S)<- c("quant_HI_county", "HI_mean", "l.Pop_density", "l.Med.HH.Income",
             "year", "dos", "T_since_alert", "alert_sum", # "More_alerts", 
             "all_hosp_mean_rate", "weekend")

sum(cate$LCI < 0 & cate$UCI > 0) # vast majority cross 0
widths<- cate$UCI - cate$LCI

DF<- data.frame(cate, S)

summary(DF$cate[DF$A==1])
summary(DF$cate[DF$A==0])

summary(widths[DF$A==1])
summary(widths[DF$A==0])


ggplot(DF, aes(x=year, y=cate)) + geom_point()

#### Look at estimated rewards:

R<- read.csv("Summer_results/Modeled_R_6-11.csv", header = FALSE)
R0<- R[,1]
R1<- R[,2]

diff<- R1-R0

