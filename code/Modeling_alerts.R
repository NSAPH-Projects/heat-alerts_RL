library(lubridate)
library(ggplot2)
library(dplyr)

load("data/HARL_prelim_image.RData")

summer$day<- day(summer$Date)
last_day<- summer[which(summer$month == 9 & summer$day == 30),]

# Think about how to summarize heat index (quantile) for modeling total number of alerts

#### Model issuance of a heat alert:

fit_day<- lm(alert ~ )

#### Model cumulative number of alerts:

plot(log(last_day$Population), last_day$alert_sum)
plot(log(last_day$Pop_density), last_day$alert_sum)

fit_cum<- lm(alert_sum)
