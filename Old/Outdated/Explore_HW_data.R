setwd("~/shared_space/ci3_analysis/ellen_heat-warnings_RL")
hihw<- readRDS("~/shared_space/ci3_analysis/ts_heat_warnings/data/ts_heatindex_heatwarnings_byFIPS_2006-2016.RDS")

library(lubridate)

## Explore data:

summary(hihw$HImaxF_PopW)

aggregate(. ~ state, hihw[,6:9], sum) # use heat alerts (all)
sum(hihw$warn == 1 & hihw$alert == 0)
sum(hihw$adv == 1 & hihw$alert == 0)

hihw$Year<- year(hihw$Date)

recent<- hihw[which(hihw$Year >= 2012),]
state_years<- aggregate(Year ~ state, recent, unique)

recent$Days<- 1
state_alerts<- aggregate(. ~ state, recent[,c(6,9,11)], sum)
state_alerts$Rate<- state_alerts$alert/state_alerts$Days
summary(state_alerts$Rate)

state_alerts$ann_avg<- state_alerts$alert/5
summary(state_alerts$ann_avg)

county_alerts<- aggregate(. ~ county + zone, recent[,c(4,5,9,11)], sum) # 84 more counties than zones
county_alerts$Rate<- county_alerts$alert/county_alerts$Days
summary(county_alerts$Rate)

county_alerts$ann_avg<- county_alerts$alert/5
summary(county_alerts$ann_avg)


