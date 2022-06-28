library(lubridate)
library(ggplot2)
library(dplyr)
library(caret)

load("data/HARL_prelim_image.RData")
rm(list=setdiff(ls(), "summer"))

n_counties<- length(unique(summer$GEOID))
n_years<- 11
n_days<- 153

summer$dos<- rep(1:n_days, n_counties*n_years) # same idea as "time" variable

#### Model issuance of a heat alert:

set.seed(321)
eda_set<- summer[sample(1:nrow(summer), 0.1*nrow(summer), replace = FALSE),]

day_0<- glm(alert ~ year + dos + dow + holiday,
            family = binomial(), data = eda_set)

day_1<- glm(alert ~ year + dos + dow + holiday + Med.HH.Income + Pop_density,
            family = binomial(), data = eda_set)

day_HI.abs<- glm(alert ~ HImaxF_PopW
                 + year + dos + dow + holiday 
                 + Med.HH.Income + Pop_density,
                 family = binomial(), data = eda_set)

day_HI.abs_yest<- glm(alert ~ HI_lag1
                 + year + dos + dow + holiday 
                 + Med.HH.Income + Pop_density,
                 family = binomial(), data = eda_set)

day_HI.qnt<- glm(alert ~ quant_HI_county
              + year + dos + dow + holiday 
              + Med.HH.Income + Pop_density,
              family = binomial(), data = eda_set)

day_HI.qnt_yest<- glm(alert ~ quant_HI_yest_county
                 + year + dos + dow + holiday 
                 + Med.HH.Income + Pop_density,
                 family = binomial(), data = eda_set)

day_HI.both<- glm(alert ~ HImaxF_PopW + quant_HI_county
                  + HI_lag1 + quant_HI_yest_county
                  + HI_lag2 
                  # + HI_3days 
                  + quant_HI_3d_county
                  + year + dos + dow + factor(holiday) 
                  + Med.HH.Income + Pop_density,
                  family = binomial(), 
                  data = data.frame(scale(eda_set[,vars<- c("year", "dos", 
                                                 "Med.HH.Income", "Pop_density",
                                                 "HImaxF_PopW", "quant_HI_county",
                                                 "HI_lag1", "quant_HI_yest_county",
                                                 "HI_lag2",
                                                 "quant_HI_3d_county")]), 
                               alert = factor(eda_set$alert),
                                    dow = factor(eda_set$dow), 
                               holiday = factor(eda_set$Holiday)))


## Include random effects by state / county / region?




#### Model cumulative number of alerts:

summer$day<- day(summer$Date)
last_day<- summer[which(summer$month == 9 & summer$day == 30),]

# Think about how to summarize heat index (quantile) for modeling total number of alerts


plot(log(last_day$Population), last_day$alert_sum)
plot(log(last_day$Pop_density), last_day$alert_sum)

total_lm<- lm(alert_sum)
