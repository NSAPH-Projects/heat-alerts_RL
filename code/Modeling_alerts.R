library(lubridate)
library(ggplot2)
library(dplyr)
library(caret)
library(mgcv)
library(lme4)

load("data/HARL_prelim_image.RData")
rm(list=setdiff(ls(), "summer"))

n_counties<- length(unique(summer$GEOID))
n_years<- 11
n_days<- 153

summer$dos<- rep(1:n_days, n_counties*n_years) # same idea as "time" variable

#### Model issuance of a heat alert:

set.seed(321)
eda_set<- summer[sample(1:nrow(summer), 0.05*nrow(summer), replace = FALSE),]
Eda_set<- data.frame(scale(eda_set[,vars<- c("year", "dos", 
                                             "Med.HH.Income", "Pop_density",
                                             "HImaxF_PopW", "quant_HI_county",
                                             "HI_lag1", "quant_HI_yest_county",
                                             "HI_lag2",
                                             "quant_HI_3d_county")]), 
                     alert = factor(eda_set$alert),
                     dow = factor(eda_set$dow), 
                     holiday = factor(eda_set$Holiday))

day_0<- glm(alert ~ year + dos + dow + holiday,
            family = binomial(), data = Eda_set)

day_1<- glm(alert ~ year + dos + dow + holiday + Med.HH.Income + Pop_density,
            family = binomial(), data = Eda_set)

day_HI.abs<- glm(alert ~ HImaxF_PopW
                 + year + dos + dow + holiday 
                 + Med.HH.Income + Pop_density,
                 family = binomial(), data = Eda_set)

day_HI.abs_yest<- glm(alert ~ HI_lag1
                 + year + dos + dow + holiday 
                 + Med.HH.Income + Pop_density,
                 family = binomial(), data = Eda_set)

day_HI.qnt<- glm(alert ~ quant_HI_county
              + year + dos + dow + holiday 
              + Med.HH.Income + Pop_density,
              family = binomial(), data = Eda_set)

day_HI.qnt_yest<- glm(alert ~ quant_HI_yest_county
                 + year + dos + dow + holiday 
                 + Med.HH.Income + Pop_density,
                 family = binomial(), data = Eda_set)

day_HI.both<- glm(alert ~ HImaxF_PopW + quant_HI_county
                  + HI_lag1 + quant_HI_yest_county
                  + HI_lag2 
                  # + HI_3days 
                  + quant_HI_3d_county
                  + year + dos + dow + factor(holiday) 
                  + Med.HH.Income + Pop_density,
                  family = binomial(), 
                  data = Eda_set)


## Include random effects by state / county / region:

day_HI.re0<- glmer(alert ~ HImaxF_PopW + quant_HI_county
                   + HI_lag1 + quant_HI_yest_county
                   + HI_lag2 
                   # + HI_3days 
                   + quant_HI_3d_county
                   + year + dos + dow + factor(holiday) 
                   + Med.HH.Income + Pop_density 
                   + (1 | geoid),
                   family = binomial(), 
                   data = data.frame(Eda_set, geoid = factor(eda_set$GEOID)))


## Include Nonlinear Effects:

day_HI.gam0<- gam(alert ~ s(HImaxF_PopW) # 2df
                 + year + s(dos) # 2df
                 + dow + factor(holiday) 
                 + s(Med.HH.Income) # 1df?
                 + s(Pop_density), # 3df? Or just need random effect to be added?
                 family = binomial(), 
                 data = Eda_set)

plot(day_HI.gam0, scale = 0)

day_HI.gam1<- gam(alert ~ s(HImaxF_PopW, k = 2, fx=TRUE)
                  + year + s(dos, k = 2, fx=TRUE)
                  + dow + factor(holiday) 
                  + Med.HH.Income
                  + s(Pop_density),
                  # + s(geoid, bs = "re"), 
                  family = binomial(), 
                  data = data.frame(Eda_set, geoid = factor(eda_set$GEOID)))

plot(day_HI.gam1, scale = 0)

## GAMM?

day_HI.gam1b<- gamm(alert ~ s(HImaxF_PopW, k = 2, fx=TRUE)
                  + year + s(dos, k = 2, fx=TRUE)
                  + dow + factor(holiday) 
                  + Med.HH.Income
                  + s(Pop_density),
                  random = list(geoid=~1), # county id
                  family = binomial(), 
                  data = data.frame(Eda_set, geoid = factor(eda_set$GEOID)))

plot(day_HI.gam1b, scale = 0)

#### Model cumulative number of alerts:

summer$day<- day(summer$Date)
last_day<- summer[which(summer$month == 9 & summer$day == 30),]

# Think about how to summarize heat index (quantile) for modeling total number of alerts


plot(log(last_day$Population), last_day$alert_sum)
plot(log(last_day$Pop_density), last_day$alert_sum)

total_lm<- lm(alert_sum)
