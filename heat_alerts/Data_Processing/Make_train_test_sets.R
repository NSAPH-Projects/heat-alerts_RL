
library(lubridate)


summer<- readRDS("data/Final_data_for_HARL_w-hosps_confounders.rds")

n_counties<- length(unique(summer$fips))
n_years<- 11
n_days<- 153

# summer$dos<- rep(1:n_days, n_counties*n_years) # same idea as "time" variable
# 
# ## Need to adjust alert_sum to account for 11 alerts from April (which I'm not including):
# summer$day<- day(summer$Date)
# april_alerts<- summer[which(summer$month==5 & summer$day == 1 & summer$alert_sum > 0), c("fips", "year")]
# summer[which(summer$year == 2009 & summer$fips %in% april_alerts$fips),"alert_sum"]<- summer[which(summer$year == 2009 & summer$fips %in% april_alerts$fips),"alert_sum"] - 1
# 
last_day<- summer[which(summer$month == 9 & summer$day == 30),] # or look at dos
# 
# ## Adjust a few other variables:
# summer$holiday<- factor(summer$holiday)
# 
summer$l.Med.HH.Income<- log(summer$Med.HH.Income)
summer$l.Pop_density<- log(summer$Pop_density)

#### Subset out the data we want to use:

over_10k<- summer[which(summer$Population >= 10000),] # 2261 counties
fips<- unique(over_10k$fips)
too_few<- setdiff(unique(last_day$fips), fips)
n_test<- round(0.2*length(fips)) # 80% training set

set.seed(321)
test_fips<- sample(fips, n_test, replace = FALSE)
train_fips<- setdiff(fips, test_fips)

Train<- over_10k[which(over_10k$fips %in% train_fips),]
Test<- over_10k[which(over_10k$fips %in% test_fips),]

save(Train, Test, file="data/Train-Test.RData")


### Save dataset without health info for Falco:

vars<- c("fips", "Date", "year", "month", "dos", "dow", "holiday",
         "BA_zone", "Population", "Pop_density", "Med.HH.Income",
         "HImaxF_PopW", "quant_HI_county", "warn", "adv", "alert")

data<- summer[,vars]

saveRDS(data, "data/Heat-alerts-data-without-health.rds")


