library(dplyr)
library(lubridate)
library(readr)

summer<- readRDS("data/Final_data_for_HARL_w-hosps_confounders.rds")

n_counties<- length(unique(summer$fips))
n_years<- 11
n_days<- 153

## Last few adjustments:

summer$l.Med.HH.Income<- log(summer$Med.HH.Income)
summer$l.Pop_density<- log(summer$Pop_density)

over_10k<- summer[which(summer$Population >= 10000),] # 2261 counties
fips<- unique(over_10k$fips)
too_few<- setdiff(unique(summer$fips), fips)

Train<- summer[which(summer$year < 2016),]
Test<- summer[which(summer$year == 2016),]

save(Train, Test, file="data/Summer23_Train-Test.RData")

## Subset further and write to format useful for Python:

Top3rd<- summer[which(summer$Population>= 65000),]

write_csv(Top3rd, "data/Summer23_Train_smaller-for-Python.csv")


