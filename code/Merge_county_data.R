library(dplyr)
library(stringr)

setwd("/n/dominici_nsaph_l3/projects/heat-alerts_mortality_RL")

ACS<- read.csv("data/County_data.csv")
ACS$GEOID<- str_pad(ACS$GEOID, width = 5, side = "left", pad = "0")

area<- read.csv("data/Counties_land_area.csv")
area$fips<- str_pad(area$fips, width = 5, side = "left", pad = "0")

County_data<- merge(ACS, area, by.x = "GEOID", by.y = "fips")

County_data$Pop_density<- County_data$Population / County_data$Area

data<- readRDS("data/Merged_with_lags.rds")

data$fips<- as.character(data$fips)
County_data$GEOID<- as.character(County_data$GEOID)

Data<- inner_join(County_data, data, by = c("GEOID" = "fips"))

saveRDS(Data, "data/Data_for_HARL.rds")
