library(dplyr)
library(stringr)
library(zoo)

setwd("/n/dominici_nsaph_l3/projects/heat-alerts_mortality_RL")

# ## Merge in more county-level information (SES and pop density):
# 
# ACS<- read.csv("data/County_data.csv")
# ACS$GEOID<- str_pad(ACS$GEOID, width = 5, side = "left", pad = "0")
# 
# area<- read.csv("data/Counties_land_area.csv")
# area$fips<- str_pad(area$fips, width = 5, side = "left", pad = "0")
# 
# County_data<- merge(ACS, area, by.x = "GEOID", by.y = "fips")
# 
# County_data$Pop_density<- County_data$Population / County_data$Area
# 
# data<- readRDS("data/Merged_with_lags.rds")
# 
# data$fips<- as.character(data$fips)
# County_data$GEOID<- as.character(County_data$GEOID)
# 
# Data<- inner_join(County_data, data, by = c("GEOID" = "fips"))
# 
# saveRDS(Data, "data/Data_for_HARL.rds")
Data<- readRDS("data/Data_for_HARL.rds")


## Add in some more lagged variables (num of heat alerts issued within last two weeks, 
  # average of HI over the past 3 days):

num_county<- length(table(Data$GEOID))

DF_list<- lapply(1:num_county, function(i){
  dat_fips_1 <- subset(Data,
                       GEOID == unique(Data$GEOID)[i])
  
  dat_fips_1$HI_lag3 = c(rep(median(dat_fips_1$HImaxF_PopW, na.rm = T),3),
                         dat_fips_1$HImaxF_PopW[1:(nrow(dat_fips_1) - 3)])

  dat_fips_1$alerts_2wks<- c(rep(0, 13), rollapply(dat_fips_1$alert, 14, sum, 
                                        align = "right"))
  
  
  return(dat_fips_1)
})

DF<- bind_rows(DF_list)

DF$HI_3days<- rowMeans(DF[,c("HI_lag1", "HI_lag2", "HI_lag3")])

saveRDS(DF, "data/Final_data_for_HARL.rds")


