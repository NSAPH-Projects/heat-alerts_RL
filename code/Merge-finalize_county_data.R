library(dplyr)
library(stringr)
library(zoo)
library(lubridate)

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

### Add in "forecasts":

DF2_list<- lapply(1:num_county, function(i){
  dat_fips_1 <- subset(DF,
                       GEOID == unique(DF$GEOID)[i])
  
  dat_fips_1$HI_fwd1 = c(dat_fips_1$HImaxF_PopW[2:(nrow(dat_fips_1))],
                         median(dat_fips_1$HImaxF_PopW, na.rm = T))
  dat_fips_1$HI_fwd2 = c(dat_fips_1$HImaxF_PopW[3:(nrow(dat_fips_1))],
                         rep(median(dat_fips_1$HImaxF_PopW, na.rm = T),2))
  dat_fips_1$HI_fwd3 = c(dat_fips_1$HImaxF_PopW[4:(nrow(dat_fips_1))],
                         rep(median(dat_fips_1$HImaxF_PopW, na.rm = T),3))
  
  return(dat_fips_1)
})

DF2<- bind_rows(DF2_list)

saveRDS(DF2, "data/Forward-backward_lags.rds")

### Continue refining:
data<- readRDS("data/Forward-backward_lags.rds")

data$Date<- as.Date(data$Date)
data$month<- month(data$Date)
summer<- data[which(data$month %in% 5:9),] # excluding April and October
summer$Holiday<- as.numeric((summer$holiday == 1) |
                              (summer$dow %in% c("Saturday", "Sunday")))

my_quant<- function(df, region_var, split_var #, probs)
){
  regions<- unique(df[, region_var]) # state or county?
  
  q<- rep(0,dim(df)[1])
  
  for(r in regions){
    pos<- which(df[, region_var] == r)
    # r_quants<- quantile(df[pos, split_var], probs)
    # q[pos]<- as.numeric(cut(df[pos, split_var], r_quants))
    percentile<- ecdf(df[pos, split_var])
    q[pos]<- percentile(df[pos, split_var])
  }
  return(q)
}

summer$quant_HI<- my_quant(summer, "state", "HImaxF_PopW")
summer$quant_HI_yest<- my_quant(summer, "state", "HI_lag1")
summer$quant_HI_3d<- my_quant(summer, "state", "HI_3days")

summer$quant_HI_county<- my_quant(summer, "GEOID", "HImaxF_PopW")
summer$quant_HI_yest_county<- my_quant(summer, "GEOID", "HI_lag1")
summer$quant_HI_3d_county<- my_quant(summer, "GEOID", "HI_3days")

# summer$failed_alert_abs<- as.numeric(summer$alert & summer$HImaxF_PopW < 90) # absolute 
# summer$failed_alert_rel<- as.numeric(summer$alert & summer$quant_HI < 0.8) # relative
# summer$failed_alert_rel_county<- as.numeric(summer$alert & summer$quant_HI_county < 0.8) # relative

### Add in population 65+ and DoE climate zones:

pop_65<- read.csv("data/Pop_Medicare-age.csv")
zones<- read.csv("data/Prepped_zones.csv")
manual_add<- c(46113, zones[which(zones$GEOID == 46047),2:3]) # this county fips changed in 2015
names(manual_add)<- names(zones)
zones<- rbind(zones, manual_add)

new_data<- inner_join(pop_65, zones, "GEOID")

new_data$GEOID<- str_pad(new_data$GEOID, 5, pad = "0")

Summer<- inner_join(new_data, summer, by = "GEOID")


saveRDS(Summer, "data/Final_data_for_HARL.rds")


