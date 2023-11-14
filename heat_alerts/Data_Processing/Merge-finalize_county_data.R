library(dplyr)
library(stringr)
library(zoo)
library(lubridate)

## Merge in more county-level information (SES and pop density):

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

## Get earlier heat index data:
heat<- readRDS("data/Heat-Index_early_years.rds") # ignoring...


my_quant<- function(df, region_var, split_var #, probs)
){
  regions<- unique(df[, region_var]) # state or county
  
  q<- rep(0,dim(df)[1])
  
  for(r in regions){
    pos<- which(df[, region_var] == r)
    percentile<- ecdf(df[pos, split_var])
    q[pos]<- percentile(df[pos, split_var])
    # for(m in months){
    #   pos<- which(df[, region_var] == r & df[,"month"] == m)
    #   percentile<- ecdf(df[pos, split_var])
    #   q[pos]<- percentile(df[pos, split_var])
    # }
    
  }
  return(q)
}

months<- 5:9

summer$quant_HI<- my_quant(summer, "state", "HImaxF_PopW")
summer$quant_HI_yest<- my_quant(summer, "state", "HI_lag1")
summer$quant_HI_3d<- my_quant(summer, "state", "HI_3days")

summer$quant_HI_county<- my_quant(summer, "GEOID", "HImaxF_PopW")
summer$quant_HI_yest_county<- my_quant(summer, "GEOID", "HI_lag1")
summer$quant_HI_3d_county<- my_quant(summer, "GEOID", "HI_3days")

summer$HI_fwd_avg<- rowMeans(summer[,c("HImaxF_PopW", "HI_fwd1", "HI_fwd2")])
summer$quant_HI_tmw<- my_quant(summer, "state", "HI_fwd1")
summer$quant_HI_fwd_avg<- my_quant(summer, "state", "HI_fwd_avg")

summer$quant_HI_tmw_county<- my_quant(summer, "GEOID", "HI_fwd1")
summer$quant_HI_fwd_avg_county<- my_quant(summer, "GEOID", "HI_fwd_avg")

### Add in population 65+ and DoE climate zones:

pop_65<- read.csv("data/Pop-Medicare_county-age.csv")
##  Make adjustments based on county changes over time: https://www.census.gov/programs-surveys/geography/technical-documentation/county-changes.2010.html
pop_65[which(pop_65$GEOID == 46102), "GEOID"]<- 46113 # this county fips changed in 2015
#ignore 51515 because it is small (will be ignored in later steps)
#ignore all starting with 02 because those are AK (outside continental US)
zones<- read.csv("data/Prepped_zones.csv")
manual_add<- c(46113, zones[which(zones$GEOID == 46102),2:3]) 
names(manual_add)<- names(zones)
zones<- rbind(zones, manual_add)

new_data<- inner_join(pop_65, zones, "GEOID")

new_data$GEOID<- str_pad(new_data$GEOID, 5, pad = "0")

Summer<- inner_join(new_data, summer, by = c("GEOID", "year")) 
# ^^^ difference in rows compared to summer is due to missing 51515 from 2011 onwards

saveRDS(Summer, "data/Final_data_for_HARL.rds")


