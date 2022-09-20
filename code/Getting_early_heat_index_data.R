
library(lubridate)

data<- readRDS("data/Forward-backward_lags.rds")

data$Date<- as.Date(data$Date)
data$month<- month(data$Date)
summer<- data[which(data$month %in% 5:9),] # excluding April and October
summer$Holiday<- as.numeric((summer$holiday == 1) |
                              (summer$dow %in% c("Saturday", "Sunday")))

heat<- readRDS("data/Heatvars_County_2000-2020_v1.2.Rds") # downloaded from https://figshare.com/articles/dataset/Daily_County-Level_Wet-Bulb_Globe_Temperature_Universal_Thermal_Climate_Index_and_Other_Heat_Metrics_for_the_Contiguous_United_States_2000-2020/19419836

HI<- heat[,c("StCoFIPS", "Date", "HImax_C")]

HI[,3]<- (9/5)*HI[,3] + 32

names(HI)<- c("GEOID", "Date", "HImaxF_PopW")

HI$Date<- as.Date(HI$Date)
HI$year<- year(HI$Date)
HI$month<- month(HI$Date)

relevant<- HI[which(HI$year < 2017 & HI$month %in% 5:9
                      & HI$GEOID %in% unique(summer$GEOID)),]

saveRDS(relevant, "data/Heat-Index_early_years.rds")

