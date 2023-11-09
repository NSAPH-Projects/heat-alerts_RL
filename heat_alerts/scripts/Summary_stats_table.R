
library(readr)
library(dplyr)
library(xtable)

n_days<- 153

counties<- c(41067, 53015, 20161, 37085, 48157,
             28049, 19153, 17167, 31153, 6071, 4013,
             34021, 19155, 17115, 29021, 29019, 5045, 40017, 21059,
             47113, 42017, 22109, 45015, 13031, 48367, 22063, 41053,
             32003, 4015, 6025)

data<- read_csv("data/Summer23_Train_smaller-for-Python.csv")
data$nohr_rate<- (data$other_hosps / data$total_count)*1000
data$pct90<- data$quant_HI_county >= 0.9

sum_Data<- aggregate(. ~ fips + year, data[,c("fips", "year", "alert", "nohr_rate", "pct90")], sum)

### Overall:

alert_days<- data[which(data$alert == 1), c("HImaxF_PopW", "quant_HI_county", "dos")]

pct90_days<- data[which(data$pct90), c("dos")]

### Our 30 counties:

sum_Data_30<- sum_Data[which(sum_Data$fips %in% counties),]

data_30<- data[which(data$fips %in% counties),]

d30_alert_days<- data_30[which(data_30$alert == 1), c("HImaxF_PopW", "quant_HI_county", "dos")]

d30_pct90_days<- data_30[which(data_30$pct90), c("dos")]

### Make table:

var_list_30<- list(
  sum_Data_30$nohr_rate, sum_Data_30$alert, 
  d30_alert_days$HImaxF_PopW, d30_alert_days$quant_HI_county*100, d30_alert_days$dos,
  sum_Data_30$pct90, d30_pct90_days$dos
)
var_list_all<- list(
  sum_Data$nohr_rate, sum_Data$alert, 
  alert_days$HImaxF_PopW, alert_days$quant_HI_county*100, alert_days$dos,
  sum_Data$pct90, pct90_days$dos
)

for(v in 1:length(var_list_30)){
  s_30<- round(summary(var_list_30[[v]]),1)
  s_all<- round(summary(var_list_all[[v]]),1)
  x<- c()
  for(i in 1:5){
    x<- paste0(x, s_30[i], " (", s_all[i], ")", " & ")
  }
  x<- paste0(x, s_30[6], " (", s_all[6], ")", " \\ ")
  print(x)
}




