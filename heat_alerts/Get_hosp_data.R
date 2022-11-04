
## Adapted from Xiao Wu's script https://github.com/wxwx1993/TS_Stochastic/blob/main/Application/RCE/batch_hosp_all.R

library(fst)
library(stringr)
library(fBasics)
library(lubridate)
library(data.table)
library(dplyr)

## Read in heat-related hospitalization data:

f<- list.files("/n/dominici_nsaph_l3/data/ci3_health_data/medicare/heat_related/2006_2016/county_ccs_hosps/data/",
                pattern = "\\.fst",
                full.names = TRUE)
vars<- c("fips", "day", "ccs_55", "ccs_157", "ccs_159", "ccs_2", "ccs_244", "ccs_114", "ccs_50", "total_count")
daily_fips_hosp <- rbindlist(lapply(f,
                                    read_fst,
                                    columns = vars,
                                    as.data.table = TRUE))
daily_fips_hosp$fips <- str_pad(daily_fips_hosp$fips, 5, pad = "0")
colnames(daily_fips_hosp)[2]<- "Date"

## Merge in with mortality (and other) data:

data<- readRDS("data/Final_data_for_HARL.rds")
names(data)[1]<- "fips"

Data<- inner_join(data, daily_fips_hosp, by=c("fips", "Date"))
Data$all_hosps<- rowSums(Data[,c("ccs_55", "ccs_157", "ccs_159", "ccs_2", "ccs_244", "ccs_114", "ccs_50")])

saveRDS(Data, "data/Final_data_for_HARL_w-hosps.rds")

## Compare total_count and Pop.65:

y<- 2011
test<- Data[which(Data$year==y), c("fips", "total_count", "Pop.65")]
Test<- distinct(test)
plot(Test$Pop.65, Test$total_count)
abline(0,1)


