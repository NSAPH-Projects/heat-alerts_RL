
## Adapted from Xiao Wu's script https://github.com/wxwx1993/TS_Stochastic/blob/main/Application/RCE/batch_hosp_all.R

library(devtools)
install_github("AndreSjuve/dretools")
library(dretools)

library(foreign)
library(fst, lib.loc="~/apps/R_4.2.2")
library(stringr)
library(fBasics, lib.loc="~/apps/R_4.2.2")
library(lubridate)
library(data.table, lib.loc="~/apps/R_4.2.2")
library(dplyr)

ssa_fips<- read.csv("/n/dominici_nsaph_l3/Lab/data_processing/ssa_fips_state_county2016.csv")
ssa_fips$fips<- ssa_fips$fipscounty

## Check columns:

folder<- "/n/dominici_nsaph_l3/data/ci3_health_data/medicare/gen_admission/1999_2016/targeted_conditions/cache_data/admissions_by_year/"

get_all_data<- function(y){
  df<- read_fst(paste0(folder, "admissions_", y, ".fst"))
  # all<- read_colnames(df)
  remove<- unique(c(which(df$ADM_TYPE %in% c(3,4)), which(df$ADM_Source == 4))) # see /n/dominici_nsaph_l3/Lab/data/ci3_health_data/medicare/gen_admission/1999_2016/targeted_conditions/condition_dictionary.md
  df<- df[-remove,]
  df$ssa<- as.integer(paste0(df$SSA_STATE_CD, df$SSA_CNTY_CD))
  df$count<- 1
  df$Date<- as.Date(df$ADATE, format="%d%b%Y")
  DF<- df %>% group_by(ssa, Date) %>% summarise(ALL_hosps = sum(count))
  DF_fips<- inner_join(DF, ssa_fips[,c("ssacounty", "fips")], by=c("ssa" = "ssacounty"))
  print(y)
  return(DF_fips)
}

ALL_data<- lapply(2006:2016, get_all_data)
ALL_data_df<- bind_rows(ALL_data)

saveRDS(ALL_data_df[,c("fips","Date", "ALL_hosps")], "data/ALL_hosps_pre-merge.rds")

## Read in heat-related hospitalization data:

f<- list.files("/n/dominici_nsaph_l3/data/ci3_health_data/medicare/heat_related/2006_2016/county_ccs_hosps/data/",
               pattern = "\\.fst",
               full.names = TRUE)
cols<- read_colnames(f[1])

vars<- c("fips", "day", "ccs_55", "ccs_157", "ccs_159", "ccs_2", "ccs_244", "ccs_114", "ccs_50", 
         "total_count", "age_65_74", "age_75_84", "age_85", "dual_count")
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
Data$heat_hosps<- rowSums(Data[,c("ccs_244", "ccs_55")])
Data$other_hosps<- rowSums(Data[,c("ccs_157", "ccs_159", "ccs_2", "ccs_114", "ccs_50")])

saveRDS(Data, "data/Final_data_for_HARL_w-hosps.rds")

## Compare total_count and Pop.65:

y<- 2011
test<- Data[which(Data$year==y), c("fips", "total_count", "Pop.65")]
Test<- distinct(test)
plot(Test$Pop.65, Test$total_count)
abline(0,1)


