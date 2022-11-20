library(dplyr)
library(stringr)
library(zoo)
library(lubridate)


Data<- readRDS("data/Final_data_for_HARL_w-hosps.rds")

num_county<- length(table(Data$fips))

fips<- unique(Data$fips)

DF_list<- lapply(1:num_county, function(i){

  dat_fips_1<- subset(Data, fips == fips[i])
  
  dat_fips_1 <- dat_fips_1 %>%
    group_by(id, year) %>%
    mutate(all_hosp_mean = cummean(ifelse(is.na(all_hosps), 0, all_hosps)),
           heat_hosp_mean = cummean(ifelse(is.na(heat_hosps), 0, heat_hosps)),
           other_hosp_mean = cummean(ifelse(is.na(other_hosps), 0, other_hosps))
           )
  dat_fips_1$all_hosp_mean = c(rep(0, 1),
                           dat_fips_1$all_hosp_mean[1:(nrow(dat_fips_1) - 1)])
  dat_fips_1$heat_hosp_mean = c(rep(0, 1),
                            dat_fips_1$heat_hosp_mean[1:(nrow(dat_fips_1) - 1)])
  dat_fips_1$other_hosp_mean = c(rep(0, 1),
                                dat_fips_1$other_hosp_mean[1:(nrow(dat_fips_1) - 1)])
  
  return(dat_fips_1)
})

DF<- bind_rows(DF_list)

DF$death_mean_rate<- DF$death_mean / DF$Pop.65
DF$all_hosp_mean_rate<- DF$all_hosp_mean / DF$total_count
DF$heat_hosp_mean_rate<- DF$heat_hosp_mean / DF$total_count
DF$other_hosp_mean_rate<- DF$other_hosp_mean / DF$total_count

sum(is.na(DF[,c("death_mean_rate", "all_hosp_mean_rate", 
                "heat_hosp_mean_rate", "other_hosp_mean_rate")]))

saveRDS(DF, "data/Final_data_for_HARL_w-hosps_confounders.rds")



