library(dplyr)
library(stringr)
library(zoo)
library(lubridate)


Data<- readRDS("data/Final_data_for_HARL_w-hosps.rds")

num_county<- length(table(Data$fips))

Fips<- unique(Data$fips)

DF_list<- lapply(1:num_county, function(i){

  dat_fips_1<- subset(Data, fips == Fips[i])
  
  dat_fips_1<- dat_fips_1 %>%
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
  
  dat_fips_1<- dat_fips_1 %>%
    group_by(id, year) %>%
    mutate(all_hosp_2wkMA = c(rep(0, 13), rollmean(ifelse(is.na(all_hosps), 0, all_hosps), 14, align="right")),
           heat_hosp_2wkMA = c(rep(0, 13), rollmean(ifelse(is.na(heat_hosps), 0, heat_hosps), 14, align="right")),
           other_hosp_2wkMA = c(rep(0, 13), rollmean(ifelse(is.na(other_hosps), 0, other_hosps), 14, align="right"))
    )
  
  dat_fips_1<- dat_fips_1 %>%
    group_by(id, year) %>%
    mutate(all_hosp_3dMA = c(rep(0, 2), rollmean(ifelse(is.na(all_hosps), 0, all_hosps), 3, align="right")),
           heat_hosp_3dMA = c(rep(0, 2), rollmean(ifelse(is.na(heat_hosps), 0, heat_hosps), 3, align="right")),
           other_hosp_3dMA = c(rep(0, 2), rollmean(ifelse(is.na(other_hosps), 0, other_hosps), 3, align="right"))
    )
  
  return(dat_fips_1)
})

DF<- bind_rows(DF_list)

DF$death_mean_rate<- DF$death_mean / DF$Pop.65
DF$all_hosp_mean_rate<- DF$all_hosp_mean / DF$total_count
DF$heat_hosp_mean_rate<- DF$heat_hosp_mean / DF$total_count
DF$other_hosp_mean_rate<- DF$other_hosp_mean / DF$total_count

DF$all_hosp_2wkMA_rate<- DF$all_hosp_2wkMA / DF$total_count
DF$heat_hosp_2wkMA_rate<- DF$heat_hosp_2wkMA / DF$total_count
DF$other_hosp_2wkMA_rate<- DF$other_hosp_2wkMA / DF$total_count

DF$all_hosp_3dMA_rate<- DF$all_hosp_3dMA / DF$total_count
DF$heat_hosp_3dMA_rate<- DF$heat_hosp_3dMA / DF$total_count
DF$other_hosp_3dMA_rate<- DF$other_hosp_3dMA / DF$total_count

sum(is.na(DF[,c("death_mean_rate", "all_hosp_mean_rate", 
                "heat_hosp_mean_rate", "other_hosp_mean_rate")]))

sum(is.na(DF[,c("all_hosp_2wkMA_rate", "heat_hosp_2wkMA_rate", "other_hosp_2wkMA_rate")]))
sum(is.na(DF[,c("all_hosp_3dMA_rate", "heat_hosp_3dMA_rate", "other_hosp_3dMA_rate")]))

DF$age_65_74_rate<- DF$age_65_74 / DF$total_count
DF$age_75_84_rate<- DF$age_75_84 / DF$total_count
DF$dual_rate<- DF$dual_count / DF$total_count

sum(is.na(DF[,c("age_65_74_rate", "age_75_84_rate", "dual_rate")]))


## Incorporate a couple more non-Medicare datasets:
hvi<- read.csv("data/HVI_county-level.csv")
hvi$fips<- str_pad(hvi$fips, 5, pad="0")


internet<- read.csv("/n/dominici_nsaph_l3/Lab/data/confounders/miscellaneous_US/broadband_data_2020October.csv", skip = 18)
Internet<- data.frame(fips = str_pad(internet$COUNTY.ID, 5, pad="0"),
                      broadband.availability=internet$BROADBAND.AVAILABILITY.PER.FCC,
                      broadband.usage=internet$BROADBAND.USAGE)
Internet$fips[which(Internet$fips == "46102")]<- "46113"
full_DF<- inner_join(DF, Internet, by = "fips")
# Full_DF<- inner_join(full_DF, hvi, by = "fips") 
# missing_fips<- unique(full_DF$fips[which(! full_DF$fips %in% Full_DF$fips)])
# Missing<- inner_join(distinct(DF[,c("fips", "Population")]), data.frame(fips=missing_fips))

politics<- read.csv("data/Cleaned_election_data.csv")

politics<- politics %>% na.omit()

# head(politics[order(politics$county_fips),])
# pol_var<- aggregate(. ~ county_fips, politics, sd)

politics$fips<- str_pad(politics$county_fips, 5, pad="0")

f<- unique(politics$fips)[1]
pos<- which(politics$fips == f)
res<- apply(politics[pos,3:4], MARGIN=2, function(x) c( mean(c(x[1], x[2])), weighted.mean(c(x[1], x[2]), c(0.25, 0.75)), x[2],
                                                        weighted.mean(c(x[2], x[3]), c(0.75, 0.25)), mean(c(x[2], x[3])),
                                                        weighted.mean(c(x[2], x[3]), c(0.25, 0.75)), x[3],
                                                        weighted.mean(c(x[3], x[4]), c(0.75, 0.25)), mean(c(x[3], x[4])),
                                                        weighted.mean(c(x[3], x[4]), c(0.25, 0.75)), x[4] ) )
df<- data.frame(year = 2006:2016, fips = f, res)
row.names(df)<- NULL
Politics<- df

for(f in unique(politics$fips)[-1]){
  pos<- which(politics$fips == f)
  res<- apply(politics[pos,3:4], MARGIN=2, function(x) c( mean(c(x[1], x[2])), weighted.mean(c(x[1], x[2]), c(0.25, 0.75)), x[2],
                 weighted.mean(c(x[2], x[3]), c(0.75, 0.25)), mean(c(x[2], x[3])),
                 weighted.mean(c(x[2], x[3]), c(0.25, 0.75)), x[3],
                 weighted.mean(c(x[3], x[4]), c(0.75, 0.25)), mean(c(x[3], x[4])),
                 weighted.mean(c(x[3], x[4]), c(0.25, 0.75)), x[4] ) )
  df<- data.frame(year = 2006:2016, fips = f, res)
  row.names(df)<- NULL
  Politics<- rbind(Politics, df)
}


Full_DF<- inner_join(full_DF, Politics, by = c("fips", "year"))

# saveRDS(Full_DF, "data/Final_data_for_HARL_w-hosps_confounders.rds")

## Add in time since last alert:
# Full_DF<- readRDS("data/Final_data_for_HARL_w-hosps_confounders.rds")

n_days<- 153
episode_inds<- seq(1, nrow(Full_DF), n_days)

T_since_alert<- rep(0, nrow(Full_DF))
for(i in 2:length(episode_inds)){
  pos<- episode_inds[i-1]:(episode_inds[i]-1)
  t<- 300 # start of each summer
  for(d in 1:n_days){
    T_since_alert[pos[d]]<- t
    if(Full_DF$alert[pos[d]] == 0){
      t<- t+1
    }else{
      t<- 0
    }
  }
  print(i)
}

# hist(T_since_alert)
Full_DF$T_since_alert<- T_since_alert

## Add in annual average air quality:

all_AQ<- read.csv(paste0("/n/dominici_nsaph_l3/Lab/exposure/pm25/whole_us/annual/counties/county_rm/data/county_pm25_2006.csv"))
all_AQ$fips<- str_pad(all_AQ$fips, 5, pad="0")

years<- 2007:2016
for(y in years){
  AQ<- read.csv(paste0("/n/dominici_nsaph_l3/Lab/exposure/pm25/whole_us/annual/counties/county_rm/data/county_pm25_", y, ".csv"))
  AQ$fips<- str_pad(AQ$fips, 5, pad="0")
  all_AQ<- rbind(all_AQ, AQ)
}

all_AQ$fips[which(all_AQ$fips == "46102")]<- "46113"
Final_DF<- left_join(Full_DF, all_AQ, by = c("fips","year"))

saveRDS(Final_DF, "data/Final_data_for_HARL_w-hosps_confounders.rds")



#### *Inspect* alert time deltas:

deltas<- read.csv("data/heat_wave_time_delta.csv")

my_split<- str_split(deltas$time_delta_issue_start, " days ")
days<- as.numeric(unlist(lapply(my_split, function(x) x[1])))
hours.0<- unlist(lapply(my_split, function(x) x[2]))
hours.1<- str_split(hours.0, ":")
hours<- as.numeric(unlist(lapply(hours.1, function(x) x[1])))
mins<- as.numeric(unlist(lapply(hours.1, function(x) x[2])))
secs<- as.numeric(unlist(lapply(hours.1, function(x) x[3])))

Deltas<- data.frame(Date = deltas$day, fips = deltas$geoid10, 
                    delta = days*24 + hours + mins/60 + secs/3600)
Deltas$fips<- str_pad(Deltas$fips, 5, pad="0")
Deltas$Date<- as.Date(Deltas$Date)

FINAL<- left_join(Final_DF, Deltas) # a lot of missingness


# saveRDS(FINAL, "data/Final_data_for_HARL_w-hosps_confounders.rds")

# ## Out of curiosity:
# 
# test<- distinct(Full_DF[,c("Population", "Pop_density", "Med.HH.Income", "broadband.usage", "Democrat", "Republican")])
# test$broadband.usage<- as.numeric(test$broadband.usage)
# Test<- test[which(test$Population >= 65000),]
# 
# library(ggplot2)
# 
# ggplot(test, aes(x=log(Pop_density), y=Democrat)) + geom_point() + geom_smooth()
# 
# ggplot(test, aes(x=log(Med.HH.Income), y=Republican)) + geom_point() + geom_smooth()
# 
# ggplot(distinct(test[,3:4]), aes(x=log(Med.HH.Income), y=broadband.usage)) + geom_point() + geom_smooth()
# 
# ggplot(test, aes(x=broadband.usage, y=Democrat))+ geom_point() + geom_smooth()


