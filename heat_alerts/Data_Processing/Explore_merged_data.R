# library(stringr)
# library(fBasics)
# library(lubridate)
# 
# ## Adapted code from Xiao's batch_death_all.R:
# fips_popsize <- read.csv("~/shared_space/ci3_analysis/ts_heat_warnings/data/matched_county_2010_popsize.csv")
# colnames(fips_popsize)[1] <- "fips"
# fips_popsize$fips <- str_pad(fips_popsize$fips, 5, pad = "0")
# 
# # daily_fips_deaths <- read_fst("~/shared_space/ci3_analysis/heat_warnings_mortality/daily_fips_deaths_1999_2016.fst")
# daily_fips_deaths <- readRDS("~/shared_space/ci3_analysis/heat_warnings_mortality/daily_fips_deaths_1999_2016.RDS")
# daily_fips_deaths$fips <- str_pad(daily_fips_deaths$fips, 5, pad = "0")
# colnames(daily_fips_deaths)[1] <- "Date"
# 
# daily_fips_heat <- readRDS("~/shared_space/ci3_analysis/ts_heat_warnings/data/ts_heatindex_heatwarnings_byFIPS_2006-2016.RDS")
# colnames(daily_fips_heat)[1] <- "fips"
# daily_fips_heat$fips <- str_pad(daily_fips_heat$fips, 5, pad = "0")
# holidays <- c(as.Date(USMemorialDay(2006:2016)),
#               as.Date(USIndependenceDay(2006:2016)),
#               as.Date(USLaborDay(2006:2016)),
#               as.Date(USColumbusDay(2006:2016)))
# daily_fips_heat$holiday <- ifelse(daily_fips_heat$Date %in% holidays, 1, 0)
# daily_fips_heat$year <- year(daily_fips_heat$Date)
# daily_fips_heat$dow <- weekdays(daily_fips_heat$Date)
# #[1] 6678298 = 214*11*2837
# 
# daily_fips_heat <- merge(daily_fips_heat,
#                          fips_popsize,
#                          by = "fips",
#                          all.x = TRUE)
# 
# daily_fips_heat_populous <- daily_fips_heat
# rm(c("daily_fips_heat"))
# 
# daily_fips_main <- data.frame(merge(daily_fips_heat_populous,
#                                     daily_fips_deaths,
#                                     by = c("fips", "Date"),
#                                     all.x = TRUE))
# daily_fips_main$N[is.na(daily_fips_main$N)] <- 0
# 
# setwd("~/shared_space/ci3_analysis/ellen_heat-warnings_RL/data")
# saveRDS(daily_fips_main, "Merged_no_lags.rds")
# 
# ## Adding in lags:
# num_county<- length(table(daily_fips_main$fips))
# 
# DF_list<- lapply(1:num_county, function(i){
#   dat_fips_1 <- subset(daily_fips_main, 
#                        fips == unique(daily_fips_main$fips)[i])
#   
#   colnames(dat_fips_1)[18] = "N"
#   dat_fips_1$HImaxF_PopW[is.na(dat_fips_1$HImaxF_PopW)] <- median(dat_fips_1$HImaxF_PopW, na.rm = T)
#   
#   unit <- nrow(dat_fips_1)/214
#   dat_fips_1$time = rep(1:(nrow(dat_fips_1)/unit), unit)
#   dat_fips_1$id = rep(1:unit,rep(nrow(dat_fips_1)/unit, unit))
#   dat_fips_1$HI_lag1 = c(rep(median(dat_fips_1$HImaxF_PopW, na.rm = T), 1),
#                          dat_fips_1$HImaxF_PopW[1:(nrow(dat_fips_1) - 1)])
#   dat_fips_1$HI_lag2 = c(rep(median(dat_fips_1$HImaxF_PopW, na.rm = T),2),
#                          dat_fips_1$HImaxF_PopW[1:(nrow(dat_fips_1) - 2)])
#   
#   dat_fips_1$alert_lag1 = c(rep(0,1), dat_fips_1$alert[1:(nrow(dat_fips_1) - 1)])
#   dat_fips_1$alert_lag2 = c(rep(0,2), dat_fips_1$alert[1:(nrow(dat_fips_1) - 2)])
#   
#   dat_fips_1 <- dat_fips_1 %>%
#     group_by(id, year) %>%
#     mutate(HI_mean = cummean(ifelse(is.na(HImaxF_PopW), 0, HImaxF_PopW)),
#            alert_sum = cumsum(ifelse(is.na(alert), 0, alert)),
#            death_mean = cummean(ifelse(is.na(N), 0, N)))
#   dat_fips_1$alert_sum = c(rep(0, 1),
#                            dat_fips_1$alert_sum[1:(nrow(dat_fips_1) - 1)])
#   dat_fips_1$death_mean = c(rep(0, 1),
#                             dat_fips_1$death_mean[1:(nrow(dat_fips_1) - 1)])
#   return(dat_fips_1)
# })
# 
# DF<- bind_rows
# 
# saveRDS("Merged_with_lags.rds")

######################################

## My EDA for the outcomes:

# setwd("~/shared_space/ci3_analysis/ellen_heat-warnings_RL")
data<- readRDS("data/Merged_with_lags.rds")

Recent<- data[which(data$year >= 2012), ]
recent_state_outcomes<- aggregate(N ~ STATE + year, Recent, sum)
recent_county_outcomes<- aggregate(N ~ STATE + county + year, 
                                   Recent, sum)
summary(recent_state_outcomes$N)
summary(recent_county_outcomes$N)

state_outcomes<- aggregate(N ~ STATE + year, data, sum)
county_outcomes<- aggregate(N ~ STATE + county + year, data, sum)
summary(state_outcomes$N)
summary(county_outcomes$N)

## When heat index > 80 (problematic according to NWS):
High<- data[which(data$HImaxF_PopW >=80),]
High$Days<- 1

state_sums<- aggregate(. ~ state + year, 
                       data = High[,c(6, 9, 11, 18, 28)], sum)
summary(state_sums$N)
state_rate<- state_sums$alert/state_sums$Days
summary(state_rate)*100
summary(state_sums$alert/11)

county_sums<- aggregate(. ~ fips + year, 
                       data = High[,c(1, 9, 11, 18, 28)], sum)
summary(county_sums$N)
county_rate<- county_sums$alert/county_sums$Days
summary(county_rate)*100
summary(county_sums$alert/11)

## Recent years (2012-2016) with high HI:
recent<- High[which(High$year >= 2012),]

state_sums<- aggregate(. ~ state + year, 
                       data = recent[,c(6, 9, 11, 18, 28)], sum)
summary(state_sums$N)
state_rate<- state_sums$alert/state_sums$Days
summary(state_rate)*100
summary(state_sums$alert/5)

county_sums<- aggregate(. ~ fips + year, 
                        data = recent[,c(1, 9, 11, 18, 28)], sum)
summary(county_sums$N)
county_rate<- county_sums$alert/county_sums$Days
summary(county_rate)*100
summary(county_sums$alert/5)

