library(readr)

load("data/Train-Test.RData")

n_days<- 153

S_inds<- as.numeric(row.names(Train[-seq(n_days, nrow(Train), n_days),]))
Top3rd<- Train[which(Train$Population>= 65000),]
S_t3_inds<- as.numeric(row.names(Top3rd[-seq(n_days, nrow(Top3rd), n_days),]))

write_csv(data.frame(S_inds), "data/S_training_indices.csv")
write_csv(data.frame(S_t3_inds), "data/S_t3_training_indices.csv")

write_csv(Top3rd, "data/Train_smaller-for-Python.csv")

#### Select "eligible" days based on heat index:

data<- data.frame(Top3rd)

fips<- unique(data$fips)

pct90<- c()
for(i in fips){
  pos<- which(data$fips == i)
  pct90<- append(pct90, quantile(data$HImaxF_PopW[pos], 0.9))
}
Pct_90<- rep(pct90, each = 153*11)
Pct_90_eligible<- data$HImaxF_PopW >= Pct_90

write_csv(data.frame(Pct_90_eligible), "data/Pct_90_eligible.csv")

county_min<- c()
for(i in fips){
  pos<- which(data$fips == i & data$alert == 1)
  if(length(pos) == 0){
    county_min<- append(county_min, NA)
  }else{
    m<- min(data$HImaxF_PopW[pos])
    county_min<- append(county_min, m)
  }
}
County_min<- rep(county_min, each = 153*11)
County_min_eligible<- data$HImaxF_PopW >= County_min

write_csv(data.frame(County_min_eligible), "data/County_min_eligible.csv")

# Both_eligible<- Pct_90_eligible & County_min_eligible

################ Also create set for R modeling:



budget<- data[which(data$dos == 153), "alert_sum"]
Budget<- rep(budget, each = n_days)
data$More_alerts<- Budget - data$alert_sum
Data<- data[-seq(n_days, nrow(data), n_days),]
Data$broadband.usage<- as.numeric(Data$broadband.usage)

DF<- data.frame(scale(Data[,vars<- c("HImaxF_PopW", "quant_HI_county",
                                     "quant_HI_yest_county",
                                     "quant_HI_3d_county",
                                     "quant_HI_fwd_avg_county",
                                     "HI_mean", 
                                     "l.Pop_density", "l.Med.HH.Income",
                                     "year", "dos",
                                     "alert_sum", "More_alerts",
                                     "death_mean_rate", "all_hosp_mean_rate",
                                     "heat_hosp_mean_rate", "broadband.usage",
                                     "Democrat", "Republican",
                                     "pm25", "T_since_alert", "alerts_2wks")]),
                # alert = Data$alert,
                alert_lag1 = Data$alert_lag1,
                alert_lag2 = Data$alert_lag2,
                dow = Data$dow,
                holiday = Data$holiday,
                Zone = Data$BA_zone)

dummy_vars<- with(DF, data.frame(model.matrix(~dow+0), model.matrix(~Zone+0)))
DF<- DF[,-which(names(DF) %in% c("dow", "Zone"))]
DF<- data.frame(DF, dummy_vars)

A<- Data$alert

R_deaths<- -1000*(Data["N"]/Data["Pop.65"])
R_all_hosps<- -1000*(Data["all_hosps"]/Data["total_count"])
R_heat_hosps<- -1000*(Data["heat_hosps"]/Data["total_count"])
R_other_hosps<- -1000*(Data["other_hosps"]/Data["total_count"])

rm(list= ls()[!(ls() %in% c('DF','n_days', 'A', 'budget',
                            'R_deaths', 'R_all_hosps', 
                            'R_heat_hosps', 'R_other_hosps'))])
save.image("data/Small_S-A-R_prepped.RData")



