library(readr)

load("data/Train-Test.RData")

n_days<- 153

S_inds<- as.numeric(row.names(Train[-seq(n_days, nrow(Train), n_days),]))
Top3rd<- Train[which(Train$Population>= 65000),]
S_t3_inds<- as.numeric(row.names(Top3rd[-seq(n_days, nrow(Top3rd), n_days),]))

write_csv(data.frame(S_inds), "data/S_training_indices.csv")
write_csv(data.frame(S_t3_inds), "data/S_t3_training_indices.csv")

write_csv(Top3rd, "data/Train_smaller-for-Python.csv")

#### Get summary stats for paper:

data<- data.frame(Top3rd)
eps<- rep(1:(11*596), each=153)

DF<- data.frame(nohr=data$other_hosps, eps)
ep_nohr<- aggregate(nohr ~ eps, DF, sum)
dens<- data$total_count[seq(1, nrow(data), 153)]

ep_NOHR<- ep_nohr$nohr/dens
round(summary(ep_NOHR*1000),1)

sum_alerts<- data$alert_sum[seq(153, nrow(data), 153)]
round(summary(sum_alerts), 1)

alert_dos<- data$dos[which(data$alert == 1)]
round(summary(alert_dos), 1)

pct_90<- data$quant_HI_county >= 0.9
ep_90th<- aggregate(pct_90~eps, data.frame(pct_90, eps), sum)
round(summary(ep_90th$pct_90), 1)

pct90_dos<- data$dos[pct_90]
round(summary(pct90_dos), 1)

alert_HI<- data$HImaxF_PopW[which(data$alert == 1)]
round(summary(alert_HI), 1)

alert_HI_quant<- data$quant_HI_county[which(data$alert == 1)]
round(summary(alert_HI_quant)*100, 1)

sum(data$alert == 0 & pct_90)/sum(pct_90)

sum(data$alert == 1 & !pct_90)/sum(data$alert)

#### Select "eligible" days based on heat index:

fips<- unique(data$fips)

# pct90<- c()
# for(i in fips){
#   pos<- which(data$fips == i)
#   pct90<- append(pct90, quantile(data$HImaxF_PopW[pos], 0.9))
# }
# Pct_90<- rep(pct90, each = 153*11)
# Pct_90_eligible<- data$HImaxF_PopW >= Pct_90 # oops, actually could have gotten this from the data$quant_HI_county variable
Pct_90_eligible<- data$quant_HI_county >= 0.9
write_csv(data.frame(Pct_90_eligible), "data/Pct_90_eligible.csv")

sums<- unname(tapply(Pct_90_eligible, (seq_along(Pct_90_eligible)-1) %/% n_days, sum))

New_terminals<- c()
episodes<- seq(153, length(Pct_90), 153)
for(s in 1:length(episodes)){
  el<- sum(Pct_90_eligible[(153*(s-1)+1):(153*s)])
  if(el > 0){
    New_terminals<- append(New_terminals, 
                           c(rep(0, el-1),1))
  }
}

write_csv(data.frame(New_terminals), "data/Pct_90_eligible_terminals.csv")

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
# Data<- data[-seq(n_days, nrow(data), n_days),] # Not using this for RL
Data<- data
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
                                     "heat_hosp_mean_rate", "all_hosp_2wkMA_rate",
                                     "heat_hosp_2wkMA_rate", "all_hosp_3dMA_rate", 
                                     "heat_hosp_3dMA_rate", "age_65_74_rate",
                                     "age_75_84_rate", "dual_rate",
                                     "Democrat", "Republican", "broadband.usage",
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



