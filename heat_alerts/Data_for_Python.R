library(readr)

load("data/Train-Test.RData")

n_days<- 153

S_inds<- as.numeric(row.names(Train[-seq(n_days, nrow(Train), n_days),]))
Top3rd<- Train[which(Train$Population>= 65000),]
S_t3_inds<- as.numeric(row.names(Top3rd[-seq(n_days, nrow(Top3rd), n_days),]))

# write_csv(data.frame(S_inds), "data/S_training_indices.csv")
# write_csv(data.frame(S_t3_inds), "data/S_t3_training_indices.csv")

# write_csv(Top3rd, "data/Train_smaller-for-Python.csv")


################ Also create set for R modeling:

data<- Top3rd

budget<- data[which(data$dos == 153), "alert_sum"]
Budget<- rep(budget, each = n_days)
data$More_alerts<- Budget - data$alert_sum
Data<- data[-seq(n_days, nrow(data), n_days),]

DF<- data.frame(scale(Data[,vars<- c("HImaxF_PopW", "quant_HI_county",
                                     "quant_HI_yest_county",
                                     "quant_HI_3d_county",
                                     "quant_HI_fwd_avg_county",
                                     "HI_mean", 
                                     "l.Pop_density", "l.Med.HH.Income",
                                     "year", "dos",
                                     "alert_sum", "More_alerts",
                                     "death_mean_rate", "all_hosp_mean_rate")]),
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

R_hosps<- -1000*(Data["all_hosps"]/Data["total_count"])
R_deaths<- -1000*(Data["N"]/Data["Pop.65"])

rm(list= ls()[!(ls() %in% c('DF','n_days', 'A',
                            'R_hosps', 'R_deaths', 'budget'))])
save.image("data/Small_S-A-R_prepped.RData")



