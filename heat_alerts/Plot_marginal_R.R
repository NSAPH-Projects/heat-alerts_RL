# library(caret)

load("data/Small_S-A-R_prepped.RData")

# # ranger_model<- readRDS("Fall_results/Rewards_VarImp_deaths.rds")
# # ranger_model<- readRDS("Fall_results/Rewards_deaths.rds")
# # ranger_model<- readRDS("Fall_results/Rewards_all_hosps.rds")
# # ranger_model<- readRDS("Fall_results/Rewards_other_hosps.rds")
# ranger_model<- readRDS("Fall_results/Rewards_heat_hosps.rds")
# 
# R0<- ranger_model$finalModel$predictions
# R0[which(A == 1)]<- predict(ranger_model, data.frame(DF[which(A == 1),], alert = 0))
# R1<- ranger_model$finalModel$predictions
# R1[which(A == 0)]<- predict(ranger_model, data.frame(DF[which(A == 0),], alert = 1))

R<- read.csv("Fall_results/R_11-27_other-hosps.csv")
# R<- read.csv("Fall_results/R_11-27_deaths.csv")
R0<- as.vector(unlist(R["X0"]))
R1<-  as.vector(unlist(R["X1"]))

Diffs<- R1 - R0
mean(Diffs)
high_quant_HI<- which(DF$quant_HI_county > 1.65) # 2.27%
mean(Diffs[high_quant_HI])

for(z in names(DF)[26:32]){
  print(z)
  pos<- which(DF[,z] == 1)
  print(mean(Diffs[intersect(pos, high_quant_HI)]))
}

quants<- quantile(DF$quant_HI_county, probs=seq(0.75, 1, 0.05))
res<- c()

for(i in 2:length(quants)){
  res<- append(res, mean(Diffs[DF$quant_HI_county < quants[i] &
                                 DF$quant_HI_county >= quants[i-1]]))
}

## Look at difference across temps:

plot(DF$HImaxF_PopW[high_quant_HI], Diffs[high_quant_HI])

for(z in names(DF)[26:32]){
  print(z)
  pos<- which(DF[,z] == 1)
  plot(DF$HImaxF_PopW[intersect(pos, high_quant_HI)], 
       Diffs[intersect(pos, high_quant_HI)], 
       main = z)
}

## Make marginal plots...

n_pts<- 10

# marg_template<- matrix(rep(colMeans(DF[,1:15]), each=n_pts*2), nrow = n_pts*2)
# pos<- which(DF$quant_HI_fwd_avg_county > 1.5 & DF$quant_HI_3d_county < 0)

DF$alert<- A
zones<- names(DF)[26:32]

for(z in zones){ # climate zones
  print(z)
  
  pos<- which(DF[,z] == 1 & DF$HImaxF_PopW >= 0.5)
  marg_template<- matrix(rep(colMeans(DF[pos,1:15]), each=n_pts*2), nrow = n_pts*2)
  marg_template<- data.frame(marg_template, 
                             alert_lag1 = 0, alert_lag2 = 0, holiday = 0,
                             dowFriday = 0, dowMonday = 0, dowSaturday = 0,
                             dowSunday = 0, dowThursday = 1, dowTuesday = 0, 
                             dowWednesday = 0, ZoneCold = 0, ZoneHot.Dry = 0, 
                             ZoneHot.Humid = 0, ZoneMarine = 0, ZoneMixed.Dry = 0,
                             ZoneMixed.Humid = 0, ZoneVery.Cold = 0, 
                             alert = c(rep(0,n_pts), rep(1,n_pts)))
  marg_template$holiday<- as.factor(marg_template$holiday)
  names(marg_template)<- names(DF)
  
  this_marg<- marg_template
  this_marg[,z]<- 1
  this_marg$HImaxF_PopW<- rep(seq(0.5, 5.2, length.out = n_pts), 2) # 90 to 145 F
  mod0<- lm(quant_HI_county ~ HImaxF_PopW, DF[pos,])
  this_marg$quant_HI_county<- rep(seq(0.5, 5.2, length.out = n_pts), 2)*mod0$coefficients[2] + mod0$coefficients[1]
  mod1<- lm(quant_HI_yest_county ~ HImaxF_PopW, DF[pos,])
  this_marg$quant_HI_yest_county<- rep(seq(0.5, 5.2, length.out = n_pts), 2)*mod1$coefficients[2] + mod1$coefficients[1]
  
  this_marg$R<- predict(ranger_model, this_marg)
  
  this_marg$alert<- as.factor(this_marg$alert)
  this_marg$HImaxF_PopW<- this_marg$HImaxF_PopW*11.71149 + 83.97772 # from S, Train_smaller-for-Python.csv
  ggplot(this_marg, aes(x = HImaxF_PopW, y = R, color = alert)) + 
    geom_line() + ggtitle(z)
}





