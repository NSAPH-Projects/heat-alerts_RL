library(caret)

load("data/Small_S-A-R_prepped.RData")

DF$alert<- A


## Make marginal plots

ranger_model<- readRDS("Fall_results/Rewards_deaths.rds")

n_pts<- 10

# marg_template<- matrix(rep(colMeans(DF[,1:15]), each=n_pts*2), nrow = n_pts*2)
# pos<- which(DF$quant_HI_fwd_avg_county > 1.5 & DF$quant_HI_3d_county < 0)

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





