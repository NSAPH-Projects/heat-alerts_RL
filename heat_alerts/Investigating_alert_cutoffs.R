## Setup:
setwd("/n/dominici_nsaph_l3/projects/heat-alerts_mortality_RL")

## Read in the data:

data<- readRDS("data/Data_for_HARL.rds")

## Explore heat index:
aggregate(HImaxF_PopW ~ state, data, function(x) quantile(x, probs = c(0.8, 0.9, 0.97, 1)))

cor(data$HImaxF_PopW, data$HI_lag1) # 0.887
plot(data$HI_lag1, data$HImaxF_PopW)

# Explore alerts and quantiles:

my_quant<- function(df, region_var, split_var #, probs)
){
  regions<- unique(df[, region_var]) # state or county?
  
  q<- rep(0,dim(df)[1])
  
  for(r in regions){
    pos<- which(df[, region_var] == r)
    # r_quants<- quantile(df[pos, split_var], probs)
    # q[pos]<- as.numeric(cut(df[pos, split_var], r_quants))
    percentile<- ecdf(df[pos, split_var])
    q[pos]<- percentile(df[pos, split_var])
  }
  return(q)
}

data$quant_HI<- my_quant(data, "state", "HImaxF_PopW")
data$quant_HI_yest<- my_quant(data, "state", "HI_lag1")

states<- unique(data$state)
alert_stats<- matrix(0, ncol = 4, nrow = length(states))

for(s in states){
  i<- which(states == s)
  df<- data[which(data$state == s),]
  alert_stats[i,1]<- min(df[which(df$alert == 1), "HImaxF_PopW"])
  alert_stats[i,2]<- max(df[which(df$alert == 0), "HImaxF_PopW"])
  alert_stats[i,3]<- min(df[which(df$alert == 1), "quant_HI"])
  alert_stats[i,4]<- max(df[which(df$alert == 0), "quant_HI"])
}

apply(alert_stats, MARGIN=2, summary)

## Explore heat alerts compared to yesterday's HI:

yest_alert_stats<- matrix(0, ncol = 4, nrow = length(states))

for(s in states){
  i<- which(states == s)
  df<- data[which(data$state == s),]
  yest_alert_stats[i,1]<- min(df[which(df$alert == 1), "HI_lag1"])
  yest_alert_stats[i,2]<- max(df[which(df$alert == 0), "HI_lag1"])
  yest_alert_stats[i,3]<- min(df[which(df$alert == 1), "quant_HI_yest"])
  yest_alert_stats[i,4]<- max(df[which(df$alert == 0), "quant_HI_yest"])
}

apply(yest_alert_stats, MARGIN=2, summary)

