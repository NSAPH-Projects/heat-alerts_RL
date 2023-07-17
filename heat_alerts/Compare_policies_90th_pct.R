library(ggplot2)

data<- read.csv("data/Summ23_Train_smaller-for-Python.csv")
n_counties<- nrow(data)/(153*11)
data$index<- rep(1:(n_counties*11), each = 153)
data$weekend<- data$dow %in% c("Saturday", "Sunday")

pct_90<- data[data$quant_HI_county >= 0.9,]
terminals_90pct<- read.csv("data/Pct_90_eligible_terminals.csv")

behavior<- pct_90$alert # sum = 26,317

# new_pol<- read.csv("Policies/Policy_vanilla_DQN_small-S_lr1e-2_B.csv")[,2] # sum = 0
new_pol<- read.csv("Policies/Policy_Double_DQN_fips-6085_MR-90pct_seed-1.csv")[,2] 
sum_alerts_DD<- sum(new_pol)
sum(new_pol)
new_pol<- read.csv("Policies/Policy_CPQ_observed-alerts_fips-6085_MR-90pct_seed-1.csv")[,2] 
sum_alerts_CPQ<- sum(new_pol)
sum(new_pol)

models<- c("Double_DQN", "CPQ_observed-alerts")
# seeds<- c(1,2,4:10)
# seeds<- (1:5)[-3]
seeds<- 1:3

for(m in models){
  for(s in seeds){
    new_pol<- read.csv(paste0("Policies/Policy_", m, "_fips-6085_MR-90pct_seed-", s, ".csv"))[,2]
    # print(paste(m,s,sum(new_pol)))
    if(m == "Double_DQN"){
      sum_alerts_DD<- append(sum_alerts_DD, sum(new_pol))
    }else{
      sum_alerts_CPQ<- append(sum_alerts_CPQ, sum(new_pol))
    }
    new_pol<- read.csv(paste0("Policies/Policy_", m, "_fips-6085_OR-90pct_seed-", s, ".csv"))[,2]
    # print(paste(m,s,sum(new_pol)))
    if(m == "Double_DQN"){
      sum_alerts_DD<- append(sum_alerts_DD, sum(new_pol))
    }else{
      sum_alerts_CPQ<- append(sum_alerts_CPQ, sum(new_pol))
    }
  }
}

sum_alerts_DD[-1]
sum_alerts_CPQ[-1]

ep_alerts<- aggregate(alert ~ index, data=pct_90, sum)$alert
pct_90$new_pol<- new_pol
new_ep_alerts<- aggregate(new_pol ~ index, data=pct_90, sum)$new_pol
budgets<- pct_90$alert_sum[which(terminals_90pct == 1)]
summary(budgets)
summary(ep_alerts)
summary(new_ep_alerts)

behavior_pos<- which(behavior == 1)
new_pos<- which(new_pol == 1)

DF_1<- data.frame(apply(pct_90, MARGIN = 2, function(y) c(y[behavior_pos], y[new_pos])),
                  Policy = c(rep("Behavior", length(behavior_pos)),
                             rep("New", length(new_pos))))

table(DF_1[which(DF_1$Policy=="Behavior"),"holiday"])/length(which(DF_1$Policy=="Behavior"))
table(DF_1[which(DF_1$Policy=="New"),"holiday"])/length(which(DF_1$Policy=="New"))

table(DF_1[which(DF_1$Policy=="Behavior"),"weekend"])/length(which(DF_1$Policy=="Behavior"))
table(DF_1[which(DF_1$Policy=="New"),"weekend"])/length(which(DF_1$Policy=="New"))

ggplot(DF_1, aes(x=as.numeric(dos), fill=Policy)) +
  geom_histogram(aes(y = ..density..),
                 position = "identity", alpha = 0.4, bins = 50) + 
  xlab("Day of Summer")

ggplot(DF_1, aes(x=as.numeric(year), fill=Policy)) +
  geom_histogram(aes(y = ..density..),
                 position = "identity", alpha = 0.4) + 
  xlab("Year")

ggplot(DF_1, aes(x=as.numeric(HImaxF_PopW), fill=Policy)) +
  geom_histogram(aes(y = ..density..),
                 position = "identity", alpha = 0.4, bins = 50) + 
  xlab("Maximum Heat Index That Day")

ggplot(DF_1, aes(x=as.numeric(quant_HI_county), fill=Policy)) +
  geom_histogram(aes(y = ..density..),
                 position = "identity", alpha = 0.4, bins = 50) + 
  xlab("County Quantile That Day")

ggplot(DF_1, aes(x=as.numeric(HI_mean), fill=Policy)) +
  geom_histogram(aes(y = ..density..),
                 position = "identity", alpha = 0.4, bins = 50) + 
  xlab("Rolling Average of Max. HI Since 5/1")

ggplot(DF_1, aes(x=as.numeric(all_hosp_mean_rate)*1000, fill=Policy)) +
  geom_histogram(aes(y = ..density..),
                 position = "identity", alpha = 0.4, bins = 50) + 
  xlab("Rolling Average Rate of Hosps. Since 5/1 per 1,000")

ggplot(DF_1, aes(x=as.numeric(T_since_alert), fill=Policy)) +
  geom_histogram(aes(y = ..density..),
                 position = "identity", alpha = 0.4, bins = 50) + 
  xlab("Time Since Last Alert")

ggplot(DF_1, aes(x=as.numeric(l.Med.HH.Income), fill=Policy)) +
  geom_histogram(aes(y = ..density..),
                 position = "identity", alpha = 0.4, bins = 50) + 
  xlab("Log of Median Household Income")

ggplot(DF_1, aes(x=as.numeric(l.Pop_density), fill=Policy)) +
  geom_histogram(aes(y = ..density..),
                 position = "identity", alpha = 0.4, bins = 50) + 
  xlab("Log of Population Density")




