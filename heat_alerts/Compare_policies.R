library(ggplot2)

## Prep data:

n_days<- 153

load("data/Small_S-A-R_prepped.RData")

behavior<- A
# new_pol<- read.csv("Fall_results/DQN_12-29_hosps_constrained_policy.csv")$policy
new_pol<- read.csv("Fall_results/DQN_12-29_deaths_constrained_policy.csv")$policy

## Compare policies:

behavior_pos<- which(behavior == 1)
new_pos<- which(new_pol == 1)

DF_1<- data.frame(apply(DF, MARGIN = 2, function(y) c(y[behavior_pos], y[new_pos])),
                  Policy = c(rep("Behavior", length(behavior_pos)),
                             rep("New", length(new_pos))))

table(DF_1[which(DF_1$Policy=="Behavior"),"holiday"])/length(which(DF_1$Policy=="Behavior"))
table(DF_1[which(DF_1$Policy=="New"),"holiday"])/length(which(DF_1$Policy=="New"))

ggplot(DF_1, aes(x=as.numeric(dos), fill=Policy)) +
  geom_histogram(aes(y = ..density..),
    position = "identity", alpha = 0.4, bins = 50) + 
  xlab("Day of Summer")

# ggplot(DF_1, aes(x=dow, fill=Policy)) +
#   geom_histogram(position = "identity", alpha = 0.4, stat = "count") + 
#   xlab("Day of Week")

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

ggplot(DF_1, aes(x=as.numeric(quant_HI_yest_county), fill=Policy)) +
  geom_histogram(aes(y = ..density..),
                 position = "identity", alpha = 0.4, bins = 50) + 
  xlab("County Quantile Yesterday")

ggplot(DF_1, aes(x=as.numeric(quant_HI_fwd_avg_county), fill=Policy)) +
  geom_histogram(aes(y = ..density..),
                 position = "identity", alpha = 0.4, bins = 50) + 
  xlab("County Quantile of 3-day Forecast Average")

ggplot(DF_1, aes(x=as.numeric(quant_HI_3d_county), fill=Policy)) +
  geom_histogram(aes(y = ..density..),
                 position = "identity", alpha = 0.4, bins = 50) + 
  xlab("County Quantile of Past 3 Days Average")

# ggplot(DF_1, aes(x=BA_zone, fill=Policy)) +
#   geom_histogram(position = "identity", alpha = 0.4, stat = "count") + 
#   xlab("Climate Zone")

ggplot(DF_1, aes(x=as.numeric(l.Med.HH.Income), fill=Policy)) +
  geom_histogram(aes(y = ..density..),
                 position = "identity", alpha = 0.4, bins = 50) + 
  xlab("Median Household Income")

# ggplot(DF_1, aes(x=holiday, fill=Policy)) +
#   geom_histogram(position = "identity", alpha = 0.4, stat = "count") + 
#   xlab("Holiday")




