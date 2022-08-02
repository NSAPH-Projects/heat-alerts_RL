# library(lubridate)
library(matrixStats)
library(dplyr)

#### Setup:

# summer<- readRDS("data/Final_data_for_HARL.rds")
# 
# n_counties<- length(unique(summer$GEOID))
# n_years<- 11
# n_days<- 153
# 
# summer$dos<- rep(1:n_days, n_counties*n_years) # same idea as "time" variable
# 
# ## Need to adjust alert_sum to account for 11 alerts from April (which I'm not including):
# summer$day<- day(summer$Date)
# april_alerts<- summer[which(summer$month==5 & summer$day == 1 & summer$alert_sum > 0), c("GEOID", "year")]
# summer[which(summer$year == 2009 & summer$GEOID %in% april_alerts$GEOID),"alert_sum"]<- summer[which(summer$year == 2009 & summer$GEOID %in% april_alerts$GEOID),"alert_sum"] - 1
# 
# last_day<- summer[which(summer$month == 9 & summer$day == 30),] # or look at dos
# 
# summer$holiday<- factor(summer$holiday)
# # summer<- as.data.frame(summer)
# 
# #### Subset out the data we want to use:
# 
# over_10k<- summer[which(summer$Population >= 10000),] # 2261 counties
# fips<- unique(over_10k$GEOID)
# too_few<- setdiff(unique(last_day$GEOID), fips)
# n_test<- round(0.1*length(fips))
# 
# set.seed(321)
# validation_fips<- sample(fips, n_test, replace = FALSE)
# test_fips<- sample(setdiff(fips, validation_fips), n_test, replace = FALSE)
# train_fips<- setdiff(fips, c(validation_fips, test_fips))
# 
# Train<- over_10k[which(over_10k$GEOID %in% train_fips),]
# Validation<- over_10k[which(over_10k$GEOID %in% validation_fips),]
# Test<- over_10k[which(over_10k$GEOID %in% test_fips),]
# 
# save(Train, Validation, Test, file="data/Train-Valid-Test.RData")

load("data/Train-Valid-Test.RData")

n_counties<- length(unique(Train$GEOID))
n_years<- 11
n_days<- 153

#### Evaluate Q:

eval_Q<- function(S, Q_model){
  
  data<- data.frame(S, A = 0)
  Q0<- predict(Q_model, data)
  
  data<- data.frame(S, A = 1)
  Q1<- predict(Q_model, data)
  
  return(cbind(Q0, Q1))
}

#### Choose action:

choose_a<- function(Q_mat){
   
  max.Q<- rowMaxs(Q_mat)
  argmax<- max.col(Q_mat, ties.method = "first") - 1

  argmax[over_pos]<- 0
  max.Q[over_pos]<- Q_mat[over_pos, 1]
  
  return(cbind(argmax, max.Q))
}

#### Final setup:

# budget<- last_day$alert_sum

## Only look at data with one or more heat alerts?
# positive<- which(budget > 0)
# budget<- budget[positive] # updated alert budget

gamma<- 0.999
A<- Train[-seq(n_days, nrow(Train), n_days),"alert"]#[positive]
R<- (-1*(Train$N*100000/Train$Pop.65))[-seq(n_days, nrow(Train), n_days)]#[positive]
ep_end<- rep(c(rep(0,n_days-2),1),length(R)/(n_days-1))#[positive] # end-of-episode indicator

## Decide what to include in the states...

# States.1<- summer[, c("HImaxF_PopW", "HI_lag1",
#                       "HI_3days", "HI_fwd_avg",
#                       "BA_zone", "Pop_density", "Med.HH.Income",
#                       "year", "dos", "holiday", # "Holiday", "dow",
#                       "alert_sum")]
# 
# States.1<- summer[, c("quant_HI_county", "quant_HI_yest_county",
#                       "quant_HI_3d_county", "quant_HI_fwd_avg_county",
#                       "BA_zone", "Pop_density", "Med.HH.Income",
#                       "year", "dos", "holiday", # "Holiday", "dow",
#                       "alert_sum")]

States.1<- Train[, c("HImaxF_PopW", "quant_HI_county", "quant_HI_yest_county",
                      "quant_HI_3d_county", "quant_HI_fwd_avg_county",
                      "BA_zone", "Pop_density", "Med.HH.Income",
                      "year", "dos", "holiday", # "Holiday", "dow",
                      "alert_sum")]

States<- States.1[-seq(n_days, nrow(Train), n_days),]
States.1<- States.1[-seq(1, nrow(Train), n_days),]

## Normalize state variables: (careful with non-numeric vars)
S<- States %>% mutate_if(is.numeric, scale)
S.1<- States.1 %>% mutate_if(is.numeric, scale)

# S[,"dow"]<- as.factor(S[,"dow"])
# S.1[,"dow"]<- as.factor(S.1[,"dow"])

ID<- rep(1:(n_counties*n_years), each = (n_days-1))#[positive]

# over_budget<- rep(0, nrow(S))
# 
# for(i in 1:max(ID)){
#   pos<- which(ID == i)
#   b<- which(States[pos,"alert_sum"] == budget[i])[1]
#   over_budget[pos][(b+1):(n_days-1)]<- 1
# }
# 
# saveRDS(over_budget, "data/Over_budget.rds")

over_budget<- readRDS("data/Over_budget.rds")
original_rows<- 1:(n_days*n_years*2837)
Over_budget<- data.frame(over_budget, row=original_rows[-seq(n_days, length(original_rows), n_days)])
Over_budget<- Over_budget[which(Over_budget$row %in% 
                                  as.numeric(row.names(States))),] # adjust for training set
over_pos<- which(Over_budget$over_budget==1)


#### Q-Learning loop:
Target<- R

# batch_size<- round(0.1*length(Target))
batch_size<- 300000

iter<- 1

s<- Sys.time()
Q_model<- glm(Target ~ A*., data = data.frame(Target, S, A),
              family = gaussian, 
              subset = sample(1:length(Target), batch_size, replace = FALSE),
              control = list(maxit = 1), warning = FALSE)
# e<- Sys.time()
# e-s
# Q_model<- lm(Target ~ A*., data = data.frame(Target, S[positive,], A))
old_coefs<- rnorm(length(coef(Q_model)))

sink("Aug_results/First_SGD.txt")

while(sum((coef(Q_model)-old_coefs)^2, na.rm = TRUE) > 0.1 & iter < 1000){
  
  Q_mat<- eval_Q(S.1, Q_model)
  # Q_mat<- eval_Q(S.1[positive,], Q_model)
  
  AMQ<- choose_a(Q_mat)
  
  Target<- R + gamma*(1-ep_end)*AMQ[,2] 
  
  old_coefs<- coef(Q_model)
  # s<- Sys.time()
  Q_model<- glm(Target ~ A*., data = data.frame(Target, S, A),
                family = gaussian, 
                subset = sample(1:length(Target), batch_size, replace = FALSE),
                control = list(maxit = 1), start = old_coefs, warning = FALSE)
  # e<- Sys.time()
  # e-s
  # Q_model<- lm(Target ~ A*., data = data.frame(Target, S[positive,], A))
  
  print(iter)
  print(sum((coef(Q_model)-old_coefs)^2))
  print(summary(Target))
  print(Q_model$coefficients)
  iter<- iter+1
}

e<- Sys.time()
e-s

sink()

saveRDS(Q_model, "Lm_8-2.rds")
saveRDS(Target, "Targets_8-2.rds")

## See restricted results:
Q_mat.S<- eval_Q(S[positive,], Q_model)
policy<- choose_a(Q_mat.S)[,1]

## See on non-restricted results:
A<- Actions
Budget<- rep(last_day$alert_sum, each = (n_days - 1))
ID<- rep(1:(n_counties*n_years), each = (n_days-1))
Q_mat<- eval_Q(S.1, Q_model)
policy<- choose_a(Q_mat)[,1]

#### Inspect:

# hist(S[positive,2][which(policy==1)])

hist(S[which(policy==1),2])

hist(S[which(policy==1 & A==0),"quant_HI_county"])
hist(S[which(policy==0 & A==1),"quant_HI_county"])

hist(S[which(policy==1 & A==0),"quant_HI_yest_county"])
hist(S[which(policy==0 & A==1),"quant_HI_yest_county"])

hist(S[which(policy==1 & A==0),"quant_HI_3d_county"])
hist(S[which(policy==0 & A==1),"quant_HI_3d_county"])

hist(S[which(policy==1 & A==0),"quant_HI_fwd_avg_county"])
hist(S[which(policy==0 & A==1),"quant_HI_fwd_avg_county"])

hist(S[which(policy==1 & A==0),"dos"])
hist(S[which(policy==0 & A==1),"dos"])

S[which(policy==1 & A==0),]



