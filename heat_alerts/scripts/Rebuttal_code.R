
source("heat_alerts/scripts/Evaluation_functions.R")

library(arrow)
library(lubridate)
library(ggplot2)
library(scales)
library(tidyr)
library(dplyr)
library(rpart)
library(rpart.plot)

## Read in results:

Bench<- read.csv("Fall_results/Benchmarks_mixed_constraints_avg_return.csv")
stationary_W<- read.csv("data/Final_30_W.csv")[,-1]
stationary_W<- stationary_W[match(Bench$County, stationary_W$Fips),]

best<- read.csv("Fall_results/December_part-2_Rstr-QHI_RL_avg_return.csv")
Best<- best[which(best$Algo=="a2c" & best$Forecast == "none"),]

#### Reviewer 3 question about evaluation years:

## Print new model runs with 2014-2016 as the eval years:

r_model<- "mixed_constraints"
algo<- "a2c"
forecasts<- "none"
prefix<- "Rebuttal"

sink("run_jobs/Online_tests_short")
for(i in 1:nrow(Best)){
  
  nhl<- Best$NHL[i]
  nhu<- Best$NHU[i]
  if(nhl == 1){
    arch<- paste0("[", nhu, "]")
  }else if(nhl == 2){
    arch<- paste0("[", nhu, ",", nhu, "]")
  }else if(nhl == 3){
    arch<- paste0("[", nhu, ",", nhu, ",", nhu, "]")
  }
  
  cat(paste0("python train_online_rl_sb3.py", " county=", Best$County[i], " algo=", algo,
             " deterministic=false",
             " train_years=[", paste(2006:2013, collapse=","), "]",
             " val_years=[", paste(2014:2016, collapse=","), "]",
             " restrict_days=qhi", " forecasts=", forecasts, " restrict_days.HI_restriction=", Best$OT[i],
             " algo.policy_kwargs.net_arch=", arch, " algo.n_steps=", Best$n_steps[i],
             " model_name=", prefix, "_", algo, "_F-", forecasts, "_Rstr-HI-", Best$OT[i],
             "_arch-", nhl, "-", nhu, "_ns-", Best$n_steps[i],
             "_fips-", Best$County[i], " \n"))
}
sink()

sink("run_jobs/Eval_jobs")
for(i in 1:nrow(Best)){
  county<- Best$County[i]
  cat(paste0("python full_evaluation_sb3.py policy_type=NWS eval.val_years=true eval.match_similar=false ", "county=", county, 
             " r_model=", r_model, " model_name=", prefix, " restrict_days=none",
             " val_years=[", paste(2014:2016, collapse=","), "]",
             "\n"))
}
sink()

## Summarize results:

results<- matrix(0, nrow=nrow(Best), ncol=2)
results<- data.frame(results)
names(results)<- c("County", "Eval")  

for(i in 1:nrow(Best)){
  results[i,]<- c(Best$County[i],
                  avg_return(by_year=FALSE, filename=paste0("Summer_results/ORL_RL_eval_samp-R_obs-W_", "Rebuttal", "_",
                                                           "a2c", "_F-", "none", "_Rstr-HI-", Best$OT[i],
                                                           "_arch-", Best$NHL[i], "-", Best$NHU[i], "_ns-", Best$n_steps[i],
                                                           "_fips-", Best$County[i], "_fips_", Best$County[i], ".csv")))
}




## First examine distribution of heat index across years

data<- read_parquet("data/processed/states.parquet")
Year<- year(data$Date)

Data<- data.frame(Year, QHI = data$quant_HI_county)
ggplot(Data, aes(x = as.factor(Year), y = QHI, group=Year)) +
  geom_boxplot() +
  labs(x = "Year", y = "Quantile of Heat Index") +
  theme_minimal()

## Then compare results across years

results<- matrix(0, nrow=nrow(Best), ncol=4)
results<- data.frame(results)
names(results)<- c("County", "2007", "2011", "2015")  

for(i in 1:nrow(Best)){
  results[i,]<- c(Best$County[i],
                  avg_return(by_year=TRUE, filename=paste0("Summer_results/ORL_RL_eval_samp-R_obs-W_", "December", "_",
         "a2c", "_F-", "none", "_Rstr-HI-", Best$OT[i],
         "_arch-", Best$NHL[i], "-", Best$NHU[i], "_ns-", Best$n_steps[i],
         "_fips-", Best$County[i], "_fips_", Best$County[i], ".csv")))
}

nws<- matrix(0, nrow=nrow(Best), ncol=4)
nws<- data.frame(results)
names(nws)<- c("County", "2007", "2011", "2015") 

for(i in 1:nrow(Best)){
  nws[i,]<- c(Best$County[i],
                  avg_return(by_year=TRUE, filename=paste0("Summer_results/ORL_NWS_eval_samp-R_obs-W_", "mixed_constraints", 
                                                           "_fips_", Best$County[i], ".csv")))
}


## Reshape data to long format
results_long<- results[,2:4] %>%
  pivot_longer(cols = everything(), names_to = "Year", values_to = "Reward")

nws_long<- nws[,2:4] %>%
  pivot_longer(cols = everything(), names_to = "Year", values_to = "Reward")

df_long<- data.frame(Year=results_long$Year, Difference=results_long$Reward - nws_long$Reward)
df_long <- df_long %>%
  mutate(Sqrt_diff = sign(Difference) * sqrt(abs(Difference)))

## Create the boxplot

ggplot(results_long, aes(x = Year, y = Reward)) +
  geom_boxplot() +
  labs(x = "Year", y = "A2C.QHI Rewards") +
  theme_minimal()

ggplot(df_long, aes(x = Year, y = Sqrt_diff)) +
  geom_boxplot() + 
  # stat_summary(fun = mean, geom = "point", shape = 20, color = "red", size = 3) +
  labs(x = "Year", y = "A2C.QHI minus NWS (Signed Square Root)") +
  theme_minimal()


#### Reviewer 1 question about scaling beyond the 30 counties:

this_rl_name<-  "A2C.QHI"

Eval_DOS_bench<- read.csv("Fall_results/Eval_DOS_mixed_constraints_benchmarks.csv")
Eval_SL_bench<- read.csv("Fall_results/Eval_Strk-Ln_mixed_constraints_benchmarks.csv")
Eval_DOS_RL<- read.csv("Fall_results/December_part-2_Eval_DOS_mixed_constraints_RL.csv")
Eval_SL_RL<- read.csv("Fall_results/December_part-2_Eval_Strk-Ln_mixed_constraints_RL.csv")

agg_dos_nws<- aggregate(Value ~ County, data=Eval_DOS_bench[which(Eval_DOS_bench$Policy=="NWS"),],
                        summary)
agg_dos_rl<- aggregate(Value ~ County, data=Eval_DOS_RL[which(Eval_DOS_RL$Policy==this_rl_name),], 
                       summary)
agg_dos_aa<- aggregate(Value ~ County, data=Eval_DOS_bench[which(Eval_DOS_bench$Policy=="Always-QHI"),],
                       summary)

agg_sl_nws<- aggregate(Value ~ County, data=Eval_SL_bench[which(Eval_SL_bench$Policy=="NWS"),],
                       summary)
agg_sl_rl<- aggregate(Value ~ County, data=Eval_SL_RL[which(Eval_SL_RL$Policy==this_rl_name),], 
                      summary)
agg_sl_aa<- aggregate(Value ~ County, data=Eval_SL_bench[which(Eval_SL_bench$Policy=="Always-QHI"),],
                      summary)

m_pos<- match(Bench$County, agg_sl_rl$County)

## Define outcomes for CART:

Diff<- Best$Eval - Bench$NWS
Y<- Diff > 0

## Copied from datautils.py: 
West<- c("AZ", "CA", "CO", "ID", "MT", "NM", "NV", "OR", "WA", "ND", "SD", "NE", "KS")
South<- c("TX", "OK", "AR", "LA", "MS", "AL", "GA", "FL", "TN", "KY", "SC", "NC", 
          "VA", "WV", "VA", "MD", "DE", 
          "NM", "AZ", "CA")

#### Manually select variables for CART analysis (either all or non-modeled set)

## All variables:
df_interpretable<- data.frame(stationary_W[,c("Region", "Med.HH.Income"  
                                          # , "Democrat", "Pop_density", "pm25", "broadband.usage"
                                                   # , "acf_auc_1d" 
                                                   , "acf_auc_3d"
                                                   # , "acf_auc_5d", "acf_auc_7d"
                                       )],
                                   Alerts=stationary_W$Alerts/3,
                                   West=stationary_W$State %in% West,
                                   South=stationary_W$State %in% South
                                  , NWS_SL_avg=agg_sl_nws[m_pos, 2][,4]
)
names(df_interpretable)<- c("Region", "Med. HH Income",
                        # "Democrat", "Pop. Density", "PM2.5", "Broadband Usage",
                                 # "Cum. ACF. 1d", 
                                  "Cum. ACF. 3d",
                                 # "Cum. ACF. 5d", "Cum. ACF. 7d",
                                 "No. Alerts", 
                                 "Western", "Southern"
                                  , "Avg. SL of NWS Alerts"
                        )

CART_df<- df_interpretable

# par(mfrow=c(1,2), mai = c(1, 0.5, 0.1, 0.75))

## Compared to NWS:

# class_fit<- rpart(Y ~ ., data = CART_df, method = "class", model = TRUE
#                   , control = rpart.control(minbucket=3)
# )
# rpart.plot(class_fit, box.palette = 0)

reg_fit<- rpart(Diff ~ ., data = CART_df, method = "anova", model = TRUE
                , control = rpart.control(minbucket=5) 
)
rpart.plot(reg_fit, box.palette = 0)

## Read in the rest of the data:

spatial<- read.csv("data/Spatial_info_orig_scale_all_counties.csv")

spatial_2<- read_parquet("data/processed/spatial_feats.parquet")

region_vars<- c("Cold", "Hot-Dry", "Marine", "Mixed-Dry", "Mixed-Humid", "Very Cold")
Region<- apply(spatial_2[,region_vars],
               MARGIN=1, function(x) region_vars[which(x == 1)])
Region[lengths(Region)==0]<- "Hot-Humid"
spatial_2$Region<- unlist(Region)

Spatial<- inner_join(spatial, spatial_2[,c("fips", "Region")])

## Apply CART results:




