
library(rpart)
library(rpart.plot)

Bench<- read.csv("Fall_results/Benchmarks_mixed_constraints_avg_return.csv")
stationary_W<- read.csv("data/Final_30_W.csv")[,-1]
stationary_W<- stationary_W[match(Bench$County, stationary_W$Fips),]

RL_F.q_d10<- read.csv("Fall_results/Main_analysis_trpo_F-Q-D10.csv")
RL_F.none<- read.csv("Fall_results/Main_analysis_trpo_F-none.csv")

Diff<- RL_F.q_d10$Eval - Bench$NWS
# Diff<- RL_F.none$Eval - Bench$NWS
Y<- factor(Diff > 0)

Diff.a<- RL_F.q_d10$Eval - Bench$AA_QHI
# Diff.a<- RL_F.none$Eval - Bench$AA_QHI
Y.a<- factor(Diff.a > 0)

# Eval_DOS_RL<- read.csv("Fall_results/Eval_DOS_mixed_constraints_RL.csv")
# Eval_DOS_NWS<- read.csv("Fall_results/Eval_DOS_mixed_constraints_benchmarks.csv")
# Eval_SL_RL<- read.csv("Fall_results/Eval_Strk-Ln_mixed_constraints_RL.csv")
# Eval_SL_NWS<- read.csv("Fall_results/Eval_Strk-Ln_mixed_constraints_benchmarks.csv")

West<- c("AZ", "CA", "CO", "ID", "MT", "NM", "NV", "OR", "WA", "ND", "SD", "NE", "KS")
South<- c("TX", "OK", "AR", "LA", "MS", "AL", "GA", "FL", "TN", "KY", "SC", "NC", 
          "VA", "WV", "VA", "MD", "DE", 
          "NM", "AZ", "CA")

CART_df<- data.frame(stationary_W[,c("Region", "Pop_density", "Med.HH.Income",
                                     # "Democrat", "broadband.usage", "pm25",
                                     "Alerts", "SD_Eff", "Offset")],
                     # QHI_OT=RL_F.q_d10$OT, 
                     West=stationary_W$State %in% West,
                     South=stationary_W$State %in% South)
# CART_df<- data.frame(stationary_W$Region, West=stationary_W$State %in% West,
#                      South=stationary_W$State %in% South,
#                      Hi_Pop_Dens=stationary_W$Pop_density >= median(stationary_W$Pop_density),
#                      Hi_HH_Inc=stationary_W$Med.HH.Income >= median(stationary_W$Med.HH.Income),
#                      Alerts=stationary_W$Alerts, SD_Eff=stationary_W$SD_Eff,
#                      Mean_NOHR=stationary_W$Offset)


class_fit<- rpart(Y ~ ., data = CART_df, method = "class", model = TRUE
      , control = rpart.control(minbucket = 5)
      )
rpart.plot(class_fit, box.palette = 0)

reg_fit<- rpart(Diff ~ ., data = CART_df, method = "anova", model = TRUE
                  , control = rpart.control(minbucket = 5)
)
rpart.plot(reg_fit, box.palette = 0)







