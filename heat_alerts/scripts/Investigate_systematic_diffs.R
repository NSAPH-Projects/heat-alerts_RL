
library(rpart)
library(rpart.plot)

Bench<- read.csv("Fall_results/Benchmarks_mixed_constraints_avg_return.csv")
stationary_W<- read.csv("data/Final_30_W.csv")[,-1]
stationary_W<- stationary_W[match(Bench$County, stationary_W$Fips),]

RL_F.q_d10<- read.csv("Fall_results/Main_analysis_trpo_F-Q-D10.csv")
RL_F.none<- read.csv("Fall_results/Main_analysis_trpo_F-none.csv")

Eval_DOS_bench<- read.csv("Fall_results/Eval_DOS_mixed_constraints_benchmarks.csv")
Eval_SL_bench<- read.csv("Fall_results/Eval_Strk-Ln_mixed_constraints_benchmarks.csv")
Eval_DOS_RL<- read.csv("Fall_results/Eval_DOS_mixed_constraints_RL.csv")
Eval_SL_RL<- read.csv("Fall_results/Eval_Strk-Ln_mixed_constraints_RL.csv")

agg_dos_nws<- aggregate(Value ~ County, data=Eval_DOS_bench[which(Eval_DOS_bench$Policy=="NWS"),],
                        summary)
agg_dos_rl<- aggregate(Value ~ County, data=Eval_DOS_RL[which(Eval_DOS_RL$Policy=="TRPO.QHI"),], summary)

agg_sl_nws<- aggregate(Value ~ County, data=Eval_SL_bench[which(Eval_SL_bench$Policy=="NWS"),],
                        summary)
agg_sl_rl<- aggregate(Value ~ County, data=Eval_SL_RL[which(Eval_SL_RL$Policy=="TRPO.QHI"),], summary)

m_pos<- match(Bench$County, agg_sl_rl$County)

# Diff<- RL_F.q_d10$Eval - Bench$NWS
Diff<- RL_F.none$Eval - Bench$NWS
Y<- factor(Diff > 0)

# Diff.a<- RL_F.q_d10$Eval - Bench$AA_QHI
Diff.a<- RL_F.none$Eval - Bench$AA_QHI
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
                                     "Alerts", 
                                     # "Mean_Eff", 
                                     "SD_Eff")],
                     # QHI_OT=RL_F.q_d10$OT, 
                     West=stationary_W$State %in% West,
                     South=stationary_W$State %in% South,
                     # NWS_DOS_med=agg_dos_nws[m_pos, 2][,3],
                     RL_DOS_med=agg_dos_rl[m_pos, 2][,3],
                     Diff_DOS_med=agg_dos_nws[m_pos, 2][,3] - agg_dos_rl[m_pos, 2][,3],
                     NWS_SL_avg=agg_sl_nws[m_pos, 2][,4]
                     )
CART_df<- data.frame(stationary_W[,c("Region", "Pop_density", "Med.HH.Income",
                                     # "Democrat", "broadband.usage", "pm25",
                                     "Alerts")],
                     West=stationary_W$State %in% West,
                     South=stationary_W$State %in% South,
                     NWS_DOS_med=agg_dos_nws[m_pos, 2][,3],
                     NWS_SL_avg=agg_sl_nws[m_pos, 2][,4]
)
paste(shQuote(names(CART_df)), collapse=", ")

par(mfrow=c(1,2))

## Compared to NWS:
class_fit<- rpart(Y ~ ., data = CART_df, method = "class", model = TRUE
      , control = rpart.control(max.depth=2)
      )
rpart.plot(class_fit, box.palette = 0)

reg_fit<- rpart(Diff ~ ., data = CART_df, method = "anova", model = TRUE
                  , control = rpart.control(max.depth=2) 
)
rpart.plot(reg_fit, box.palette = 0)

## Compared to AA.QHI:
class_fit<- rpart(Y.a ~ ., data = CART_df, method = "class", model = TRUE
                  , control = rpart.control(max.depth=2)
)
rpart.plot(class_fit, box.palette = 0)

reg_fit<- rpart(Diff.a ~ ., data = CART_df, method = "anova", model = TRUE
                , control = rpart.control(max.depth=2)
)
rpart.plot(reg_fit, box.palette = 0)





