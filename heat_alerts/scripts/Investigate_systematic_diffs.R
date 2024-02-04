
library(rpart)
library(rpart.plot)

Bench<- read.csv("Fall_results/Benchmarks_mixed_constraints_avg_return.csv")
stationary_W<- read.csv("data/Final_30_W.csv")[,-1]
stationary_W<- stationary_W[match(Bench$County, stationary_W$Fips),]

rl_results1<- read.csv("Fall_results/December_Rstr-QHI_RL_avg_return.csv")
rl_results2<- read.csv("Fall_results/December_part-2_Rstr-QHI_RL_avg_return.csv")

trpo_results<- rl_results1[which(rl_results1$Algo=="trpo" & rl_results1$Forecast=="none"),]
dqn_results<- rl_results1[which(rl_results1$Algo=="dqn" & rl_results1$Forecast=="none"),]
a2c_results<- rl_results2[which(rl_results2$Algo=="a2c" & rl_results2$Forecast=="none"),]
qrdqn_results<- rl_results2[which(rl_results2$Algo=="qrdqn" & rl_results2$Forecast=="none"),]

this_rl_name<-  "A2C.QHI" # "TRPO.QHI"

Eval_DOS_bench<- read.csv("Fall_results/Eval_DOS_mixed_constraints_benchmarks.csv")
Eval_SL_bench<- read.csv("Fall_results/Eval_Strk-Ln_mixed_constraints_benchmarks.csv")
# Eval_DOS_RL<- read.csv("Fall_results/December_Eval_DOS_mixed_constraints_RL.csv")
# Eval_SL_RL<- read.csv("Fall_results/December_Eval_Strk-Ln_mixed_constraints_RL.csv")
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

Diff<- a2c_results$Eval - Bench$NWS
# Y<- factor(Diff > 0)
all_results<- data.frame(NWS=Bench$NWS, AA.QHI=Bench$AA_QHI, A2C.QHI=a2c_results$Eval, TRPO.QHI=trpo_results$Eval)
names(all_results)<- c("NWS", "AA.QHI", "A2C.QHI (RL)", "TRPO.QHI (RL)")
Y<- names(all_results)[apply(all_results, MARGIN=1, which.max)]
Y_rl<- c("A2C.QHI (RL)", "TRPO.QHI (RL)")[apply(all_results[,c("A2C.QHI (RL)", "TRPO.QHI (RL)")], MARGIN=1, which.max)]

Diff.a<- a2c_results$Eval - Bench$AA_QHI
# Y.a<- factor(Diff.a > 0)

## Copied from datautils.py: 
West<- c("AZ", "CA", "CO", "ID", "MT", "NM", "NV", "OR", "WA", "ND", "SD", "NE", "KS")
South<- c("TX", "OK", "AR", "LA", "MS", "AL", "GA", "FL", "TN", "KY", "SC", "NC", 
          "VA", "WV", "VA", "MD", "DE", 
          "NM", "AZ", "CA")

#### Manually select variables for CART analysis (either all or non-modeled set)

## All variables:
df_most_interpretable<- data.frame(stationary_W[,c("Region", "Med.HH.Income",   
                                                   "acf_auc_1d", "acf_auc_3d", 
                                                   "acf_auc_5d", "acf_auc_7d")],
                                   Alerts=stationary_W$Alerts/3,
                                   West=stationary_W$State %in% West,
                                   South=stationary_W$State %in% South
                                   )
names(df_most_interpretable)<- c("Region", "Med. HH Income", 
                                 "Cum. ACF. 1d", "Cum. ACF. 3d",
                                 "Cum. ACF. 5d", "Cum. ACF. 7d",
                                 "No. Alerts", 
                                 "Western", "Southern")

df_alert_related<- data.frame(SD_Eff=stationary_W$SD_Eff,
                              Alerts=stationary_W$Alerts/3,
                              RL_DOS_med=agg_dos_rl[m_pos, 2][,3],
                              # NWS_DOS_med=agg_dos_nws[m_pos, 2][,3],
                              # AA_DOS_med=agg_dos_aa[m_pos, 2][,3],
                              NWS_SL_avg=agg_sl_nws[m_pos, 2][,4],
                              AA_SL_avg=agg_sl_aa[m_pos, 2][,4]
)
names(df_alert_related)<- c("stdev(Alert Effectiveness)", "No. Alerts",
                            "Med. DOS of RL Alerts", 
                            "Avg. SL of NWS Alerts", "Avg. SL of AA.QHI Alerts"
                           )

df_all<- data.frame(stationary_W[,c("Region", "Med.HH.Income",
                                     "SD_Eff", 
                                     "acf_auc_1d", "acf_auc_3d", "acf_auc_5d", 
                                     "acf_auc_7d")],
                     Alerts=stationary_W$Alerts/3,
                     West=stationary_W$State %in% West,
                     South=stationary_W$State %in% South,
                     RL_DOS_med=agg_dos_rl[m_pos, 2][,3],
                     # NWS_DOS_med=agg_dos_nws[m_pos, 2][,3],
                     # AA_DOS_med=agg_dos_aa[m_pos, 2][,3],
                     NWS_SL_avg=agg_sl_nws[m_pos, 2][,4],
                     AA_SL_avg=agg_sl_aa[m_pos, 2][,4]
                     )
names(df_all)<- c("Region", "Med. HH Income", 
                  "stdev(Alert Effectiveness)",
                  "Cum. ACF. 1d", "Cum. ACF. 3d",
                  "Cum. ACF. 5d", "Cum. ACF. 7d",
                  "No. Alerts", 
                  "Western", "Southern",
                  "Med. DOS of RL Alerts", 
                  "Avg. SL of NWS Alerts", "Avg. SL of AA.QHI Alerts"
                  )

CART_df<- df_most_interpretable
# CART_df<- df_alert_related
# CART_df<- df_all

paste(shQuote(names(CART_df)), collapse=", ")

par(mfrow=c(1,2), mai = c(1, 0.5, 0.1, 0.75))

## Compared to NWS:
class_fit<- rpart(Y ~ ., data = CART_df, method = "class", model = TRUE
      , control = rpart.control(minbucket=3)
      )
rpart.plot(class_fit, box.palette = 0)

reg_fit<- rpart(Diff ~ ., data = CART_df, method = "anova", model = TRUE
                  , control = rpart.control(minbucket=5) 
)
rpart.plot(reg_fit, box.palette = 0)

## Comparing A2C.QHI and TRPO.QHI:
class_fit<- rpart(Y_rl ~ ., data = CART_df, method = "class", model = TRUE
                  , control = rpart.control(max.depth=3)
)
rpart.plot(class_fit, box.palette = 0)

## Compared to AA.QHI:
class_fit<- rpart(Y.a ~ ., data = CART_df, method = "class", model = TRUE
                  , control = rpart.control(max.depth=2)
)
rpart.plot(class_fit, box.palette = 0)

reg_fit<- rpart(Diff.a ~ ., data = CART_df, method = "anova", model = TRUE
                , control = rpart.control(max.depth=2)
)
rpart.plot(reg_fit, box.palette = 0)


######## Investigating intuitions:

plot(stationary_W$Med.HH.Income, stationary_W$Alerts)
abline(lm(Alerts ~ Med.HH.Income, stationary_W))

plot(stationary_W$Pop_density, stationary_W$Alerts)
abline(lm(Alerts ~ Pop_density, stationary_W))

plot(stationary_W$Alerts, stationary_W$SD_Eff)
abline(lm(SD_Eff ~ Alerts, stationary_W))

plot(stationary_W$Med.HH.Income, stationary_W$SD_Eff)
abline(lm(SD_Eff ~ Med.HH.Income, stationary_W))

plot(stationary_W$Med.HH.Income, stationary_W$Mean_Eff)
abline(lm(Mean_Eff ~ Med.HH.Income, stationary_W))

##### Making case study plot:

DF<- data.frame(Diff, RL_F.none)

m<- which.max(Diff)
DF[m,] # county 22063, OT = 0.5, NHL = 3, NHU = 64, n_steps = 4096
county<- 22063


test<- read.csv(paste0("Summer_results/ORL_RL_eval_samp-R_obs-W_", 
                       "mixed_constraints_trpo_F-none_fips-41067", "_fips_", county, ".csv"))[,-1]

df<- read.csv(paste0("Summer_results/ORL_RL_eval_samp-R_obs-W_Tune_F-none", 
             "_Rstr-HI-0.5",
             "_arch-", 2, "-", 64,
             "_ns-", 4096, "_fips-", county, "_fips_", county, ".csv"))[,-1]
df$Count<- 1
agg_df<- aggregate(. ~ Year, df, sum)

# nws<- read.csv(paste0("Summer_results/ORL_NWS_eval_samp-R_obs-W_", "mixed_constraints", "_fips_", county, ".csv"))
nws<- read.csv(paste0("ORL_NWS_eval_samp-R_obs-W_", "mixed_constraints", "_fips_", county, ".csv"))
nws$Count<- 1
agg_nws<- aggregate(. ~ Year, nws, sum)

df.y<- df[which(df$Year == 2011),]
DOS<- as.numeric(row.names(df.y)) %% 152
table(DOS[which(df.y$Actions == 1)])

nws.y<- nws[which(nws$Year == 2011),]
nws.DOS<- as.numeric(row.names(nws.y)) %% 152
table(nws.DOS[which(nws.y$Actions == 1)])
