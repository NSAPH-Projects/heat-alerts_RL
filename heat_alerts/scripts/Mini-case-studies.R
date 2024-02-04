library(arrow)
library(lubridate)
library(ggplot2)
library(cowplot)
library(extrafont)
loadfonts()

rl_results1<- read.csv("Fall_results/December_Rstr-QHI_RL_avg_return.csv")
rl_results2<- read.csv("Fall_results/December_part-2_Rstr-QHI_RL_avg_return.csv")
# rl_results3<- read.csv("Fall_results/February_Rstr-QHI_RL_avg_return.csv")
# rl_results4<- read.csv("Fall_results/NEW_Main_analysis_trpo_F-none.csv")

trpo_results<- rl_results1[which(rl_results1$Algo=="trpo" & rl_results1$Forecast=="none"),]
dqn_results<- rl_results1[which(rl_results1$Algo=="dqn" & rl_results1$Forecast=="none"),]
a2c_results<- rl_results2[which(rl_results2$Algo=="a2c" & rl_results2$Forecast=="none"),]
qrdqn_results<- rl_results2[which(rl_results2$Algo=="qrdqn" & rl_results2$Forecast=="none"),]

# a2c_det_results<- rl_results3[which(rl_results3$Algo=="a2c" & rl_results3$Forecast=="none"),]
# trpo_det_results<- rl_results4

bench_results<- read.csv("Fall_results/Benchmarks_mixed_constraints_avg_return.csv")
aa_results<- bench_results[c("County", "aqhi_ot")]

trpo_diffs<- trpo_results$Eval - bench_results$NWS
dqn_diffs<- dqn_results$Eval - bench_results$NWS
a2c_diffs<- a2c_results$Eval - bench_results$NWS
qrdqn_diffs<- qrdqn_results$Eval - bench_results$NWS

# trpo_det_diffs<- trpo_det_results$Eval - bench_results$NWS
# a2c_det_diffs<- a2c_det_results$Eval - bench_results$NWS

data<- read_parquet("data/processed/states.parquet")
QHI<- data$quant_HI_county
Year<- year(data$Date)

W<- read.csv("data/Final_30_W.csv")

gg_color_hue <- function(n) {
  hues = seq(15, 375, length = n + 1)
  hcl(h = hues, l = 65, c = 100)[1:n]
}
# cols<- gg_color_hue(3)
cols<- gg_color_hue(6)

y<- 2015 # the year to plot
j<-1
# for(i in 1:30){
for(i in which(trpo_results$County %in%
               # c("28049", "19155", # "17115",
               #      "29019", "5045", "41053")
              # c("53015", # "17115",
              #   "5045", "48367", "32003")
              # c("21059", "5045", "17115", 
              #   "28049", "37085", "53015", "41067")
              # c("40017", "29019", "29021", "19115", 
              #   "34021", "17167", "19153", "48157", "20161")
              c("17115", "19153", "5045", "28049", "29021")
  )){ # RL does better + interesting patterns: "32003", "21059", "5045", "17115", "31153", "28049", "37085", "53015", "41067"
      # Interesting patterns, but RL doesn't do better: "40017", "29019", "29021", "19115", "34021", "17167", "19153", "48157", "20161"

  county<- trpo_results$County[i]
  # c_name<- c("Hinds County, MS", "Pottawattamie County, IA",
  #            # "Macon County, IL", 
  #            "Boone County, MO", 
  #            "Faulkner County, AR", "Polk County, OR")[j]
  # c_name<- c("Cowlitz County, WA", # "Macon County, IL",
  #            "Faulkner County, AR", 
  #            "Parker County, TX", 
  #            "Clark County, NV")[j]
  c_name<- rep("County", 30)[j]
  trpo<- read.csv(paste0("Summer_results/ORL_RL_eval_samp-R_obs-W_December_",
                        "trpo", "_F-", "none", "_Rstr-HI-", trpo_results$OT[i],
                        "_arch-", trpo_results$NHL[i], "-", trpo_results$NHU[i], "_ns-", trpo_results$n_steps[i],
                        "_fips-", county, "_fips_", county, ".csv"))
  # trpo_det<- read.csv(paste0("Summer_results/ORL_RL_eval_samp-R_obs-W_Tune_F-",
  #                        "none", "_Rstr-HI-", trpo_det_results$OT[i],
  #                        "_arch-", trpo_det_results$NHL[i], "-", trpo_det_results$NHU[i], "_ns-", trpo_det_results$n_steps[i],
  #                        "_fips-", county, "_fips_", county, ".csv"))
  dqn<- read.csv(paste0("Summer_results/ORL_RL_eval_samp-R_obs-W_December_",
                         "dqn", "_F-", "none", "_Rstr-HI-", dqn_results$OT[i],
                         "_arch-", dqn_results$NHL[i], "-", dqn_results$NHU[i], "_ns-", dqn_results$n_steps[i],
                         "_fips-", county, "_fips_", county, ".csv"))
  a2c<- read.csv(paste0("Summer_results/ORL_RL_eval_samp-R_obs-W_December_",
                       "a2c", "_F-", "none", "_Rstr-HI-", a2c_results$OT[i],
                       "_arch-", a2c_results$NHL[i], "-", a2c_results$NHU[i], "_ns-", a2c_results$n_steps[i],
                       "_fips-", county, "_fips_", county, ".csv"))
  # a2c_det<- read.csv(paste0("Summer_results/ORL_RL_eval_samp-R_obs-W_February_",
  #                       "a2c", "_F-", "none", "_Rstr-HI-", a2c_results$OT[i],
  #                       "_arch-", a2c_results$NHL[i], "-", a2c_results$NHU[i], "_ns-", a2c_results$n_steps[i],
  #                       "_fips-", county, "_fips_", county, ".csv"))
  qrdqn<- read.csv(paste0("Summer_results/ORL_RL_eval_samp-R_obs-W_December_",
                        "qrdqn", "_F-", "none", "_Rstr-HI-", qrdqn_results$OT[i],
                        "_arch-", qrdqn_results$NHL[i], "-", qrdqn_results$NHU[i], "_ns-", qrdqn_results$n_steps[i],
                        "_fips-", county, "_fips_", county, ".csv"))
  
  # a2c_0<- a2c_det[which(a2c_det$Year == y)[1:152], "Actions"]
  # trpo_0<- trpo_det[which(trpo_det$Year == y)[1:152], "Actions"]
  
  dqn_0<- dqn[which(dqn$Year == y)[1:152], "Actions"]
  qrdqn_0<- qrdqn[which(qrdqn$Year == y)[1:152], "Actions"]
  
  a2c_1<- a2c[which(a2c$Year == y)[1:152], "Actions"]
  a2c_2<- a2c[which(a2c$Year == y)[(152*50+1):(152*50+152)], "Actions"]
  a2c_3<- a2c[which(a2c$Year == y)[(152*100+1):(152*100+152)], "Actions"]
  a2c_4<- a2c[which(a2c$Year == y)[(152*150+1):(152*150+152)], "Actions"]
  a2c_5<- a2c[which(a2c$Year == y)[(152*200+1):(152*200+152)], "Actions"]
  
  trpo_1<- trpo[which(trpo$Year == y)[1:152], "Actions"]
  trpo_2<- trpo[which(trpo$Year == y)[(152*50+1):(152*50+152)], "Actions"]
  trpo_3<- trpo[which(trpo$Year == y)[(152*100+1):(152*100+152)], "Actions"]
  trpo_4<- trpo[which(trpo$Year == y)[(152*150+1):(152*150+152)], "Actions"]
  trpo_5<- trpo[which(trpo$Year == y)[(152*200+1):(152*200+152)], "Actions"]
    
  nws<- read.csv(paste0("Summer_results/ORL_NWS_eval_samp-R_obs-W_", "mixed_constraints", "_fips_", county, ".csv"))
  nws_y<- nws[which(nws$Year == y)[1:152], "Actions"]
  aa_ot<- aa_results[i, "aqhi_ot"]
  aa.qhi<- read.csv(paste0("Summer_results/ORL_AA_eval_samp-R_obs-W_", "mixed_constraints", "_Rstr-HI-", aa_ot, "_fips_", county, ".csv"))
  aa.qhi_y<- aa.qhi[which(aa.qhi$Year == y)[1:152], "Actions"]
  
  # a2c_det_eval<- aggregate(Rewards ~ Year, a2c_det, mean)[3, "Rewards"]*152
  # trpo_det_eval<- aggregate(Rewards ~ Year, trpo_det, mean)[3, "Rewards"]*152
  a2c_eval<- aggregate(Rewards ~ Year, a2c, mean)[3, "Rewards"]*152
  dqn_eval<- aggregate(Rewards ~ Year, dqn, mean)[3, "Rewards"]*152
  qrdqn_eval<- aggregate(Rewards ~ Year, qrdqn, mean)[3, "Rewards"]*152
  trpo_eval<- aggregate(Rewards ~ Year, trpo, mean)[3, "Rewards"]*152
  
  nws_eval<- aggregate(Rewards ~ Year, nws, mean)[3, "Rewards"]*152
  aa_eval<- aggregate(Rewards ~ Year, aa.qhi, mean)[3, "Rewards"]*152
  
  w<- W[which(W$Fips == county),]
  
  # a2c_det_hosps<- ((1 - a2c_det_eval/152)*w$Offset/w$total_count)*152*w$total_count # or could do per 10,000
  # trpo_det_hosps<- ((1 - trpo_det_eval/152)*w$Offset/w$total_count)*152*w$total_count # or could do per 10,000
  a2c_hosps<- ((1 - a2c_eval/152)*w$Offset/w$total_count)*152*w$total_count # or could do per 10,000
  dqn_hosps<- ((1 - dqn_eval/152)*w$Offset/w$total_count)*152*w$total_count # or could do per 10,000
  qrdqn_hosps<- ((1 - qrdqn_eval/152)*w$Offset/w$total_count)*152*w$total_count # or could do per 10,000
  trpo_hosps<- ((1 - trpo_eval/152)*w$Offset/w$total_count)*152*w$total_count # or could do per 10,000
  
  nws_hosps<- ((1 - nws_eval/152)*w$Offset/w$total_count)*152*w$total_count # or could do per 10,000
  aa_hosps<- ((1 - aa_eval/152)*w$Offset/w$total_count)*152*w$total_count # or could do per 10,000
  
  qhi<- QHI[which(data$fips == county & Year == y)][1:152]
  # autocorr<- acf(qhi, plot=FALSE)
  # AUC_5days<- sum(autocorr$acf[2:4])
  
  # aa.qhi<- as.numeric(qhi >= aa_ot)
  # aa.qhi[which(aa.qhi == 1)[(sum(nws_y)+1):sum(aa.qhi)]]<- 0
  
  pol_factor<- as.factor(c(rep("NWS", 152), rep("AA.QHI", 152), 
                 rep("TRPO.QHI (RL)", 152*5), rep("DQN.QHI (RL)", 152), 
                 rep("A2C.QHI (RL)", 152*5), rep("QRDQN.QHI (RL)", 152) #,
                 # rep("A2C.QHI determ.", 152), rep("TRPO.QHI determ.", 152)
                 ))
  pol_factor<- relevel(pol_factor, "NWS")
  pol_factor<- relevel(pol_factor, "AA.QHI")
  
  df<- data.frame(DOS=rep(1:152,14), QHI=rep(qhi,14),
                  Alerts = c(nws_y*1.05, aa.qhi_y*1.10,
                             # trpo_1*1.3, trpo_2*1.35, trpo_3*1.4, trpo_4*1.45, trpo_5*1.5,
                             # dqn_0*1.15, a2c_0*1.25, qrdqn_0*1.2
                             trpo_1*1.5, trpo_2*1.55, trpo_3*1.6, trpo_4*1.65, trpo_5*1.7,
                             dqn_0*1.15, a2c_1*1.25, a2c_2*1.3, a2c_3*1.35, a2c_4*1.4, a2c_5*1.45,
                             qrdqn_0*1.2 # , a2c_0*1.25, trpo_0*1.3
                             ),
                  Policy=pol_factor)
  
  p<- ggplot(df, aes(x=DOS, y=QHI)) + 
    # geom_text(aes(label=paste0("Area under the curve of the ACF function, lagged three days = ",
    #                            round(AUC_5days,3)), x=87, y=0, fontface="plain"), color="black") +
    geom_hline(yintercept=aa_ot, color=cols[1], lty="solid", lwd=2) +
    geom_hline(yintercept=qrdqn_results$OT[i], color=cols[5], lty="longdash", lwd=2) +
    geom_hline(yintercept=dqn_results$OT[i], color=cols[4], lty="dashed", lwd=2) +
    # geom_hline(yintercept=a2c_det_results$OT[i], color=cols[4], lty="solid", lwd=2) +
    geom_hline(yintercept=a2c_results$OT[i], color=cols[3], lty="longdash", lwd=2) +
    geom_hline(yintercept=trpo_results$OT[i], color=cols[6], lty="dashed", lwd=2) +
    geom_line() + 
    geom_point(aes(y=Alerts, color=Policy), alpha=df$Alerts > 0) +
    ylim(c(0, 1.7)) + # ylim(c(0, 1.5)) +
    xlim(c(0, 175)) +
    ylab("Quantile of Heat Index; Alert Status") + xlab("Day of Summer") +
    geom_text(aes(label="No. Alerts", x=5, y=1.7), color="black", fontface="plain") +
    geom_text(aes(label=mean(sum(trpo_1), sum(trpo_2), sum(trpo_3), sum(trpo_4), sum(trpo_5)), x=5, y=1.55), color=cols[6], fontface="plain") +
    geom_text(aes(label=mean(sum(a2c_1), sum(a2c_2), sum(a2c_3), sum(a2c_4), sum(a2c_5)), x=5, y=1.4), color=cols[3], fontface="plain") +
    # geom_text(aes(label=sum(a2c_0), x=5, y=1.3), color=cols[4], fontface="plain") +
    geom_text(aes(label=sum(qrdqn_0), x=5, y=1.25), color=cols[5], fontface="plain") +
    geom_text(aes(label=sum(dqn_0), x=5, y=1.15), color=cols[4], fontface="plain") +
    geom_text(aes(label=sum(aa.qhi_y), x=5, y=1.05), color=cols[1], fontface="plain") +
    geom_text(aes(label="Saved, per 10k", x=162, y=1.7), color="black", fontface="plain") +
    geom_text(aes(label=capture.output(cat(round(c(1, 10000/w$total_count)*(nws_hosps - trpo_hosps),3),sep=", ")), x=162, y=1.55), color=cols[6], fontface="plain") +
    geom_text(aes(label=capture.output(cat(round(c(1, 10000/w$total_count)*(nws_hosps - a2c_hosps),3),sep=", ")), x=162, y=1.4), color=cols[3], fontface="plain") +
    # geom_text(aes(label=capture.output(cat(round(c(1, 10000/w$total_count)*(nws_hosps - a2c_det_hosps),3),sep=", ")), x=162, y=1.3), color=cols[4], fontface="plain") +
    geom_text(aes(label=capture.output(cat(round(c(1, 10000/w$total_count)*(nws_hosps - qrdqn_hosps),3),sep=", ")), x=162, y=1.25), color=cols[5], fontface="plain") +
    geom_text(aes(label=capture.output(cat(round(c(1, 10000/w$total_count)*(nws_hosps - dqn_hosps),3),sep=", ")), x=162, y=1.15), color=cols[4], fontface="plain") +
    geom_text(aes(label=capture.output(cat(round(c(1, 10000/w$total_count)*(nws_hosps - aa_hosps),3),sep=", ")), x=162, y=1.05), color=cols[1], fontface="plain") +
    # ggtitle(paste("County", county, "Diff =", round(diffs[i], 3)))
    # ggtitle(paste0(c_name, " (", county, "); Avg. Return Diff. (vs. NWS) = ", round(rl_eval - nws_eval,3)))
    ggtitle(paste0(c_name, " ", county, " in ", y, ": ", sum(nws_y), " Alerts, ", round(nws_hosps), " NOHR Hosps Under the NWS Policy"))
  
  print(p)
  j<- j+1
}


############## OLD

# ## Make final plot:
# 
# counties<- c("48367", "5045")
# c_names<- c("Parker County, TX", "Faulkner County, AR")
# 
# county<- counties[1]
# i<- which(RL_results$County == county)
# # rl<- read.csv(paste0("Summer_results/ORL_RL_eval_samp-R_obs-W_December_",
# #                       "trpo", "_F-", "none", "_Rstr-HI-", RL_results$OT[i],
# #                       "_arch-", RL_results$NHL[i], "-", RL_results$NHU[i], "_ns-", RL_results$n_steps[i],
# #                       "_fips-", county, "_fips_", county, ".csv"))
# rl<- read.csv(paste0("Summer_results/ORL_RL_eval_samp-R_obs-W_February_",
#                      "a2c", "_F-", "none", "_Rstr-HI-", RL_results$OT[i],
#                      "_arch-", RL_results$NHL[i], "-", RL_results$NHU[i], "_ns-", RL_results$n_steps[i],
#                      "_fips-", county, "_fips_", county, ".csv"))
# rl_0<- rl[which(rl$Year == y)[1:152], "Actions"]
# 
# # rl_1<- rl[which(rl$Year == y)[1:152], "Actions"]
# # rl_2<- rl[which(rl$Year == y)[(152*50+1):(152*50+152)], "Actions"]
# # rl_3<- rl[which(rl$Year == y)[(152*100+1):(152*100+152)], "Actions"]
# # rl_4<- rl[which(rl$Year == y)[(152*150+1):(152*150+152)], "Actions"]
# # rl_5<- rl[which(rl$Year == y)[(152*200+1):(152*200+152)], "Actions"]
# 
# nws<- read.csv(paste0("Summer_results/ORL_NWS_eval_samp-R_obs-W_", "mixed_constraints", "_fips_", county, ".csv"))
# nws_y<- nws[which(nws$Year == y)[1:152], "Actions"]
# 
# rl_eval<- aggregate(Rewards ~ Year, rl, mean)[3, "Rewards"]*152
# nws_eval<- aggregate(Rewards ~ Year, nws, mean)[3, "Rewards"]*152
# 
# w<- W[which(W$Fips == county),]
# rl_hosps<- ((1 - rl_eval/152)*w$Offset/w$total_count)*152*w$total_count # or could do per 10,000
# nws_hosps<- ((1 - nws_eval/152)*w$Offset/w$total_count)*152*w$total_count # or could do per 10,000
# 
# aa_ot<- aa_results[i, "aqhi_ot"]
# 
# qhi<- QHI[which(data$fips == county & Year == y)][1:152]
# 
# aa.qhi<- as.numeric(qhi >= aa_ot)
# aa.qhi[which(aa.qhi == 1)[(sum(nws_y)+1):sum(aa.qhi)]]<- 0
# 
# pol_factor<- as.factor(c(rep("NWS", 152), rep("AA.QHI", 152), 
#                          # rep("TRPO.QHI samples", 152*5)
#                          rep("A2C.QHI (RL) samples", 152)))
# pol_factor<- relevel(pol_factor, "NWS")
# pol_factor<- relevel(pol_factor, "AA.QHI")
# 
# df<- data.frame(# DOS=rep(1:152,7), QHI=rep(qhi,7),
#   DOS=rep(1:152,3), QHI=rep(qhi,3),
#   Alerts = c(nws_y*1.05, aa.qhi*1.10, 
#              # rl_1*1.15, rl_2*1.2, rl_3*1.25, rl_4*1.3, rl_5*1.35
#              rl_0*1.15
#   ),
#   Policy=pol_factor)
# 
# p1<- ggplot(df, aes(x=DOS)) + geom_line(aes(y=QHI)) + 
#   geom_hline(yintercept=aa_ot, color=cols[1], lty=2) + 
#   geom_hline(yintercept=RL_results$OT[i], color=cols[3], lty=2) +
#   geom_point(aes(y=Alerts, color=Policy), alpha=df$Alerts > 0) +
#   # ylim(c(0, 1.4)) + 
#   ylim(c(0, 1.2)) +
#   ylab("Quantile of Heat Index; Alert Status") + xlab("Day of Summer") +
#   # ggtitle(paste("County", county, "Diff =", round(diffs[i], 3)))
#   # ggtitle(paste0(c_name, " (", county, "); Avg. Return Diff. (vs. NWS) = ", round(rl_eval - nws_eval,3)))
#   ggtitle(paste0(c_names[1], " (", county, "); NOHR Hosps Saved (vs. NWS) in ", y, " = ", round(nws_hosps-rl_hosps,2)))
# 
# 
# county<- counties[2]
# i<- which(RL_results$County == county)
# # rl<- read.csv(paste0("Summer_results/ORL_RL_eval_samp-R_obs-W_December_",
# #                       "trpo", "_F-", "none", "_Rstr-HI-", RL_results$OT[i],
# #                       "_arch-", RL_results$NHL[i], "-", RL_results$NHU[i], "_ns-", RL_results$n_steps[i],
# #                       "_fips-", county, "_fips_", county, ".csv"))
# rl<- read.csv(paste0("Summer_results/ORL_RL_eval_samp-R_obs-W_February_",
#                      "a2c", "_F-", "none", "_Rstr-HI-", RL_results$OT[i],
#                      "_arch-", RL_results$NHL[i], "-", RL_results$NHU[i], "_ns-", RL_results$n_steps[i],
#                      "_fips-", county, "_fips_", county, ".csv"))
# rl_0<- rl[which(rl$Year == y)[1:152], "Actions"]
# 
# # rl_1<- rl[which(rl$Year == y)[1:152], "Actions"]
# # rl_2<- rl[which(rl$Year == y)[(152*50+1):(152*50+152)], "Actions"]
# # rl_3<- rl[which(rl$Year == y)[(152*100+1):(152*100+152)], "Actions"]
# # rl_4<- rl[which(rl$Year == y)[(152*150+1):(152*150+152)], "Actions"]
# # rl_5<- rl[which(rl$Year == y)[(152*200+1):(152*200+152)], "Actions"]
# 
# nws<- read.csv(paste0("Summer_results/ORL_NWS_eval_samp-R_obs-W_", "mixed_constraints", "_fips_", county, ".csv"))
# nws_y<- nws[which(nws$Year == y)[1:152], "Actions"]
# 
# rl_eval<- aggregate(Rewards ~ Year, rl, mean)[3, "Rewards"]*152
# nws_eval<- aggregate(Rewards ~ Year, nws, mean)[3, "Rewards"]*152
# 
# w<- W[which(W$Fips == county),]
# rl_hosps<- ((1 - rl_eval/152)*w$Offset/w$total_count)*152*w$total_count # or could do per 10,000
# nws_hosps<- ((1 - nws_eval/152)*w$Offset/w$total_count)*152*w$total_count # or could do per 10,000
# 
# aa_ot<- aa_results[i, "aqhi_ot"]
# 
# qhi<- QHI[which(data$fips == county & Year == y)][1:152]
# 
# aa.qhi<- as.numeric(qhi >= aa_ot)
# aa.qhi[which(aa.qhi == 1)[(sum(nws_y)+1):sum(aa.qhi)]]<- 0
# 
# pol_factor<- as.factor(c(rep("NWS", 152), rep("AA.QHI", 152), 
#                          # rep("TRPO.QHI samples", 152*5)
#                          rep("A2C.QHI (RL) samples", 152)))
# pol_factor<- relevel(pol_factor, "NWS")
# pol_factor<- relevel(pol_factor, "AA.QHI")
# 
# df<- data.frame(# DOS=rep(1:152,7), QHI=rep(qhi,7),
#   DOS=rep(1:152,3), QHI=rep(qhi,3),
#   Alerts = c(nws_y*1.05, aa.qhi*1.10, 
#              # rl_1*1.15, rl_2*1.2, rl_3*1.25, rl_4*1.3, rl_5*1.35
#              rl_0*1.15
#   ),
#   Policy=pol_factor)
# 
# p2<- ggplot(df, aes(x=DOS)) + geom_line(aes(y=QHI)) + 
#   geom_hline(yintercept=aa_ot, color=cols[1], lty=2) + 
#   geom_hline(yintercept=RL_results$OT[i], color=cols[3], lty=2) +
#   geom_point(aes(y=Alerts, color=Policy), alpha=df$Alerts > 0) +
#   # ylim(c(0, 1.4)) + 
#   ylim(c(0, 1.2)) +
#   ylab("Quantile of Heat Index; Alert Status") + xlab("Day of Summer") +
#   # ggtitle(paste("County", county, "Diff =", round(diffs[i], 3)))
#   # ggtitle(paste0(c_name, " (", county, "); Avg. Return Diff. (vs. NWS) = ", round(rl_eval - nws_eval,3)))
#   ggtitle(paste0(c_names[2], " (", county, "); NOHR Hosps Saved (vs. NWS) in ", y, " = ", round(nws_hosps-rl_hosps,2)))
# 
# 
# this_legend<- get_legend(p1 + theme(legend.position = "bottom"))
# 
# # png("Mini-case-studies.png", width=500, height=500)
# # plot_grid(p1 + theme(legend.position = "none"),
# #           p2 + theme(legend.position = "none"),
# #           this_legend, nrow=3, rel_heights=c(5, 5, 0.5))
# # dev.off()
# 
# # png("Mini-case-studies_2-legends.png", width=600, height=500)
# png("Mini-case-studies_2-legends_a2c.png", width=600, height=500)
# plot_grid(p1, p2, nrow=2)
# dev.off()


