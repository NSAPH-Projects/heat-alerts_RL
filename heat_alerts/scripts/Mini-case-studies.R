library(arrow)
library(lubridate)
library(ggplot2)
library(cowplot)
library(extrafont)
loadfonts()

rl_results1<- read.csv("Fall_results/December_Rstr-QHI_RL_avg_return.csv")
rl_results2<- read.csv("Fall_results/December_part-2_Rstr-QHI_RL_avg_return.csv")

trpo_results<- rl_results1[which(rl_results1$Algo=="trpo" & rl_results1$Forecast=="none"),]
dqn_results<- rl_results1[which(rl_results1$Algo=="dqn" & rl_results1$Forecast=="none"),]
a2c_results<- rl_results2[which(rl_results2$Algo=="a2c" & rl_results2$Forecast=="none"),]
qrdqn_results<- rl_results2[which(rl_results2$Algo=="qrdqn" & rl_results2$Forecast=="none"),]

bench_results<- read.csv("Fall_results/Benchmarks_mixed_constraints_avg_return.csv")
aa_results<- bench_results[c("County", "aqhi_ot")]

trpo_diffs<- trpo_results$Eval - bench_results$NWS
dqn_diffs<- dqn_results$Eval - bench_results$NWS
a2c_diffs<- a2c_results$Eval - bench_results$NWS
qrdqn_diffs<- qrdqn_results$Eval - bench_results$NWS

data<- read_parquet("data/processed/states.parquet")
QHI<- data$quant_HI_county
Year<- year(data$Date)

W<- read.csv("data/Final_30_W.csv")

gg_color_hue <- function(n) {
  hues = seq(15, 375, length = n + 1)
  hcl(h = hues, l = 65, c = 100)[1:n]
}

cols<- gg_color_hue(6)

y<- 2015 # the year to plot
j<-1
for(i in which(trpo_results$County %in% c("17115", "19153", "5045", "28049", "29021")
  )){
  county<- trpo_results$County[i]
  c_name<- rep("County", 30)[j]
  trpo<- read.csv(paste0("Summer_results/ORL_RL_eval_samp-R_obs-W_December_",
                        "trpo", "_F-", "none", "_Rstr-HI-", trpo_results$OT[i],
                        "_arch-", trpo_results$NHL[i], "-", trpo_results$NHU[i], "_ns-", trpo_results$n_steps[i],
                        "_fips-", county, "_fips_", county, ".csv"))
  dqn<- read.csv(paste0("Summer_results/ORL_RL_eval_samp-R_obs-W_December_",
                         "dqn", "_F-", "none", "_Rstr-HI-", dqn_results$OT[i],
                         "_arch-", dqn_results$NHL[i], "-", dqn_results$NHU[i], "_ns-", dqn_results$n_steps[i],
                         "_fips-", county, "_fips_", county, ".csv"))
  a2c<- read.csv(paste0("Summer_results/ORL_RL_eval_samp-R_obs-W_December_",
                       "a2c", "_F-", "none", "_Rstr-HI-", a2c_results$OT[i],
                       "_arch-", a2c_results$NHL[i], "-", a2c_results$NHU[i], "_ns-", a2c_results$n_steps[i],
                       "_fips-", county, "_fips_", county, ".csv"))
  qrdqn<- read.csv(paste0("Summer_results/ORL_RL_eval_samp-R_obs-W_December_",
                        "qrdqn", "_F-", "none", "_Rstr-HI-", qrdqn_results$OT[i],
                        "_arch-", qrdqn_results$NHL[i], "-", qrdqn_results$NHU[i], "_ns-", qrdqn_results$n_steps[i],
                        "_fips-", county, "_fips_", county, ".csv"))
  
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
  
  a2c_eval<- aggregate(Rewards ~ Year, a2c, mean)[3, "Rewards"]*152
  dqn_eval<- aggregate(Rewards ~ Year, dqn, mean)[3, "Rewards"]*152
  qrdqn_eval<- aggregate(Rewards ~ Year, qrdqn, mean)[3, "Rewards"]*152
  trpo_eval<- aggregate(Rewards ~ Year, trpo, mean)[3, "Rewards"]*152
  
  nws_eval<- aggregate(Rewards ~ Year, nws, mean)[3, "Rewards"]*152
  aa_eval<- aggregate(Rewards ~ Year, aa.qhi, mean)[3, "Rewards"]*152
  
  w<- W[which(W$Fips == county),]
  
  a2c_hosps<- ((1 - a2c_eval/152)*w$Offset/w$total_count)*152*10000
  dqn_hosps<- ((1 - dqn_eval/152)*w$Offset/w$total_count)*152*10000
  qrdqn_hosps<- ((1 - qrdqn_eval/152)*w$Offset/w$total_count)*152*10000
  trpo_hosps<- ((1 - trpo_eval/152)*w$Offset/w$total_count)*152*10000
  
  nws_hosps<- ((1 - nws_eval/152)*w$Offset/w$total_count)*152*10000
  aa_hosps<- ((1 - aa_eval/152)*w$Offset/w$total_count)*152*10000
  
  qhi<- QHI[which(data$fips == county & Year == y)][1:152]
  
  pol_factor<- as.factor(c(rep("NWS", 152), rep("AA.QHI", 152), 
                 rep("TRPO.QHI (RL)", 152*5), rep("DQN.QHI (RL)", 152), 
                 rep("A2C.QHI (RL)", 152*5), rep("QRDQN.QHI (RL)", 152)
                 ))
  pol_factor<- relevel(pol_factor, "NWS")
  pol_factor<- relevel(pol_factor, "AA.QHI")
  
  df<- data.frame(DOS=rep(1:152,14), QHI=rep(qhi,14),
                  Alerts = c(nws_y*1.05, aa.qhi_y*1.10,
                             trpo_1*1.5, trpo_2*1.55, trpo_3*1.6, trpo_4*1.65, trpo_5*1.7,
                             dqn_0*1.15, a2c_1*1.25, a2c_2*1.3, a2c_3*1.35, a2c_4*1.4, a2c_5*1.45,
                             qrdqn_0*1.2 # , a2c_0*1.25, trpo_0*1.3
                             ),
                  Policy=pol_factor)
  
  p<- ggplot(df, aes(x=DOS, y=QHI)) + 
    geom_hline(yintercept=aa_ot, color=cols[1], lty="solid", lwd=2) +
    geom_hline(yintercept=qrdqn_results$OT[i], color=cols[5], lty="longdash", lwd=2) +
    geom_hline(yintercept=dqn_results$OT[i], color=cols[4], lty="dashed", lwd=2) +
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
    geom_text(aes(label=sum(qrdqn_0), x=5, y=1.25), color=cols[5], fontface="plain") +
    geom_text(aes(label=sum(dqn_0), x=5, y=1.15), color=cols[4], fontface="plain") +
    geom_text(aes(label=sum(aa.qhi_y), x=5, y=1.05), color=cols[1], fontface="plain") +
    geom_text(aes(label="Saved per 10k", x=162, y=1.7), color="black", fontface="plain") +
    geom_text(aes(label=round(nws_hosps - trpo_hosps, 3), x=162, y=1.55), color=cols[6], fontface="plain") + 
    geom_text(aes(label=round(nws_hosps - a2c_hosps, 3), x=162, y=1.4), color=cols[3], fontface="plain") +
    geom_text(aes(label=round(nws_hosps - qrdqn_hosps, 3), x=162, y=1.25), color=cols[5], fontface="plain") +
    geom_text(aes(label=round(nws_hosps - dqn_hosps, 3), x=162, y=1.15), color=cols[4], fontface="plain") +
    geom_text(aes(label=round(nws_hosps - aa_hosps, 3), x=162, y=1.05), color=cols[1], fontface="plain") +
    ggtitle(paste0(c_name, " ", county, " in ", y, ": ", sum(nws_y), " Alerts, ", round(nws_hosps), " NOHR Hosps per 10,000 Under the NWS Policy"))
  
  print(p)
  j<- j+1
}


