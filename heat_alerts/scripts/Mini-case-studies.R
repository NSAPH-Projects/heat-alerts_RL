library(arrow)
library(lubridate)
library(ggplot2)
library(cowplot)

rl_results<- read.csv("Fall_results/December_Rstr-QHI_RL_avg_return.csv")

RL_results<- rl_results[which(rl_results$Algo=="trpo" & rl_results$Forecast=="none"),]

bench_results<- read.csv("Fall_results/Benchmarks_mixed_constraints_avg_return.csv")
diffs<- RL_results$Eval - bench_results$NWS

aa_results<- bench_results[c("County", "aqhi_ot")]

data<- read_parquet("data/processed/states.parquet")
QHI<- data$quant_HI_county
Year<- year(data$Date)

W<- read.csv("data/Final_30_W.csv")

gg_color_hue <- function(n) {
  hues = seq(15, 375, length = n + 1)
  hcl(h = hues, l = 65, c = 100)[1:n]
}
cols<- gg_color_hue(3)

y<- 2015 # the year to plot
# for(i in 1:30){
j<-1
for(i in which(RL_results$County %in% c("28049", "19155", 
                                        # "17115", 
                                        "29019", "5045", "41053"))){
  county<- RL_results$County[i]
  c_name<- c("Hinds County, MS", "Pottawattamie County, IA",
             # "Macon County, IL", 
             "Boone County, MO", 
             "Faulkner County, AR", "Polk County, OR")[j]
  rl<- read.csv(paste0("Summer_results/ORL_RL_eval_samp-R_obs-W_December_",
                        "trpo", "_F-", "none", "_Rstr-HI-", RL_results$OT[i],
                        "_arch-", RL_results$NHL[i], "-", RL_results$NHU[i], "_ns-", RL_results$n_steps[i],
                        "_fips-", county, "_fips_", county, ".csv"))
  
  rl_1<- rl[which(rl$Year == y)[1:152], "Actions"]
  rl_2<- rl[which(rl$Year == y)[(152*50+1):(152*50+152)], "Actions"]
  rl_3<- rl[which(rl$Year == y)[(152*100+1):(152*100+152)], "Actions"]
  rl_4<- rl[which(rl$Year == y)[(152*150+1):(152*150+152)], "Actions"]
  rl_5<- rl[which(rl$Year == y)[(152*200+1):(152*200+152)], "Actions"]
    
  nws<- read.csv(paste0("Summer_results/ORL_NWS_eval_samp-R_obs-W_", "mixed_constraints", "_fips_", county, ".csv"))
  nws_y<- nws[which(nws$Year == y)[1:152], "Actions"]
  
  rl_eval<- aggregate(Rewards ~ Year, rl, mean)[3, "Rewards"]*152
  nws_eval<- aggregate(Rewards ~ Year, nws, mean)[3, "Rewards"]*152
  
  w<- W[which(W$Fips == county),]
  rl_hosps<- ((1 - rl_eval/152)*w$Offset/w$total_count)*152*w$total_count # or could do per 10,000
  nws_hosps<- ((1 - nws_eval/152)*w$Offset/w$total_count)*152*w$total_count # or could do per 10,000
  
  aa_ot<- aa_results[i, "aqhi_ot"]
  
  qhi<- QHI[which(data$fips == county & Year == y)][1:152]
  
  aa.qhi<- as.numeric(qhi >= aa_ot)
  aa.qhi[which(aa.qhi == 1)[(sum(nws_y)+1):sum(aa.qhi)]]<- 0
  
  df<- data.frame(DOS=rep(1:152,7), QHI=rep(qhi,7), 
                  Alerts = c(nws_y*1.05, aa.qhi*1.10, 
                             rl_1*1.15, rl_2*1.2, rl_3*1.25, rl_4*1.3, rl_5*1.35),
                  Policy=c(rep("NWS", 152), rep("AA.QHI", 152), 
                           rep("TRPO.QHI samples", 152*5)))
  
  p<- ggplot(df, aes(x=DOS)) + geom_line(aes(y=QHI)) + 
    geom_hline(yintercept=aa_ot, color=cols[1], lty=2) + 
    geom_hline(yintercept=RL_results$OT[i], color=cols[3], lty=2) +
    geom_point(aes(y=Alerts, color=Policy), alpha=df$Alerts > 0) +
    ylim(c(0, 1.4)) + ylab("Quantile of Heat Index; Alert Status") + xlab("Day of Summer") +
    # ggtitle(paste("County", county, "Diff =", round(diffs[i], 3)))
    # ggtitle(paste0(c_name, " (", county, "); Avg. Return Diff. (vs. NWS) = ", round(rl_eval - nws_eval,3)))
    ggtitle(paste0(c_name, " (", county, "); NOHR Hosps Saved (vs. NWS) in ", y, " = ", round(nws_hosps-rl_hosps,2)))
  
  print(p)
  print(sum(nws_y))
  j<- j+1
}

## Make final plot:

counties<- c("5045", "19155")
c_names<- c("Faulkner County, AR", "Pottawattamie County, IA")

county<- counties[1]
i<- which(RL_results$County == county)
rl<- read.csv(paste0("Summer_results/ORL_RL_eval_samp-R_obs-W_December_",
                     "trpo", "_F-", "none", "_Rstr-HI-", RL_results$OT[i],
                     "_arch-", RL_results$NHL[i], "-", RL_results$NHU[i], "_ns-", RL_results$n_steps[i],
                     "_fips-", county, "_fips_", county, ".csv"))

rl_1<- rl[which(rl$Year == y)[1:152], "Actions"]
rl_2<- rl[which(rl$Year == y)[(152*50+1):(152*50+152)], "Actions"]
rl_3<- rl[which(rl$Year == y)[(152*100+1):(152*100+152)], "Actions"]
rl_4<- rl[which(rl$Year == y)[(152*150+1):(152*150+152)], "Actions"]
rl_5<- rl[which(rl$Year == y)[(152*200+1):(152*200+152)], "Actions"]

nws<- read.csv(paste0("Summer_results/ORL_NWS_eval_samp-R_obs-W_", "mixed_constraints", "_fips_", county, ".csv"))
nws_y<- nws[which(nws$Year == y)[1:152], "Actions"]

rl_eval<- aggregate(Rewards ~ Year, rl, mean)[3, "Rewards"]*152
nws_eval<- aggregate(Rewards ~ Year, nws, mean)[3, "Rewards"]*152

w<- W[which(W$Fips == county),]
rl_hosps<- ((1 - rl_eval/152)*w$Offset/w$total_count)*152*w$total_count # or could do per 10,000
nws_hosps<- ((1 - nws_eval/152)*w$Offset/w$total_count)*152*w$total_count # or could do per 10,000

aa_ot<- aa_results[i, "aqhi_ot"]

qhi<- QHI[which(data$fips == county & Year == y)][1:152]

aa.qhi<- as.numeric(qhi >= aa_ot)
aa.qhi[which(aa.qhi == 1)[(sum(nws_y)+1):sum(aa.qhi)]]<- 0

df<- data.frame(DOS=rep(1:152,7), QHI=rep(qhi,7), 
                Alerts = c(nws_y*1.05, aa.qhi*1.10, 
                           rl_1*1.15, rl_2*1.2, rl_3*1.25, rl_4*1.3, rl_5*1.35),
                Policy=c(rep("NWS", 152), rep("AA.QHI", 152), 
                         rep("TRPO.QHI samples", 152*5)))

p1<- ggplot(df, aes(x=DOS)) + geom_line(aes(y=QHI)) + 
  geom_hline(yintercept=aa_ot, color=cols[1], lty=2) + 
  geom_hline(yintercept=RL_results$OT[i], color=cols[3], lty=2) +
  geom_point(aes(y=Alerts, color=Policy), alpha=df$Alerts > 0) +
  ylim(c(0, 1.4)) + xlab("Day of Summer") + ylab("Quantile of Heat Index; Alert Status") +
  # ggtitle(paste("County", county, "Diff =", round(diffs[i], 3)))
  # ggtitle(paste0(c_names[1], " (", county, "); Avg. Return Diff. (vs. NWS) = ", round(rl_eval - nws_eval,3)))
  ggtitle(paste0(c_names[1], " (", county, "); NOHR Hosps Saved (vs. NWS) in ", y, " = ", round(nws_hosps-rl_hosps,2)))


county<- counties[2]
i<- which(RL_results$County == county)
rl<- read.csv(paste0("Summer_results/ORL_RL_eval_samp-R_obs-W_December_",
                     "trpo", "_F-", "none", "_Rstr-HI-", RL_results$OT[i],
                     "_arch-", RL_results$NHL[i], "-", RL_results$NHU[i], "_ns-", RL_results$n_steps[i],
                     "_fips-", county, "_fips_", county, ".csv"))

rl_1<- rl[which(rl$Year == y)[1:152], "Actions"]
rl_2<- rl[which(rl$Year == y)[(152*50+1):(152*50+152)], "Actions"]
rl_3<- rl[which(rl$Year == y)[(152*100+1):(152*100+152)], "Actions"]
rl_4<- rl[which(rl$Year == y)[(152*150+1):(152*150+152)], "Actions"]
rl_5<- rl[which(rl$Year == y)[(152*200+1):(152*200+152)], "Actions"]

nws<- read.csv(paste0("Summer_results/ORL_NWS_eval_samp-R_obs-W_", "mixed_constraints", "_fips_", county, ".csv"))
nws_y<- nws[which(nws$Year == y)[1:152], "Actions"]

rl_eval<- aggregate(Rewards ~ Year, rl, mean)[3, "Rewards"]*152
nws_eval<- aggregate(Rewards ~ Year, nws, mean)[3, "Rewards"]*152

w<- W[which(W$Fips == county),]
rl_hosps<- ((1 - rl_eval/152)*w$Offset/w$total_count)*152*w$total_count # or could do per 10,000
nws_hosps<- ((1 - nws_eval/152)*w$Offset/w$total_count)*152*w$total_count # or could do per 10,000

aa_ot<- aa_results[i, "aqhi_ot"]

qhi<- QHI[which(data$fips == county & Year == y)][1:152]

aa.qhi<- as.numeric(qhi >= aa_ot)
aa.qhi[which(aa.qhi == 1)[(sum(nws_y)+1):sum(aa.qhi)]]<- 0

df<- data.frame(DOS=rep(1:152,7), QHI=rep(qhi,7), 
                Alerts = c(nws_y*1.05, aa.qhi*1.10, 
                           rl_1*1.15, rl_2*1.2, rl_3*1.25, rl_4*1.3, rl_5*1.35),
                Policy=c(rep("NWS", 152), rep("AA.QHI", 152), 
                         rep("TRPO.QHI samples", 152*5)))

p2<- ggplot(df, aes(x=DOS)) + geom_line(aes(y=QHI)) + 
  geom_hline(yintercept=aa_ot, color=cols[1], lty=2) + 
  geom_hline(yintercept=RL_results$OT[i], color=cols[3], lty=2) +
  geom_point(aes(y=Alerts, color=Policy), alpha=df$Alerts > 0) +
  ylim(c(0, 1.4)) + ylab("Quantile of Heat Index; Alert Status") + xlab("Day of Summer") +
  # ggtitle(paste("County", county, "Diff =", round(diffs[i], 3)))
  # ggtitle(paste0(c_names[2], " (", county, "); Avg. Return Diff. (vs. NWS) = ", round(rl_eval - nws_eval,3)))
  ggtitle(paste0(c_names[2], " (", county, "); NOHR Hosps Saved (vs. NWS) in ", y, " = ", round(nws_hosps-rl_hosps,2)))


this_legend<- get_legend(p1 + theme(legend.position = "bottom"))

png("Mini-case-studies.png", width=500, height=500)
plot_grid(p1 + theme(legend.position = "none"),
          p2 + theme(legend.position = "none"),
          this_legend, nrow=3, rel_heights=c(5, 5, 0.5))
dev.off()

png("Mini-case-studies_2-legends.png", width=600, height=500)
plot_grid(p1, p2, nrow=2)
dev.off()


