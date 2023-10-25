
library(ggplot2)
library(cowplot)

n_days<- 153

counties<- c(41067, 53015, 20161, 37085, 48157,
             28049, 19153, 17167, 31153, 6071, 4013,
             34021, 19155, 17115, 29021, 29019, 5045, 40017, 21059,
             47113, 42017, 22109, 45015, 13031, 48367, 22063, 41053,
             32003, 4015, 6025)

Bench<- read.csv("Fall_results/Benchmarks_mixed_constraints_avg_return.csv")

stationary_W<- read.csv("data/Final_30_W.csv")[,-1]
state<- stationary_W$State[match(Bench$County, stationary_W$Fips)]
region<- stationary_W$Region[match(Bench$County, stationary_W$Fips)]

########## Summary of returns across counties and years:

DF<- data.frame(matrix(ncol = 4, nrow = 0))
names(DF)<- c("Policy", "Diff", "State", "Region")

for(pol in c("Zero", "Random", "Top_K", "Random_QHI", "AA_QHI", "basic_NWS")){
  DF<- rbind(DF, data.frame(Policy=pol, Diff=Bench[,pol] - Bench$NWS, 
                            State=state, Region=region))
  print(pol)
}

# RL<- read.csv("Fall_results/RL_evals_mixed_constraints_avg_return.csv")

RL_F.q_d10<- read.csv("Fall_results/Main_analysis_trpo_F-Q-D10.csv")
DF<- rbind(DF, data.frame(Policy="TRPO.QHI.F", Diff=RL_F.q_d10$Eval - Bench$NWS, 
                          State=state, Region=region))

RL_F.none<- read.csv("Fall_results/Main_analysis_trpo_F-none.csv")
DF<- rbind(DF, data.frame(Policy="TRPO.QHI", Diff=RL_F.q_d10$Eval - Bench$NWS, 
                          State=state, Region=region))

# write.csv(DF, "Fall_results/Final_30_summary.csv")

plot_DF<- DF[which(! DF$Policy %in% c("Zero", "basic_NWS", "Random")),]
plot_DF$Policy<- factor(plot_DF$Policy,
                        levels=c("Top_K", "Random_QHI", 
                                 "AA_QHI", "TRPO.QHI", "TRPO.QHI.F"))

ggplot(plot_DF[which(plot_DF$Policy!="TRPO.QHI.F"),], aes(x=Policy, y=Diff, color = Region, label = State)) +
  geom_hline(yintercept=0) + 
  geom_boxplot() +
  geom_point(position = position_jitterdodge(), alpha=0.5) +
  # geom_text(position = position_jitterdodge(), size=2) +
  ylab("Policy Return - NWS Return") + 
  ggtitle("Comparison to NWS: Average Return on Evaluation Years")
# facet_grid(rows = vars(Year)) + 
# geom_text(size = 2,
#   position=position_jitter(width=0.5,height=0)
#   # nudge_x = 0.3, nudge_y = 0 #,
#   # check_overlap = T
# ) # +
# geom_errorbar()


########## Histograms of DOS and Streak Lengths:

streaks<- function(D){
  if(length(D)>0){
    diffs<- D[2:length(D)] - D[1:(length(D)-1)]
    L<- rle(diffs)
    streaks<- L$lengths[which(L$values == 1)] + 1
    return(streaks)
  }else{
    return(NA)
  }
}

Eval_DOS<- data.frame(matrix(ncol = 3, nrow = 0))
names(Eval_DOS)<- c("Policy", "Value")
Eval_Strk.Ln<- data.frame(matrix(ncol = 3, nrow = 0))
names(Eval_Strk.Ln)<- c("Policy", "Value")

r_model<- "mixed_constraints"

## Benchmark stats:
for(k in counties){
  ## Read in results:
  i<- which(counties == k)
  a_eval<- read.csv(paste0("Summer_results/ORL_NWS_eval_samp-R_obs-W_", r_model, "_fips_", k, ".csv"))
  
  ph_eval<- read.csv(paste0("Summer_results/ORL_random_eval_samp-R_obs-W_", r_model,"_Rstr-HI-", Bench[i,"rqhi_ot"], "_fips_", k, ".csv"))
  ah_eval<- read.csv(paste0("Summer_results/ORL_AA_eval_samp-R_obs-W_", r_model,"_Rstr-HI-", Bench[i,"aqhi_ot"], "_fips_", k, ".csv"))
  
  ## Calculate stats:
  n_eps<- nrow(a_eval)/(n_days-1)
  Days<- rep(1:(n_days-1),n_eps)
  
  aeD<- Days[which(a_eval$Actions == 1)]
  
  pheD<- Days[which(ph_eval$Actions == 1)]
  aheD<- Days[which(ah_eval$Actions == 1)]
  
  Eval_DOS<- rbind(Eval_DOS, data.frame(County = k, Policy="NWS", Value = aeD))
  Eval_DOS<- rbind(Eval_DOS, data.frame(County = k, Policy="Random-QHI", Value = pheD))
  Eval_DOS<- rbind(Eval_DOS, data.frame(County = k, Policy="Always-QHI", Value = aheD))
  
  Eval_Strk.Ln<- rbind(Eval_Strk.Ln, data.frame(County = k, Policy="NWS", Value = streaks(aeD)))
  Eval_Strk.Ln<- rbind(Eval_Strk.Ln, data.frame(County = k, Policy="Random-QHI", Value = streaks(pheD)))
  Eval_Strk.Ln<- rbind(Eval_Strk.Ln, data.frame(County = k, Policy="Always-QHI", Value = streaks(aheD)))
  
  print(k)
}

write.csv(Eval_DOS, paste0("Fall_results/Eval_DOS_", r_model, "_benchmarks.csv"))
write.csv(Eval_Strk.Ln, paste0("Fall_results/Eval_Strk-Ln_", r_model, "_benchmarks.csv"))


## RL stats:
for(k in counties){
  ## Read in results:
  i<- which(counties == k)
  
  q_f.q_d10_eval<- read.csv(paste0("Summer_results/ORL_RL_eval_samp-R_obs-W_", 
                           "Tune_F-Q_D10_Rstr-HI-", RL_F.q_d10[i, "OT"], 
                           "_arch-", RL_F.q_d10[i, "NHL"], "-", RL_F.q_d10[i, "NHU"],
                           "_ns-", RL_F.q_d10[i, "n_steps"], "_fips-", k, "_fips_", k, ".csv"))
  q_f.none_eval<- read.csv(paste0("Summer_results/ORL_RL_eval_samp-R_obs-W_",
                           "Tune_F-none_Rstr-HI-", RL_F.none[i, "OT"],
                           "_arch-", RL_F.none[i, "NHL"], "-", RL_F.none[i, "NHU"],
                           "_ns-", RL_F.none[i, "n_steps"], "_fips-", k, "_fips_", k, ".csv"))
  
  ## Calculate stats:
  n_eps<- nrow(q_f.q_d10_eval)/(n_days-1)
  Days<- rep(1:(n_days-1),n_eps)
  
  q1eD<- Days[which(q_f.q_d10_eval$Actions == 1)]
  Eval_DOS<- rbind(Eval_DOS, data.frame(County = k, Policy="TRPO.QHI.F", Value = q1eD))
  Eval_Strk.Ln<- rbind(Eval_Strk.Ln, data.frame(County = k, Policy="TRPO.QHI.F", Value = streaks(q1eD)))
  
  q2eD<- Days[which(q_f.none_eval$Actions == 1)]
  Eval_DOS<- rbind(Eval_DOS, data.frame(County = k, Policy="TRPO.QHI", Value = q2eD))
  Eval_Strk.Ln<- rbind(Eval_Strk.Ln, data.frame(County = k, Policy="TRPO.QHI", Value = streaks(q2eD)))
  
  print(k)
}

write.csv(Eval_DOS, paste0("Fall_results/Eval_DOS_", r_model, "_RL.csv"))
write.csv(Eval_Strk.Ln, paste0("Fall_results/Eval_Strk-Ln_", r_model, "_RL.csv"))


############### Comparing to the benchmarks:
Eval_DOS<- read.csv(paste0("Fall_results/Eval_DOS_", r_model, "_benchmarks.csv"))
Eval_DOS<- rbind(Eval_DOS, read.csv(paste0("Fall_results/Eval_DOS_", r_model, "_RL.csv")))
Eval_Strk.Ln<- read.csv(paste0("Fall_results/Eval_Strk-Ln_", r_model, "_benchmarks.csv"))
Eval_Strk.Ln<- rbind(Eval_Strk.Ln, read.csv(paste0("Fall_results/Eval_Strk-Ln_", r_model, "_RL.csv")))

d1<- ggplot(Eval_DOS[which(Eval_DOS$Policy %in% c("NWS", "Random-QHI")),], aes(x=Value, fill=Policy)) + 
  geom_histogram(alpha=0.4, position="identity", aes(y = ..density..)) +
  ggtitle("Day of Summer (a)") + 
  ylab("Density") + xlab("Day of Summer") +
  theme(legend.position="bottom")

d2<- ggplot(Eval_DOS[which(Eval_DOS$Policy %in% c("Always-QHI", "TRPO.QHI")),], aes(x=Value, fill=Policy)) + 
  geom_histogram(alpha=0.4, position="identity", aes(y = ..density..)) +
  ggtitle("Day of Summer (b)") +
  ylab("Density") + xlab("Day of Summer") +
  theme(legend.position="bottom")

s1<- ggplot(Eval_Strk.Ln[which(Eval_Strk.Ln$Policy %in% c("NWS", "Random-QHI")),], aes(x=Value, fill=Policy)) + 
  geom_histogram(alpha=0.4, position="identity") + 
  ggtitle("Streak Length (a)") +
  scale_y_continuous(trans = "sqrt") + 
  ylab("Sqrt(Count)") + xlab("Streak Lengths") +
  theme(legend.position="bottom")

s2<- ggplot(Eval_Strk.Ln[which(Eval_Strk.Ln$Policy %in% c("Always-QHI", "TRPO.QHI")),], aes(x=Value, fill=Policy)) + 
  geom_histogram(alpha=0.4, position="identity") + 
  ggtitle("Streak Length (b)") +
  scale_y_continuous(trans = "sqrt") + 
  ylab("Sqrt(Count)") + xlab("Streak Lengths") +
  theme(legend.position="bottom")


plot_grid(d1, d2, nrow=1)
plot_grid(s1, s2, nrow=1)

## All together:

Eval_DOS$Policy<- as.factor(Eval_DOS$Policy)
levels(Eval_DOS$Policy)<- c("A.QHI", "NWS", "R.QHI", "TRPO.QHI", "TRPO.QHI.F")
Eval_Strk.Ln$Policy<- as.factor(Eval_Strk.Ln$Policy)
levels(Eval_Strk.Ln$Policy)<- c("A.QHI", "NWS", "R.QHI", "TRPO.QHI", "TRPO.QHI.F")

D<- ggplot(Eval_DOS[which(Eval_DOS$Policy %in% c("NWS", "R.QHI", "A.QHI", "TRPO.QHI")),], aes(x=Value, fill=Policy)) + 
  geom_histogram(alpha=0.4, position="identity", aes(y = ..density..)) +
  ggtitle("Alert Density Across Days of Summer (After May 1)") + 
  ylab("Density") + xlab("Day of Summer") +
  theme(legend.position="bottom")

S<- ggplot(Eval_Strk.Ln[which(Eval_Strk.Ln$Policy %in% c("NWS", "R.QHI", "A.QHI", "TRPO.QHI")),], 
       aes(x=Value, fill=Policy)) + 
  geom_histogram(alpha=0.4, position="identity") + 
  ggtitle("Density of Alert Streak Lengths") +
  scale_y_continuous(trans = "sqrt") + 
  ylab("Sqrt(Count)") + xlab("Streak Lengths") +
  theme(legend.position="bottom")

plot_grid(D, S, nrow=1, rel_widths = c(2,1.75))

## Comparing RL models:

d<- ggplot(Eval_DOS[which(Eval_DOS$Policy %in% c("TRPO.QHI", "TRPO.QHI.F")),], aes(x=Value, fill=Policy)) + 
  geom_histogram(alpha=0.4, position="identity", aes(y = ..density..)) +
  ggtitle("Alert Density Across Days of Summer") + 
  ylab("Density") + xlab("Day of Summer") +
  theme(legend.position="bottom")

s<- ggplot(Eval_Strk.Ln[which(Eval_Strk.Ln$Policy %in% c("TRPO.QHI", "TRPO.QHI.F")),], 
           aes(x=Value, fill=Policy)) + 
  geom_histogram(alpha=0.4, position="identity") + 
  ggtitle("Density of Alert Streak Lengths") +
  scale_y_continuous(trans = "sqrt") + 
  ylab("Sqrt(Count)") + xlab("Streak Lengths") +
  theme(legend.position="bottom")

plot_grid(d, s, nrow=1, rel_widths = c(2,1.5))


