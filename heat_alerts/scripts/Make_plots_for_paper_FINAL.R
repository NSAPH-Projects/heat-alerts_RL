
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

########## Boxplot:

DF<- data.frame(matrix(ncol = 4, nrow = 0))
names(DF)<- c("Policy", "Diff", "State", "Region")

for(pol in c("Zero", "Random", "Top_K", "Random_QHI", "AA_QHI", "basic_NWS")){
  DF<- rbind(DF, data.frame(Policy=pol, Diff=Bench[,pol] - Bench$NWS, 
                            State=state, Region=region))
}

# plain_RL<- read.csv("Fall_results/December_plain_RL_avg_return.csv")
# QHI_RL<- read.csv("Fall_results/December_Rstr-QHI_RL_avg_return.csv")
plain_RL<- read.csv("Fall_results/December_part-2_plain_RL_avg_return.csv")
QHI_RL<- read.csv("Fall_results/December_part-2_Rstr-QHI_RL_avg_return.csv")

algos<- unique(plain_RL$Algo)
forecasts<- unique(plain_RL$Forecast)

for(algo in algos){
  for(forecast in forecasts){
    a<- "TRPO"
    if(algo == "dqn"){
      a<- "DQN"
    }else if(algo == "a2c"){
      a<- "A2C"
    }else if(algo == "qrdqn"){
      a<- "QRDQN"
    }
    
    f<- ".F"
    if(forecast == "none"){
      f<- ""
    }
    
    df<- plain_RL[which(plain_RL$Algo == algo & plain_RL$Forecast == forecast),]
    DF<- rbind(DF, data.frame(Policy=paste0(a,f), Diff=df$Eval - Bench$NWS, 
                              State=state, Region=region))
    
    df<- QHI_RL[which(QHI_RL$Algo == algo & QHI_RL$Forecast == forecast),]
    DF<- rbind(DF, data.frame(Policy=paste0(a,".QHI",f), Diff=df$Eval - Bench$NWS, 
                              State=state, Region=region))
    
  }
}


plot_DF<- DF[which(DF$Policy %in% c("Top_K", "AA_QHI", # "DQN.QHI", 
                                    "TRPO.QHI", # "TRPO.QHI.F"
                                    "A2C.QHI"
                                    )),]
plot_DF$Policy<- factor(plot_DF$Policy,
                        levels=c("Top_K", "AA_QHI", # "DQN.QHI", 
                                 "TRPO.QHI", # "TRPO.QHI.F"
                                 "A2C.QHI"
                                 ))
levels(plot_DF$Policy)<- c("TopK", "AA.QHI", #"DQN.QHI", 
                           "TRPO.QHI (RL)", # "TRPO.QHI.F"
                           "A2C.QHI (RL)"
                           )

plot_DF$outlier<- (plot_DF$Diff > 0.1) | (plot_DF$Diff < -0.05)
plot_DF$State[which(!plot_DF$outlier)]<- NA

ggplot(plot_DF, aes(x=Policy, y=Diff, color = Region)) +
  geom_hline(yintercept=0) + 
  geom_boxplot() +
  geom_point(position = position_jitterdodge(seed=1), alpha=0.5) +
  # geom_text(aes(label=State), na.rm=TRUE, size=2) +
  ggrepel::geom_text_repel(aes(label=State), na.rm=TRUE, size=2,
                           position = position_jitterdodge(seed=1)) +
  ylab("Policy Return - NWS Return") + 
  ggtitle("Comparison to NWS: Average Return on Evaluation Years")


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


## RL stats: TRPO and DQN
for(k in counties){
  ## Read in results:
  i_t<- which(QHI_RL$County == k & QHI_RL$Algo == "trpo" & QHI_RL$Forecast == "none")
  i_d<- which(QHI_RL$County == k & QHI_RL$Algo == "dqn" & QHI_RL$Forecast == "none")
  i_t_f<- which(QHI_RL$County == k & QHI_RL$Algo == "trpo" & QHI_RL$Forecast == "Q_D10")
  
  q_t<- read.csv(paste0("Summer_results/ORL_RL_eval_samp-R_obs-W_December_", 
                        "trpo_F-none_Rstr-HI-", QHI_RL[i_t, "OT"], 
                        "_arch-", QHI_RL[i_t, "NHL"], "-", QHI_RL[i_t, "NHU"],
                        "_ns-", QHI_RL[i_t, "n_steps"], "_fips-", k, "_fips_", k, ".csv"))
  
  q_d<- read.csv(paste0("Summer_results/ORL_RL_eval_samp-R_obs-W_December_", 
                        "dqn_F-none_Rstr-HI-", QHI_RL[i_d, "OT"], 
                        "_arch-", QHI_RL[i_d, "NHL"], "-", QHI_RL[i_d, "NHU"],
                        "_ns-", QHI_RL[i_d, "n_steps"], "_fips-", k, "_fips_", k, ".csv"))
  
  q_t_f<- read.csv(paste0("Summer_results/ORL_RL_eval_samp-R_obs-W_December_", 
                        "trpo_F-Q_D10_Rstr-HI-", QHI_RL[i_t_f, "OT"], 
                        "_arch-", QHI_RL[i_t_f, "NHL"], "-", QHI_RL[i_t_f, "NHU"],
                        "_ns-", QHI_RL[i_t_f, "n_steps"], "_fips-", k, "_fips_", k, ".csv"))
  
  ## Calculate stats:
  n_eps<- nrow(q_t)/(n_days-1)
  Days<- rep(1:(n_days-1),n_eps)
  
  qteD<- Days[which(q_t$Actions == 1)]
  Eval_DOS<- rbind(Eval_DOS, data.frame(County = k, Policy="TRPO.QHI", Value = qteD))
  Eval_Strk.Ln<- rbind(Eval_Strk.Ln, data.frame(County = k, Policy="TRPO.QHI", Value = streaks(qteD)))
  
  qdeD<- Days[which(q_d$Actions == 1)]
  Eval_DOS<- rbind(Eval_DOS, data.frame(County = k, Policy="DQN.QHI", Value = qdeD))
  Eval_Strk.Ln<- rbind(Eval_Strk.Ln, data.frame(County = k, Policy="DQN.QHI", Value = streaks(qdeD)))
  
  qtfeD<- Days[which(q_t_f$Actions == 1)]
  Eval_DOS<- rbind(Eval_DOS, data.frame(County = k, Policy="TRPO.QHI.F", Value = qtfeD))
  Eval_Strk.Ln<- rbind(Eval_Strk.Ln, data.frame(County = k, Policy="TRPO.QHI.F", Value = streaks(qtfeD)))
  
  print(k)
}

write.csv(Eval_DOS, paste0("Fall_results/December_Eval_DOS_", r_model, "_RL.csv"))
write.csv(Eval_Strk.Ln, paste0("Fall_results/December_Eval_Strk-Ln_", r_model, "_RL.csv"))

## RL stats: A2C and QRDQN
for(k in counties){
  ## Read in results:
  i_a<- which(QHI_RL$County == k & QHI_RL$Algo == "a2c" & QHI_RL$Forecast == "none")
  i_q<- which(QHI_RL$County == k & QHI_RL$Algo == "qrdqn" & QHI_RL$Forecast == "none")
  i_a_f<- which(QHI_RL$County == k & QHI_RL$Algo == "a2c" & QHI_RL$Forecast == "Q_D10")
  
  q_a<- read.csv(paste0("Summer_results/ORL_RL_eval_samp-R_obs-W_December_", 
                        "a2c_F-none_Rstr-HI-", QHI_RL[i_a, "OT"], 
                        "_arch-", QHI_RL[i_a, "NHL"], "-", QHI_RL[i_a, "NHU"],
                        "_ns-", QHI_RL[i_a, "n_steps"], "_fips-", k, "_fips_", k, ".csv"))
  
  q_q<- read.csv(paste0("Summer_results/ORL_RL_eval_samp-R_obs-W_December_", 
                        "qrdqn_F-none_Rstr-HI-", QHI_RL[i_q, "OT"], 
                        "_arch-", QHI_RL[i_q, "NHL"], "-", QHI_RL[i_q, "NHU"],
                        "_ns-", QHI_RL[i_q, "n_steps"], "_fips-", k, "_fips_", k, ".csv"))
  
  q_a_f<- read.csv(paste0("Summer_results/ORL_RL_eval_samp-R_obs-W_December_", 
                          "a2c_F-Q_D10_Rstr-HI-", QHI_RL[i_a_f, "OT"], 
                          "_arch-", QHI_RL[i_a_f, "NHL"], "-", QHI_RL[i_a_f, "NHU"],
                          "_ns-", QHI_RL[i_a_f, "n_steps"], "_fips-", k, "_fips_", k, ".csv"))
  
  ## Calculate stats:
  n_eps<- nrow(q_a)/(n_days-1)
  Days<- rep(1:(n_days-1),n_eps)
  
  qaeD<- Days[which(q_a$Actions == 1)]
  Eval_DOS<- rbind(Eval_DOS, data.frame(County = k, Policy="A2C.QHI", Value = qaeD))
  Eval_Strk.Ln<- rbind(Eval_Strk.Ln, data.frame(County = k, Policy="A2C.QHI", Value = streaks(qaeD)))
  
  qqeD<- Days[which(q_q$Actions == 1)]
  Eval_DOS<- rbind(Eval_DOS, data.frame(County = k, Policy="QRDQN.QHI", Value = qqeD))
  Eval_Strk.Ln<- rbind(Eval_Strk.Ln, data.frame(County = k, Policy="QRDQN.QHI", Value = streaks(qqeD)))
  
  qafeD<- Days[which(q_a_f$Actions == 1)]
  Eval_DOS<- rbind(Eval_DOS, data.frame(County = k, Policy="A2C.QHI.F", Value = qafeD))
  Eval_Strk.Ln<- rbind(Eval_Strk.Ln, data.frame(County = k, Policy="A2C.QHI.F", Value = streaks(qafeD)))
  
  print(k)
}

write.csv(Eval_DOS, paste0("Fall_results/December_part-2_Eval_DOS_", r_model, "_RL.csv"))
write.csv(Eval_Strk.Ln, paste0("Fall_results/December_part-2_Eval_Strk-Ln_", r_model, "_RL.csv"))



############### Comparing to the benchmarks:
Eval_DOS<- read.csv(paste0("Fall_results/Eval_DOS_", r_model, "_benchmarks.csv"))
# Eval_DOS<- rbind(Eval_DOS, read.csv(paste0("Fall_results/December_Eval_DOS_", r_model, "_RL.csv")))
Eval_DOS<- rbind(Eval_DOS, read.csv(paste0("Fall_results/December_part-2_Eval_DOS_", r_model, "_RL.csv")))
Eval_Strk.Ln<- read.csv(paste0("Fall_results/Eval_Strk-Ln_", r_model, "_benchmarks.csv"))
# Eval_Strk.Ln<- rbind(Eval_Strk.Ln, read.csv(paste0("Fall_results/December_Eval_Strk-Ln_", r_model, "_RL.csv")))
Eval_Strk.Ln<- rbind(Eval_Strk.Ln, read.csv(paste0("Fall_results/December_part-2_Eval_Strk-Ln_", r_model, "_RL.csv")))

## All together:

Eval_DOS$Policy<- as.factor(Eval_DOS$Policy)
# levels(Eval_DOS$Policy)<- c("AA.QHI", "DQN.QHI", "NWS", "R.QHI", "TRPO.QHI (RL)", "TRPO.QHI.F")
levels(Eval_DOS$Policy)<- c("A2C.QHI (RL)", "A2C.QHI.F", "AA.QHI", "NWS", "QRDQN.QHI", "R.QHI")
Eval_Strk.Ln$Policy<- as.factor(Eval_Strk.Ln$Policy)
# levels(Eval_Strk.Ln$Policy)<- c("AA.QHI", "DQN.QHI", "NWS", "R.QHI", "TRPO.QHI (RL)", "TRPO.QHI.F")
levels(Eval_Strk.Ln$Policy)<- c("A2C.QHI (RL)", "A2C.QHI.F", "AA.QHI", "NWS", "QRDQN.QHI", "R.QHI")

Eval_DOS$Policy<- relevel(Eval_DOS$Policy, "NWS")
Eval_DOS$Policy<- relevel(Eval_DOS$Policy, "AA.QHI")

Eval_Strk.Ln$Policy<- relevel(Eval_Strk.Ln$Policy, "NWS")
Eval_Strk.Ln$Policy<- relevel(Eval_Strk.Ln$Policy, "AA.QHI")

D<- ggplot(Eval_DOS[which(Eval_DOS$Policy %in% c("NWS", "AA.QHI", "A2C.QHI (RL)" # "TRPO.QHI (RL)"
                                                 )),], aes(x=Value, fill=Policy)) + 
  geom_histogram(alpha=0.4, position="identity", aes(y = ..density..)) +
  ggtitle("Alert Density Across Days of Summer") + 
  ylab("Density") + xlab("Day of Summer") +
  theme(legend.position="bottom")

S<- ggplot(Eval_Strk.Ln[which(Eval_Strk.Ln$Policy %in% c("NWS", "AA.QHI", "A2C.QHI (RL)" # "TRPO.QHI (RL)"
                                                         )),], aes(x=Value, fill=Policy)) +
  geom_histogram(alpha=0.4, position="identity") + 
  ggtitle("Density of Alert Streak Lengths") +
  scale_y_continuous(trans = "sqrt") + 
  ylab("Sqrt(Count)") + xlab("Streak Lengths") +
  theme(legend.position="bottom")

plot_grid(D, S, nrow=1, rel_widths = c(2,1.5))


