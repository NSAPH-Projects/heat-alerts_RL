library(ggplot2)
library(cowplot, lib.loc = "~/apps/R_4.2.2")

n_days<- 153

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

counties<- c(41067, 53015, 20161, 37085, 48157,
             28049, 19153, 17167, 31153, 6071, 4013,
             34021, 19155, 17115, 29021, 29019, 5045, 40017, 21059,
             47113, 42017, 22109, 45015, 13031, 48367, 22063, 41053,
             32003, 4015, 6025)

r_model<-  "test" # "NC_model"
results<- read.csv("Fall_results/Final_eval_30_best-T7-T8.csv")
# results<- read.csv("Fall_results/Final_eval_30_NC1.csv")
random_qhi<- read.csv("Fall_results/Final_eval_30_test_random-w-rstr-hi.csv")
# random_qhi<- read.csv("Fall_results/Final_eval_30_NC_model_random-w-rstr-hi.csv")
aa_qhi<- read.csv("Fall_results/Final_eval_30_test_AA-w-rstr-hi.csv")
# aa_qhi<- read.csv("Fall_results/Final_eval_30_NC_model_AA-w-rstr-hi.csv")
fc_N<- read.csv("Fall_results/Final_eval_30_test_FC-num_elig-w-rstr-hi.csv")
fc_Q<- read.csv("Fall_results/Final_eval_30_test_FC-quantiles-w-rstr-hi.csv")
fc_D10<- read.csv("Fall_results/Final_eval_30_test_FC-ten_day-w-rstr-hi.csv")
fc_Av4<- read.csv("Fall_results/Final_eval_30_test_FC-quarters-w-rstr-hi.csv")
fc_All<- read.csv("Fall_results/Final_eval_30_test_FC-all-w-rstr-hi.csv")

Eval_DOS<- data.frame(matrix(ncol = 2, nrow = 0))
names(Eval_DOS)<- c("Policy", "Value")
Eval_Strk.Ln<- data.frame(matrix(ncol = 2, nrow = 0))
names(Eval_Strk.Ln)<- c("Policy", "Value")

for(k in counties){
  ## Read in results:
  i<- which(counties == k)
  a_eval<- read.csv(paste0("Summer_results/ORL_NWS_eval_samp-R_obs-W_", r_model, "_fips_", k, ".csv"))
  
  ph_eval<- read.csv(paste0("Summer_results/ORL_random_eval_samp-R_obs-W_", r_model,"_Rstr-HI-", random_qhi[i,"opt_HI_thr"], "_fips_", k, ".csv"))
  ah_eval<- read.csv(paste0("Summer_results/ORL_AA_eval_samp-R_obs-W_", r_model,"_Rstr-HI-", aa_qhi[i,"opt_HI_thr"], "_fips_", k, ".csv"))
  
  fc_n<- read.csv(paste0("Summer_results/ORL_RL_eval_samp-R_obs-W_FC1_fips-", k, "_Rstr-HI-", fc_N[i,"opt_HI_thr"], "_fips_", k, ".csv"))
  fc_q<- read.csv(paste0("Summer_results/ORL_RL_eval_samp-R_obs-W_FC1_fips-", k, "_Rstr-HI-", fc_Q[i,"opt_HI_thr"], "_FC-quantiles_fips_", k, ".csv"))
  fc_d10<- read.csv(paste0("Summer_results/ORL_RL_eval_samp-R_obs-W_FC1_fips-", k, "_Rstr-HI-", fc_D10[i,"opt_HI_thr"], "_FC-ten_day_fips_", k, ".csv"))
  fc_av4<- read.csv(paste0("Summer_results/ORL_RL_eval_samp-R_obs-W_FC1_fips-", k, "_Rstr-HI-", fc_Av4[i,"opt_HI_thr"], "_FC-quarters_fips_", k, ".csv"))
  fc_all<- read.csv(paste0("Summer_results/ORL_RL_eval_samp-R_obs-W_FC1_fips-", k, "_Rstr-HI-", fc_All[i,"opt_HI_thr"], "_FC-all_fips_", k, ".csv"))
  
  if(r_model == "test"){
    if(results[i,"Best_Model"] == "NN_2-16"){
      # q_train<- read.csv(paste0("Summer_results/ORL_RL_train_samp-R_obs-W_", "T8", "_fips-", k, "_", "Rstr-HI-", results[i,"opt_HI_thr"], "_fips_", k, ".csv"))
      q_eval<- read.csv(paste0("Summer_results/ORL_RL_eval_samp-R_obs-W_", "T8", "_fips-", k, "_", "Rstr-HI-", results[i,"opt_HI_thr"], "_fips_", k, ".csv"))
    }else{
      # q_train<- read.csv(paste0("Summer_results/ORL_RL_train_samp-R_obs-W_", "T7", "_fips-", k, "_", "Rstr-HI-", results[i,"opt_HI_thr"], "_fips_", k, ".csv"))
      q_eval<- read.csv(paste0("Summer_results/ORL_RL_eval_samp-R_obs-W_", "T7", "_fips-", k, "_", "Rstr-HI-", results[i,"opt_HI_thr"], "_fips_", k, ".csv"))
    }
  }else{
    if(results[i,"Best_model"] == "NN_2-16"){
      # q_train<- read.csv(paste0("Summer_results/ORL_RL_train_samp-R_obs-W_", "T8", "_fips-", k, "_", "Rstr-HI-", results[i,"opt_HI_thr"], "_fips_", k, ".csv"))
      q_eval<- read.csv(paste0("Summer_results/ORL_RL_eval_samp-R_obs-W_", "NC1", "_fips-", k, "_", "Rstr-HI-", results[i,"opt_HI_thr"], "_2-16", "_fips_", k, ".csv"))
    }else{
      # q_train<- read.csv(paste0("Summer_results/ORL_RL_train_samp-R_obs-W_", "T7", "_fips-", k, "_", "Rstr-HI-", results[i,"opt_HI_thr"], "_fips_", k, ".csv"))
      q_eval<- read.csv(paste0("Summer_results/ORL_RL_eval_samp-R_obs-W_", "NC1", "_fips-", k, "_", "Rstr-HI-", results[i,"opt_HI_thr"], "_1-16", "_fips_", k, ".csv"))
    }
  }
  
  ## Calculate stats:
  n_eps<- nrow(a_eval)/(n_days-1)
  Days<- rep(1:(n_days-1),n_eps)
  
  aeD<- Days[which(a_eval$Actions == 1)]
  
  pheD<- Days[which(ph_eval$Actions == 1)]
  aheD<- Days[which(ah_eval$Actions == 1)]
  
  qeD<- Days[which(q_eval$Actions == 1)]
  fcNeD<- Days[which(fc_n$Actions == 1)]
  fcQeD<- Days[which(fc_q$Actions == 1)]
  fcDeD<- Days[which(fc_d10$Actions == 1)]
  fcAeD<- Days[which(fc_av4$Actions == 1)]
  fcLeD<- Days[which(fc_all$Actions == 1)]
  
  Eval_DOS<- rbind(Eval_DOS, data.frame(Policy="NWS", Value = aeD))
  Eval_DOS<- rbind(Eval_DOS, data.frame(Policy="Random-QHI", Value = pheD))
  Eval_DOS<- rbind(Eval_DOS, data.frame(Policy="Always-QHI", Value = aheD))
  Eval_DOS<- rbind(Eval_DOS, data.frame(Policy="RL", Value = qeD))
  Eval_DOS<- rbind(Eval_DOS, data.frame(Policy="FC-N", Value = fcNeD))
  Eval_DOS<- rbind(Eval_DOS, data.frame(Policy="FC-Q", Value = fcQeD))
  Eval_DOS<- rbind(Eval_DOS, data.frame(Policy="FC-D10", Value = fcDeD))
  Eval_DOS<- rbind(Eval_DOS, data.frame(Policy="FC-Av4", Value = fcAeD))
  Eval_DOS<- rbind(Eval_DOS, data.frame(Policy="FC-All", Value = fcLeD))
  
  Eval_Strk.Ln<- rbind(Eval_Strk.Ln, data.frame(Policy="NWS", Value = streaks(aeD)))
  Eval_Strk.Ln<- rbind(Eval_Strk.Ln, data.frame(Policy="Random-QHI", Value = streaks(pheD)))
  Eval_Strk.Ln<- rbind(Eval_Strk.Ln, data.frame(Policy="Always-QHI", Value = streaks(aheD)))
  Eval_Strk.Ln<- rbind(Eval_Strk.Ln, data.frame(Policy="RL", Value = streaks(qeD)))
  Eval_Strk.Ln<- rbind(Eval_Strk.Ln, data.frame(Policy="FC-N", Value = streaks(fcNeD)))
  Eval_Strk.Ln<- rbind(Eval_Strk.Ln, data.frame(Policy="FC-Q", Value = streaks(fcQeD)))
  Eval_Strk.Ln<- rbind(Eval_Strk.Ln, data.frame(Policy="FC-D10", Value = streaks(fcDeD)))
  Eval_Strk.Ln<- rbind(Eval_Strk.Ln, data.frame(Policy="FC-Av4", Value = streaks(fcAeD)))
  Eval_Strk.Ln<- rbind(Eval_Strk.Ln, data.frame(Policy="FC-All", Value = streaks(fcLeD)))
  
  print(k)
}

# write.csv(Eval_DOS[which(Eval_DOS$Policy == "RL"),], paste0("Fall_results/Eval_DOS_", r_model, ".csv"))
# write.csv(Eval_Strk.Ln[which(Eval_Strk.Ln$Policy == "RL"),], paste0("Fall_results/Eval_Strk-Ln_", r_model, ".csv"))


############### Comparing to the benchmarks:

d1<- ggplot(Eval_DOS[which(Eval_DOS$Policy %in% c("Random-QHI", "Always-QHI", "RL")),], aes(x=Value, fill=Policy)) + 
  geom_histogram(alpha=0.4, position="identity", aes(y = ..density..)) +
  ggtitle("Day of Summer (a)") + 
  ylab("Density") + xlab("Day of Summer") +
  theme(legend.position="bottom")

d2<- ggplot(Eval_DOS[which(Eval_DOS$Policy %in% c("Always-QHI", "FC-N", "FC-Q", "NWS")),], aes(x=Value, fill=Policy)) + 
  geom_histogram(alpha=0.4, position="identity", aes(y = ..density..)) +
  ggtitle("Day of Summer (b)") +
  ylab("Density") + xlab("Day of Summer") +
  theme(legend.position="bottom")

s1<- ggplot(Eval_Strk.Ln[which(Eval_Strk.Ln$Policy %in% c("Random-QHI", "Always-QHI", "RL")),], aes(x=Value, fill=Policy)) + 
  geom_histogram(alpha=0.4, position="identity") + 
  ggtitle("Streak Length (a)") +
  scale_y_continuous(trans = "sqrt") + 
  ylab("Sqrt(Count)") + xlab("Streak Lengths") +
  theme(legend.position="bottom")

s2<- ggplot(Eval_Strk.Ln[which(Eval_Strk.Ln$Policy %in% c("Always-QHI", "FC-N", "FC-Q", "NWS")),], aes(x=Value, fill=Policy)) + 
  geom_histogram(alpha=0.4, position="identity") + 
  ggtitle("Streak Length (b)") +
  scale_y_continuous(trans = "sqrt") + 
  ylab("Sqrt(Count)") + xlab("Streak Lengths") +
  theme(legend.position="bottom")


plot_grid(d1, d2, nrow=1)
plot_grid(s1, s2, nrow=1)


