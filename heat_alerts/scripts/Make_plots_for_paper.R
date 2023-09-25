
library(ggplot2)
library(cowplot, lib.loc = "~/apps/R_4.2.2")

n_days<- 153

Train_DOS<- data.frame(matrix(ncol = 2, nrow = 0))
names(Train_DOS)<- c("Policy", "Value")
Train_Strk.Ln<- data.frame(matrix(ncol = 2, nrow = 0))
names(Train_Strk.Ln)<- c("Policy", "Value")
Eval_DOS<- data.frame(matrix(ncol = 2, nrow = 0))
names(Eval_DOS)<- c("Policy", "Value")
Eval_Strk.Ln<- data.frame(matrix(ncol = 2, nrow = 0))
names(Eval_Strk.Ln)<- c("Policy", "Value")

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

for(k in counties){
  ## Read in results:
  # p_train<- read.csv(paste0("Summer_results/ORL_random_train_samp-R_obs-W_test_fips_", k, ".csv"))
  p_eval<- read.csv(paste0("Summer_results/ORL_random_eval_samp-R_obs-W_", r_model,"_fips_", k, ".csv"))
  
  i<- which(counties == k)
  ph_eval<- read.csv(paste0("Summer_results/ORL_random_eval_samp-R_obs-W_", r_model,"_Rstr-HI-", random_qhi[i,"opt_HI_thr"], "_fips_", k, ".csv"))
  
  ah_eval<- read.csv(paste0("Summer_results/ORL_AA_eval_samp-R_obs-W_", r_model,"_Rstr-HI-", aa_qhi[i,"opt_HI_thr"], "_fips_", k, ".csv"))
  
  tk_eval<- read.csv(paste0("Summer_results/ORL_TK_eval_samp-R_obs-W_", r_model, "_fips_", k, ".csv"))
  
  # a_train<- read.csv(paste0("Summer_results/ORL_NWS_train_samp-R_obs-W_", r_model, "_fips_", k, ".csv"))
  a_eval<- read.csv(paste0("Summer_results/ORL_NWS_eval_samp-R_obs-W_", r_model, "_fips_", k, ".csv"))

  if(r_model == "test"){
    if(results[i,"Best_Model"] == "NN_2-16"){
      q_train<- read.csv(paste0("Summer_results/ORL_RL_train_samp-R_obs-W_", "T8", "_fips-", k, "_", "Rstr-HI-", results[i,"opt_HI_thr"], "_fips_", k, ".csv"))
      q_eval<- read.csv(paste0("Summer_results/ORL_RL_eval_samp-R_obs-W_", "T8", "_fips-", k, "_", "Rstr-HI-", results[i,"opt_HI_thr"], "_fips_", k, ".csv"))
    }else{
      q_train<- read.csv(paste0("Summer_results/ORL_RL_train_samp-R_obs-W_", "T7", "_fips-", k, "_", "Rstr-HI-", results[i,"opt_HI_thr"], "_fips_", k, ".csv"))
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
  n_eps<- nrow(p_eval)/(n_days-1)
  Days<- rep(1:(n_days-1),n_eps)
  
  # ptD<- Days[which(p_train$Actions == 1)]
  peD<- Days[which(p_eval$Actions == 1)]
  
  pheD<- Days[which(ph_eval$Actions == 1)]
  aheD<- Days[which(ah_eval$Actions == 1)]
  tkeD<- Days[which(tk_eval$Actions == 1)]
  
  # atD<- Days[which(a_train$Actions == 1)]
  aeD<- Days[which(a_eval$Actions == 1)]
  
  # qtD<- Days[which(q_train$Actions == 1)]
  qeD<- Days[which(q_eval$Actions == 1)]
  
  # Train_DOS<- rbind(Train_DOS, data.frame(Policy="Random", Value = ptD))
  # Train_DOS<- rbind(Train_DOS, data.frame(Policy="NWS", Value = atD))
  # Train_DOS<- rbind(Train_DOS, data.frame(Policy="RL", Value = qtD))
  
  Eval_DOS<- rbind(Eval_DOS, data.frame(Policy="Random", Value = peD))
  Eval_DOS<- rbind(Eval_DOS, data.frame(Policy="Top K", Value = tkeD))
  Eval_DOS<- rbind(Eval_DOS, data.frame(Policy="Random-QHI", Value = pheD))
  Eval_DOS<- rbind(Eval_DOS, data.frame(Policy="Always-QHI", Value = aheD))
  Eval_DOS<- rbind(Eval_DOS, data.frame(Policy="NWS", Value = aeD))
  Eval_DOS<- rbind(Eval_DOS, data.frame(Policy="RL", Value = qeD))
  
  # Train_Strk.Ln<- rbind(Train_Strk.Ln, data.frame(Policy="Random", Value = streaks(ptD)))
  # Train_Strk.Ln<- rbind(Train_Strk.Ln, data.frame(Policy="NWS", Value = streaks(atD)))
  # Train_Strk.Ln<- rbind(Train_Strk.Ln, data.frame(Policy="RL", Value = streaks(qtD)))
  
  Eval_Strk.Ln<- rbind(Eval_Strk.Ln, data.frame(Policy="Random", Value = streaks(peD)))
  Eval_Strk.Ln<- rbind(Eval_Strk.Ln, data.frame(Policy="Top K", Value = streaks(tkeD)))
  Eval_Strk.Ln<- rbind(Eval_Strk.Ln, data.frame(Policy="Random-QHI", Value = streaks(pheD)))
  Eval_Strk.Ln<- rbind(Eval_Strk.Ln, data.frame(Policy="Always-QHI", Value = streaks(aheD)))
  Eval_Strk.Ln<- rbind(Eval_Strk.Ln, data.frame(Policy="NWS", Value = streaks(aeD)))
  Eval_Strk.Ln<- rbind(Eval_Strk.Ln, data.frame(Policy="RL", Value = streaks(qeD)))
  
  print(k)
}

write.csv(Eval_DOS[which(Eval_DOS$Policy == "RL"),], paste0("Fall_results/Eval_DOS_", r_model, ".csv"))
write.csv(Eval_Strk.Ln[which(Eval_Strk.Ln$Policy == "RL"),], paste0("Fall_results/Eval_Strk-Ln_", r_model, ".csv"))


#### Make histograms:

nc_rl_dos<- read.csv(paste0("Fall_results/Eval_DOS_NC_model.csv"))
nc_rl_strk<- read.csv(paste0("Fall_results/Eval_Strk-Ln_NC_model.csv"))
c_rl_dos<- read.csv(paste0("Fall_results/Eval_DOS_test.csv"))
c_rl_strk<- read.csv(paste0("Fall_results/Eval_Strk-Ln_test.csv"))

nc_rl_dos$Policy<- "NC Rewards"
nc_rl_strk$Policy<- "NC Rewards"
c_rl_dos$Policy<- "Rewards"
c_rl_strk$Policy<- "Rewards"

p1<- ggplot(rbind(nc_rl_dos, c_rl_dos), aes(x=Value, fill=Policy)) + 
  geom_histogram(alpha=0.4, position="identity", aes(y = ..density..)) +
  ggtitle("Day of Summer") + 
  ylab("Density") + xlab("Day of Summer") +
  theme(legend.position="bottom")

p2<- ggplot(rbind(nc_rl_strk, c_rl_strk), aes(x=Value, fill=Policy)) + 
  geom_histogram(alpha=0.4, position="identity", bins = 15) + 
  ggtitle("Streak Length") +
  scale_y_continuous(trans = "sqrt") + 
  ylab("Sqrt(Count)") + xlab("Streak Lengths") +
  theme(legend.position="bottom")

plot_grid(p1, p2, nrow=1)


############### Comparing to the benchmarks:

d1<- ggplot(Eval_DOS[which(Eval_DOS$Policy %in% c("Top K", "Always-QHI", "Random")),], aes(x=Value, fill=Policy)) + 
  geom_histogram(alpha=0.4, position="identity", aes(y = ..density..)) +
  ggtitle("Day of Summer (a)") + 
  ylab("Density") + xlab("Day of Summer") +
  theme(legend.position="bottom")

d2<- ggplot(Eval_DOS[which(Eval_DOS$Policy %in% c("NWS", "Random-QHI", "RL")),], aes(x=Value, fill=Policy)) + 
  geom_histogram(alpha=0.4, position="identity", aes(y = ..density..)) +
  ggtitle("Day of Summer (b)") +
  ylab("Density") + xlab("Day of Summer") +
  theme(legend.position="bottom")

s1<- ggplot(Eval_Strk.Ln[which(Eval_Strk.Ln$Policy %in% c("Top K", "Always-QHI", "Random")),], aes(x=Value, fill=Policy)) + 
  geom_histogram(alpha=0.4, position="identity") + 
  ggtitle("Streak Length (a)") +
  scale_y_continuous(trans = "sqrt") + 
  ylab("Sqrt(Count)") + xlab("Streak Lengths") +
  theme(legend.position="bottom")

s2<- ggplot(Eval_Strk.Ln[which(Eval_Strk.Ln$Policy %in% c("NWS", "Random-QHI", "RL")),], aes(x=Value, fill=Policy)) + 
  geom_histogram(alpha=0.4, position="identity") + 
  ggtitle("Streak Length (b)") +
  scale_y_continuous(trans = "sqrt") + 
  ylab("Sqrt(Count)") + xlab("Streak Lengths") +
  theme(legend.position="bottom")


plot_grid(d1, d2, nrow=1)
plot_grid(s1, s2, nrow=1)


####################################################### Including training stats:

d1<- ggplot(Train_DOS, aes(x=Value, fill=Policy)) + 
  geom_histogram(alpha=0.4, position="identity", aes(y = ..density..)) +
  ggtitle("Day of Summer (Training)") + 
  ylab("Density") + xlab("Day of Summer")

d2<- ggplot(Eval_DOS[-which(Eval_DOS$Policy%in% c("Random", "RL")),], aes(x=Value, fill=Policy)) + 
  geom_histogram(alpha=0.4, position="identity", aes(y = ..density..)) +
  ggtitle("Day of Summer (Evaluation)") +
  # theme(legend.position="none") + 
  ylab("Density") + xlab("Day of Summer")

s1<- ggplot(Train_Strk.Ln, aes(x=Value, fill=Policy)) + 
  geom_histogram(alpha=0.4, position="identity") + 
  ggtitle("Streak Length (Training)") +
  theme(legend.position="none") +
  scale_y_continuous(trans = "sqrt") + 
  ylab("Sqrt(Count)") + xlab("Streak Lengths")

s2<- ggplot(Eval_Strk.Ln[-which(Eval_Strk.Ln$Policy%in% c("Random", "RL")),], aes(x=Value, fill=Policy)) + 
  geom_histogram(alpha=0.4, position="identity") + 
  ggtitle("Streak Length (Evaluation)") +
  # theme(legend.position="none") +
  scale_y_continuous(trans = "sqrt") + 
  ylab("Sqrt(Count)") + xlab("Streak Lengths")

legend <- get_legend(d1)

plot_grid(d1 + theme(legend.position="none"), 
          d2, legend, rel_widths = c(2, 2, .5), nrow=1)
plot_grid(s1, s2, legend, rel_widths = c(2, 2, .5), nrow=1)



