
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
results<- read.csv("Fall_results/Final_eval_30_best-T7-T8.csv")

for(k in counties){
  ## Read in results:
  p_train<- read.csv(paste0("Summer_results/ORL_random_train_samp-R_obs-W_test_fips_", k, ".csv"))
  p_eval<- read.csv(paste0("Summer_results/ORL_random_eval_samp-R_obs-W_test_fips_", k, ".csv"))
  
  a_train<- read.csv(paste0("Summer_results/ORL_NWS_train_samp-R_obs-W_test_fips_", k, ".csv"))
  a_eval<- read.csv(paste0("Summer_results/ORL_NWS_eval_samp-R_obs-W_test_fips_", k, ".csv"))

  i<- which(counties == k)
  if(results[i,"Best_Model"] == "NN_2-16"){
    q_train<- read.csv(paste0("Summer_results/ORL_RL_train_samp-R_obs-W_", "T8", "_fips-", k, "_", "Rstr-HI-", results[i,"opt_HI_thr"], "_fips_", k, ".csv"))
    q_eval<- read.csv(paste0("Summer_results/ORL_RL_eval_samp-R_obs-W_", "T8", "_fips-", k, "_", "Rstr-HI-", results[i,"opt_HI_thr"], "_fips_", k, ".csv"))
  }else{
    q_train<- read.csv(paste0("Summer_results/ORL_RL_train_samp-R_obs-W_", "T7", "_fips-", k, "_", "Rstr-HI-", results[i,"opt_HI_thr"], "_fips_", k, ".csv"))
    q_eval<- read.csv(paste0("Summer_results/ORL_RL_eval_samp-R_obs-W_", "T7", "_fips-", k, "_", "Rstr-HI-", results[i,"opt_HI_thr"], "_fips_", k, ".csv"))
  }
  
  ## Calculate stats:
  n_eps<- nrow(p_train)/(n_days-1)
  Days<- rep(1:(n_days-1),n_eps)
  
  ptD<- Days[which(p_train$Actions == 1)]
  peD<- Days[which(p_eval$Actions == 1)]
  
  atD<- Days[which(a_train$Actions == 1)]
  aeD<- Days[which(a_eval$Actions == 1)]
  
  qtD<- Days[which(q_train$Actions == 1)]
  qeD<- Days[which(q_eval$Actions == 1)]
  
  Train_DOS<- rbind(Train_DOS, data.frame(Policy="Random", Value = ptD))
  Train_DOS<- rbind(Train_DOS, data.frame(Policy="NWS", Value = atD))
  Train_DOS<- rbind(Train_DOS, data.frame(Policy="RL", Value = qtD))
  
  Eval_DOS<- rbind(Eval_DOS, data.frame(Policy="Random", Value = peD))
  Eval_DOS<- rbind(Eval_DOS, data.frame(Policy="NWS", Value = aeD))
  Eval_DOS<- rbind(Eval_DOS, data.frame(Policy="RL", Value = qeD))
  
  Train_Strk.Ln<- rbind(Train_Strk.Ln, data.frame(Policy="Random", Value = streaks(ptD)))
  Train_Strk.Ln<- rbind(Train_Strk.Ln, data.frame(Policy="NWS", Value = streaks(atD)))
  Train_Strk.Ln<- rbind(Train_Strk.Ln, data.frame(Policy="RL", Value = streaks(qtD)))
  
  Eval_Strk.Ln<- rbind(Eval_Strk.Ln, data.frame(Policy="Random", Value = streaks(peD)))
  Eval_Strk.Ln<- rbind(Eval_Strk.Ln, data.frame(Policy="NWS", Value = streaks(aeD)))
  Eval_Strk.Ln<- rbind(Eval_Strk.Ln, data.frame(Policy="RL", Value = streaks(qeD)))
  
  print(k)
}


#### Make histograms:

d1<- ggplot(Train_DOS, aes(x=Value, fill=Policy)) + 
  geom_histogram(alpha=0.4, position="identity", aes(y = ..density..)) +
  ggtitle("Day of Summer (Training)") + 
  ylab("Density") + xlab("Day of Summer")

d2<- ggplot(Eval_DOS, aes(x=Value, fill=Policy)) + 
  geom_histogram(alpha=0.4, position="identity", aes(y = ..density..)) +
  ggtitle("Day of Summer (Evaluation)") +
  theme(legend.position="none") + 
  ylab("Density") + xlab("Day of Summer")

s1<- ggplot(Train_Strk.Ln, aes(x=Value, fill=Policy)) + 
  geom_histogram(alpha=0.4, position="identity") + 
  ggtitle("Streak Length (Training)") +
  theme(legend.position="none") +
  scale_y_continuous(trans = "sqrt") + 
  ylab("Sqrt(Count)") + xlab("Streak Lengths")

s2<- ggplot(Eval_Strk.Ln, aes(x=Value, fill=Policy)) + 
  geom_histogram(alpha=0.4, position="identity") + 
  ggtitle("Streak Length (Evaluation)") +
  theme(legend.position="none") +
  scale_y_continuous(trans = "sqrt") + 
  ylab("Sqrt(Count)") + xlab("Streak Lengths")

legend <- get_legend(d1)

plot_grid(d1 + theme(legend.position="none"), 
          d2, legend, rel_widths = c(2, 2, .5), nrow=1)
plot_grid(s1, s2, legend, rel_widths = c(2, 2, .5), nrow=1)



