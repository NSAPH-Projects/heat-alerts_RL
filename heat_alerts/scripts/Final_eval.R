
library(dplyr)

library(reticulate)
np<- import("numpy")

## Define functions:
n_days<- 153

my_proc<- function(filename){
  f<- file.exists(filename)
  if(f){
    df<- read.csv(filename)[,-1]
    df$Count = 1
    agg_df<- aggregate(. ~ Year, df, sum)
    agg_df$Budget<- agg_df$Budget/(n_days-1)
    agg_df$budget_frac<- agg_df$Actions/agg_df$Budget
    agg_df$Frac<- agg_df$Count/sum(agg_df$Count)
    estimated_reward<- sum(agg_df$Rewards*(1/nrow(agg_df))/agg_df$Frac)/1000
    # return(list(agg_df, estimated_reward))
    return(estimated_reward)
  }else{
    return(NA)
  }
}

assess<- function(filename){
  f<- file.exists(filename)
  if(f){
    df<- read.csv(filename)[,-1]
    n_eps<- nrow(df)/(n_days-1)
    Days<- rep(1:(n_days-1),n_eps)
    D<- Days[which(df$Actions == 1)]
    if(length(D)>0){
      num_alerts<- length(D)/n_eps
      summary_dos<- summary(D)
      diffs<- D[2:length(D)] - D[1:(length(D)-1)]
      L<- rle(diffs)
      streaks<- L$lengths[which(L$values == 1)]
      num_streaks<- length(streaks)/n_eps
      avg_streak_length<- mean(streaks + 1)
      avg_streak_length_overall<- mean(c(streaks + 1, rep(1,length(D)-length(streaks))))
      b_50<- mean(D[which(df$B_50 == 1)], na.rm=TRUE)
      b_80<- mean(D[which(df$B_80 == 1)], na.rm=TRUE)
      b_100<- mean(D[which(df$B_100 == 1)], na.rm=TRUE)
      above_thresh_skipped<- sum(df$Above_Thresh_Skipped)/n_eps
      fraction_skipped<- above_thresh_skipped / num_alerts
      # return(list(agg_df, estimated_reward))
      # x<- c(num_alerts, as.vector(summary_dos), num_streaks, avg_streak_length, avg_streak_length_overall)
      x<- c(num_alerts, summary_dos["Min."], b_50, b_80, b_100, 
            num_streaks, avg_streak_length, avg_streak_length_overall,
            above_thresh_skipped, fraction_skipped)
      result<- data.frame(t(x))
      # names(result)<- c("AvNAl", "Min_dos", "Q1_dos", "Median_dos", "Mean_dos", "Q3_dos", "Max_dos", "AvNStrk", "AvStrkLn", "AvStrkLn_all")
      names(result)<- c("AvNAl", "Min_dos", "B_50pct", "B_80pct", "B_last", 
                        "AvNStrk", "AvStrkLn", "AvStrkLn_all",
                        "Abv_Skp", "Frac_Abv_Skp")
      return(result)
    }else{
      return(rep(NA,10))
    }
  }else{
    return(NA)
  }
}


### Identify optimal HI threshold and save the associated eval:

r_model<- "test" # "NC_model"
prefix<-"D1"  # "NC1" 
splitvar<- "dqn_Rstr-HI-" # "Rstr-HI-"
these<- c("DQN") # "TRPO", "PPO", "DQN", "LSTM", "QRDQN"
Algo<- rep(these, 4)
# type<- c("eval", "eval_samp", "train", "train_samp")
# Type<- rep(type, each=length(these))

# counties<- c(41067, 53015, 20161, 37085, 48157,
#              28049, 19153, 17167, 31153, 6071, 4013)
# 
# counties<- c(34021, 19155, 17115, 29021, 29019, 5045, 40017, 21059,
#              47113, 42017, 22109, 45015, 13031, 48367, 22063, 41053,
#              32003, 4015, 6025)

counties<- c(41067, 53015, 20161, 37085, 48157,
             28049, 19153, 17167, 31153, 6071, 4013,
             34021, 19155, 17115, 29021, 29019, 5045, 40017, 21059,
             47113, 42017, 22109, 45015, 13031, 48367, 22063, 41053,
             32003, 4015, 6025)

HI_thresholds<- seq(0.5, 0.9, 0.05)
opt_HI_thr_1_16<- rep(0, length(counties))
opt_HI_thr_2_16<- rep(0, length(counties))
Eval_samp_1_16<- rep(0, length(counties))
Eval_samp_2_16<- rep(0, length(counties))
Eval_1_16<- rep(0, length(counties))
Eval_2_16<- rep(0, length(counties))
NWS<- rep(0, length(counties))
Random<- rep(0, length(counties))

for(k in 1:length(counties)){
  county<- counties[k]
  
  folders<- list.files("logs/SB", pattern=paste0(prefix, "_fips-", county))
  Models<- sapply(folders, function(s){strsplit(s, splitvar)[[1]][2]})
  Models<- paste0(splitvar, as.vector(unique(Models)))
  
  proc_NWS_eval<- my_proc(paste0("Summer_results/ORL_NWS_eval_samp-R_obs-W_", r_model, "_fips_", county, ".csv"))
  proc_random_eval<- my_proc(paste0("Summer_results/ORL_random_eval_samp-R_obs-W_", r_model, "_fips_", county, ".csv"))
  
  NWS[k]<- proc_NWS_eval
  Random[k]<- proc_random_eval
  
  for(i in 1:length(Models)){
    model<- Models[i]
    proc_TRPO_train_samp<- my_proc(paste0("Summer_results/ORL_RL_train_samp-R_samp-W_", prefix, "_fips-", county, "_", model, "_fips_", county, ".csv"))
    proc_TRPO_eval<- my_proc(paste0("Summer_results/ORL_RL_eval_samp-R_obs-W_", prefix, "_fips-", county, "_", model, "_fips_", county, ".csv"))
  
    if(substr(model, nchar(model)-3, nchar(model)-3) == "1"){
      if(i == 1){
        Eval_samp_1_16[k]<- proc_TRPO_train_samp
        Eval_1_16[k]<- proc_TRPO_eval
        j1<- 1
      }else{
        if(proc_TRPO_train_samp > Eval_samp_1_16[k]){
          Eval_samp_1_16[k]<- proc_TRPO_train_samp
          Eval_1_16[k]<- proc_TRPO_eval
          j1<- (i+1)/2
        }
      }
    }else{
      if(i == 2){
        Eval_samp_2_16[k]<- proc_TRPO_train_samp
        Eval_2_16[k]<- proc_TRPO_eval
        j2<- 1
      }else{
        if(proc_TRPO_train_samp > Eval_samp_2_16[k]){
          Eval_samp_2_16[k]<- proc_TRPO_train_samp
          Eval_2_16[k]<- proc_TRPO_eval
          j2<- i/2
        }
      }
    }
  }
  opt_HI_thr_1_16[k]<- HI_thresholds[j1]
  opt_HI_thr_2_16[k]<- HI_thresholds[j2]
  print(county) 
}

results<- data.frame(Fips=counties, Random, NWS, 
                     Eval_1_16, opt_HI_thr_1_16, Eval_samp_1_16,
                     Eval_2_16, opt_HI_thr_2_16,  Eval_samp_2_16) # Eval_samp
results[,c("Random", "NWS", 
           "Eval_1_16", "Eval_2_16",
           "Eval_samp_1_16", "Eval_samp_2_16")]<- apply(results[,c("Random", "NWS", 
                                                         "Eval_1_16", "Eval_2_16",
                                                         "Eval_samp_1_16", "Eval_samp_2_16")],
                                                          MARGIN=2, function(x){round(x,3)})
# results

names(results)

best<- t(apply(results, MARGIN=1, function(x){
  first<- x[6] > x[9]
  if(first){
    return(c(x[4], x[5], "NN_1-16"))
  }else{
    return(c(x[7], x[8], "NN_2-16"))
  }
}))

results$Eval<- as.numeric(best[,1])
results$opt_HI_thr<- as.numeric(best[,2])
results$Best_model<- best[,3]

write.csv(results, paste0("Fall_results/Final_eval_30_", prefix, ".csv"))

results[,c("Fips", "Random", "NWS", "Eval", "opt_HI_thr", "Best_model")]

s<- results$Eval - results$NWS

# s<- t(apply(results[,c("Random", "NWS", "Eval", "Eval_1_16", "Eval_2_16")], MARGIN=1, 
#             function(x){
#               a<- (x[2]-x[1])/abs(x[1])
#               b<- (x[3]-x[1])/abs(x[1])
#               c<- (x[4]-x[1])/abs(x[1])
#               d<- (x[5]-x[1])/abs(x[1])
#               return(c(a,b,c,d))
#             }))
#
# colMeans(s)
# 
# t.test(s[,2], s[,1], alternative="g")
wilcox.test(results$Eval, results$NWS, paired = TRUE, alternative = "greater", exact=FALSE)

# ## Old code, when net arch exp was spread across T7 and T8:
# HI_thresholds<- seq(0.5, 0.9, 0.05)
# opt_HI_thr_1_16<- rep(0, length(counties))
# opt_HI_thr_2_16<- rep(0, length(counties))
# Eval_samp_1_16<- rep(0, length(counties))
# Eval_samp_2_16<- rep(0, length(counties))
# Eval_1_16<- rep(0, length(counties))
# Eval_2_16<- rep(0, length(counties))
# NWS<- rep(0, length(counties))
# Random<- rep(0, length(counties))
# 
# splitvar<- "Rstr-HI-"
# 
# for(k in 1:length(counties)){
#   county<- counties[k]
#   
#   folders<- c()
#   for(prefix in c("T7", "T8")){
#     folders<- append(folders, list.files("logs/SB", pattern=paste0(prefix, "_fips-", county)))
#   }
#   p<- which(sapply(folders, function(s){substr(s, nchar(s)-3, nchar(s)) == "2-16"}))
#   folders<- folders[-p]
#   
#   Models<- sapply(folders, function(s){strsplit(s, splitvar)[[1]][2]})
#   Models<- paste0(splitvar, as.vector(unique(Models)))
#   
#   proc_NWS_eval<- my_proc(paste0("Summer_results/ORL_NWS_eval_samp-R_obs-W_", r_model, "_fips_", county, ".csv"))
#   proc_random_eval<- my_proc(paste0("Summer_results/ORL_random_eval_samp-R_obs-W_", r_model, "_fips_", county, ".csv"))
#   
#   NWS[k]<- proc_NWS_eval
#   Random[k]<- proc_random_eval
#   
#   for(i in 1:length(Models)){
#     model<- Models[i]
#     
#     prefix<- "T7"
#     proc_TRPO_train_samp<- my_proc(paste0("Summer_results/ORL_RL_train_samp-R_samp-W_", prefix, "_fips-", county, "_", model, "_fips_", county, ".csv"))
#     proc_TRPO_eval<- my_proc(paste0("Summer_results/ORL_RL_eval_samp-R_obs-W_", prefix, "_fips-", county, "_", model, "_fips_", county, ".csv"))
#     if(i == 1){
#       Eval_samp_1_16[k]<- proc_TRPO_train_samp
#       Eval_1_16[k]<- proc_TRPO_eval
#       j1<- 1
#     }else{
#       if(proc_TRPO_train_samp > Eval_samp_1_16[k]){
#         Eval_samp_1_16[k]<- proc_TRPO_train_samp
#         Eval_1_16[k]<- proc_TRPO_eval
#         j1<- i
#       }
#     }
#     
#     prefix<- "T8"
#     proc_TRPO_train_samp<- my_proc(paste0("Summer_results/ORL_RL_train_samp-R_samp-W_", prefix, "_fips-", county, "_", model, "_fips_", county, ".csv"))
#     proc_TRPO_eval<- my_proc(paste0("Summer_results/ORL_RL_eval_samp-R_obs-W_", prefix, "_fips-", county, "_", model, "_fips_", county, ".csv"))
#     if(i == 1){
#       Eval_samp_2_16[k]<- proc_TRPO_train_samp
#       Eval_2_16[k]<- proc_TRPO_eval
#       j2<- 1
#     }else{
#       if(proc_TRPO_train_samp > Eval_samp_2_16[k]){
#         Eval_samp_2_16[k]<- proc_TRPO_train_samp
#         Eval_2_16[k]<- proc_TRPO_eval
#         j2<- i
#       }
#     }
#   }
#   opt_HI_thr_1_16[k]<- HI_thresholds[j1]
#   opt_HI_thr_2_16[k]<- HI_thresholds[j2]
#   print(county) 
# }
# 
# results<- data.frame(Fips=counties, Random, NWS, 
#                      Eval_1_16, opt_HI_thr_1_16, Eval_samp_1_16,
#                      Eval_2_16, opt_HI_thr_2_16,  Eval_samp_2_16) # Eval_samp
# results[,c("Random", "NWS", 
#            "Eval_1_16", "Eval_2_16",
#            "Eval_samp_1_16", "Eval_samp_2_16")]<- apply(results[,c("Random", "NWS", 
#                                                                    "Eval_1_16", "Eval_2_16",
#                                                                    "Eval_samp_1_16", "Eval_samp_2_16")],
#                                                         MARGIN=2, function(x){round(x,3)})
# # results
# 
# names(results)
# 
# best<- t(apply(results, MARGIN=1, function(x){
#   first<- x[6] > x[9]
#   if(first){
#     return(c(x[4], x[5], "NN_1-16"))
#   }else{
#     return(c(x[7], x[8], "NN_2-16"))
#   }
# }))
# 
# results$Eval<- as.numeric(best[,1])
# results$opt_HI_thr<- as.numeric(best[,2])
# results$Best_model<- best[,3]
# 
# prefix<- "T7-T8"
# write.csv(results, paste0("Fall_results/Final_eval_30_", prefix, ".csv"))
# 
# results[,c("Fips", "Random", "NWS", "Eval", "opt_HI_thr", "Best_model")]
# 
# wilcox.test(results$Eval, results$NWS, paired = TRUE, alternative = "greater", exact=FALSE)

#### Incorporating "forecasts":

# r_model<- "NC_model"
r_model<- "test"
prefix<- "FC1"
FC_type<- "FC-all" # "FC-num_elig", "FC-quantiles", "FC-ten_day", "FC-quarters"

HI_thresholds<- seq(0.5, 0.9, 0.05)
opt_HI_thr<- rep(0, length(counties))
Eval_samp<- rep(0, length(counties))
Eval<- rep(0, length(counties))

for(k in 1:length(counties)){
  county<- counties[k]
  
  # Models<- paste0(r_model, "_Rstr-HI-", HI_thresholds)
  Models<- paste0("_Rstr-HI-", HI_thresholds)
  
  for(i in 1:length(Models)){
    model<- Models[i]
    if(FC_type == "FC-num_elig"){
      proc_train_samp<- my_proc(paste0("Summer_results/ORL_RL_train_samp-R_samp-W_", prefix, "_fips-", county, model, "_fips_", county, ".csv"))
      proc_eval<- my_proc(paste0("Summer_results/ORL_RL_eval_samp-R_obs-W_", prefix, "_fips-", county, model, "_fips_", county, ".csv"))
    }else{
      proc_train_samp<- my_proc(paste0("Summer_results/ORL_RL_train_samp-R_samp-W_", prefix, "_fips-", county, model, "_", FC_type, "_fips_", county, ".csv"))
      proc_eval<- my_proc(paste0("Summer_results/ORL_RL_eval_samp-R_obs-W_", prefix, "_fips-", county, model, "_", FC_type, "_fips_", county, ".csv"))
    }
    
    if(i == 1){
      Eval_samp[k]<- proc_train_samp
      Eval[k]<- proc_eval
      j<- 1
    }else{
      if(proc_train_samp > Eval_samp[k]){
        Eval_samp[k]<- proc_train_samp
        Eval[k]<- proc_eval
        j<- i
      }
    }
  }
  opt_HI_thr[k]<- HI_thresholds[j]
  print(county) 
}

results<- data.frame(Fips=counties, Eval, opt_HI_thr) # Eval_samp
results[,c("Eval")]<- round(results[,c("Eval")],3)
results
write.csv(results, paste0("Fall_results/Final_eval_30_", r_model, "_", FC_type, "-w-rstr-hi.csv"))

earlier<- read.csv("Fall_results/Final_eval_30_T7-T8.csv")
wilcox.test(results$Eval, earlier$NWS, paired = TRUE, alternative = "greater", exact=FALSE)


###########################

## Choosing best size of net_arch based on eval_samp:
old_results<- read.csv("Fall_results/Final_eval_30_best-T7-T8.csv") # "Fall_results/Final_eval_30_T7.csv"
new_results<- read.csv("Fall_results/Final_eval_30_T9.csv") # "Fall_results/Final_eval_30_T8.csv"

Best_Model<- rep("", length(counties))
Best_Eval_Samp<- rep(0, length(counties))
Eval<- rep(0, length(counties))
opt_HI_thr<- rep(0, length(counties))
NWS_samp<- rep(0, length(counties))
Best_Iter<- rep(0, length(counties))

for(k in 1:nrow(old_results)){
  NWS_samp[k]<- my_proc(paste0("Summer_results/ORL_NWS_eval_samp-R_samp-W_test_fips_", old_results[k,"Fips"], ".csv"))
  
  if(old_results[k,"Best_Model"] == "NN_1-16"){
    prefix<- "T7"
  }else{
    prefix<- "T8"
  }
  old<- my_proc(paste0("Summer_results/ORL_RL_eval_samp-R_samp-W_", prefix, "_fips-", old_results[k,"Fips"], "_", "Rstr-HI-", old_results[k,"opt_HI_thr"], "_fips_", old_results[k,"Fips"], ".csv"))
  print(old)
  new<- my_proc(paste0("Summer_results/ORL_RL_eval_samp-R_samp-W_", "T9", "_fips-", new_results[k,"Fips"], "_", "PE_Rstr-HI-", new_results[k,"opt_HI_thr"], "_fips_", new_results[k,"Fips"], ".csv"))
  print(new)
  
  if(new > old){
    Best_Model[k]<- "PE"
    Best_Eval_Samp[k]<- new
    Eval[k]<- new_results[k, "Eval"]
    opt_HI_thr[k]<- new_results[k, "opt_HI_thr"]
    
    npz<- np$load(paste0("logs/SB/", "T9", "_fips-", new_results[k,"Fips"], "_", "PE_Rstr-HI-", new_results[k,"opt_HI_thr"], "/results/evaluations.npz")) 
    iters<- npz$f[["timesteps"]]
    evals<- rowMeans(npz$f[["results"]])
    Best_Iter[k]<- iters[which.max(evals)]
  }else{
    Best_Model[k]<- "No_PE"
    Best_Eval_Samp[k]<- old
    Eval[k]<- old_results[k, "Eval"]
    opt_HI_thr[k]<- old_results[k, "opt_HI_thr"]
    
    npz<- np$load(paste0("logs/SB/", prefix, "_fips-", old_results[k,"Fips"], "_", "Rstr-HI-", old_results[k,"opt_HI_thr"], "/results/evaluations.npz")) 
    iters<- npz$f[["timesteps"]]
    evals<- rowMeans(npz$f[["results"]])
    Best_Iter[k]<- iters[which.max(evals)]
  }
  print(old_results[k,"Fips"])
}
results<- data.frame(old_results[,c("Fips", "Random", "NWS")], 
                     Best_Model, Eval, opt_HI_thr, NWS_samp, Best_Eval_Samp, Best_Iter)
results[,c("Random", "NWS", "Eval", "NWS_samp", "Best_Eval_Samp")]<- apply(results[,c("Random", "NWS", "Eval", "NWS_samp", "Best_Eval_Samp")],
                                             MARGIN=2, function(x){round(x,3)})
results
write.csv(results, "Fall_results/Final_eval_30_best-T7-T8-T9.csv") # "Fall_results/Final_eval_30_best-T7-T8.csv"

x<- (results$Eval - results$Random)/results$Random
y<- (results$NWS - results$Random)/results$Random
t.test(x,y,alternative="g")

### Make table of alert issuance characteristics for the best models:

r_model<- "test"
prefix<- "FC1"
splitvar<- "Rstr-HI-"
aa_qhi<- read.csv(paste0("Fall_results/Final_eval_30_", r_model, "_AA-w-rstr-hi.csv"))

split2<- "FC-"
fc_N<- read.csv(paste0("Fall_results/Final_eval_30_", r_model, "_FC-num_elig-w-rstr-hi.csv"))
fc_Q<- read.csv(paste0("Fall_results/Final_eval_30_", r_model, "_FC-quantiles-w-rstr-hi.csv"))
fc_D10<- read.csv(paste0("Fall_results/Final_eval_30_", r_model, "_FC-ten_day-w-rstr-hi.csv"))
fc_Av4<- read.csv(paste0("Fall_results/Final_eval_30_", r_model, "_FC-quarters-w-rstr-hi.csv"))
fc_All<- read.csv(paste0("Fall_results/Final_eval_30_", r_model, "_FC-all-w-rstr-hi.csv"))

alerts_results<- data.frame(matrix(ncol = 10, nrow = 0))

for(k in 1:length(counties)){
  county<- counties[k]
  
  folders<- list.files("logs/SB", pattern=paste0(prefix, "_fips-", county))
  Models<- sapply(folders, function(s){strsplit(s, splitvar)[[1]][2]})
  Models<- paste0(splitvar, as.vector(unique(Models)))
  types<- unique(sapply(Models, function(s){strsplit(s, split2)[[1]][2]}))
  types[1]<- "num_elig"
  
  as_NWS_eval<- assess(paste0("Summer_results/ORL_NWS_eval_samp-R_obs-W_", r_model, "_fips_", county, ".csv"))
  # as_random_eval<- assess(paste0("Summer_results/ORL_random_eval_samp-R_obs-W_", r_model, "_fips_", county, ".csv"))
  as_AQHI_eval<- assess(paste0("Summer_results/ORL_AA_eval_samp-R_obs-W_", r_model, "_Rstr-HI-",  aa_qhi[k,"opt_HI_thr"], "_fips_", county, ".csv"))
  
  # as_random_eval$Policy<- "random"
  as_NWS_eval$Policy<- "NWS"
  as_AQHI_eval$Policy<- "Aqhi"
  
  as_df<- rbind( #as_random_eval, 
                as_NWS_eval,
                as_AQHI_eval)
  
  for(m in types){
    oht<- read.csv(paste0("Fall_results/Final_eval_30_", r_model, "_FC-", m, "-w-rstr-hi.csv"))$opt_HI_thr[k]
    if(oht %% 0.1 == 0.1){
      y<- assess(paste0("Summer_results/ORL_RL_eval_samp-R_obs-W_", prefix, "_fips-", county, "_", splitvar, round(oht,1), "_fips_", county, ".csv"))
    }else{
      y<- assess(paste0("Summer_results/ORL_RL_eval_samp-R_obs-W_", prefix, "_fips-", county, "_", splitvar, oht, "_fips_", county, ".csv"))
    }
    y$Policy<- paste0("FC-", m)
    as_df<- rbind(as_df, y)
  }
  
  alerts_results<- rbind(alerts_results, as_df)
  print(county) 
}

# alerts_results$Fips<- rep(counties, each=3)
alerts_results$Fips<- rep(counties, each=7)

alerts_results[1:85,c(12, 11, 1:10)]
alerts_results[85:161,c(12, 11, 1:10)]
alerts_results[162:nrow(alerts_results),c(12, 11, 1:10)]

### Making an expanded table to compare across experiments:
earlier<- read.csv("Fall_results/Final_eval_30_T7.csv")
earlier<- earlier[,c(2:4,6,5)]

# P_0<- rep(0, length(counties))
# Obs_W<- rep(0, length(counties))
NN_2l<- rep(0, length(counties))
Saved_Iter<- rep(0, length(counties))

for(k in 1:length(counties)){
  county<- counties[k]
  
  # P_0[k]<- my_proc(paste0("Summer_results/ORL_RL_eval_samp-R_obs-W_", "E1", "_fips-", county, "_P-0_Rstr-HI-opt", "_fips_", county, ".csv"))
  # Obs_W[k]<- my_proc(paste0("Summer_results/ORL_RL_eval_samp-R_obs-W_", "E1", "_fips-", county, "_obs-W_Rstr-HI-opt", "_fips_", county, ".csv"))
  NN_2l[k]<- my_proc(paste0("Summer_results/ORL_RL_eval_samp-R_obs-W_", "T7", "_fips-", county, "_Rstr-HI-opt_", "2-16", "_fips_", county, ".csv"))
  
  trpo_npz<- np$load(paste0("logs/SB/", "T7", "_fips-", county, "_Rstr-HI-", earlier$opt_HI_thr[k], "/results/evaluations.npz")) 
  trpo_iters<- trpo_npz$f[["timesteps"]]
  trpo_evals<- rowMeans(trpo_npz$f[["results"]])
  Saved_Iter[k]<- trpo_iters[which.max(trpo_evals)]
  
  print(county)
}

# results<- data.frame(earlier, Saved_Iter, P_0, Obs_W)
results<- data.frame(earlier, Saved_Iter, NN_2l)
results[,-1]<- apply(results[,-1], MARGIN=2, function(x){round(x,3)})
results



###### Baseline comparisons:
counties<- c(41067, 53015, 20161, 37085, 48157,
             28049, 19153, 17167, 31153, 6071, 4013,
             34021, 19155, 17115, 29021, 29019, 5045, 40017, 21059,
             47113, 42017, 22109, 45015, 13031, 48367, 22063, 41053,
             32003, 4015, 6025)

## Top k QHI days:
Eval_samp<- rep(0, length(counties))
Eval<- rep(0, length(counties))

for(k in 1:length(counties)){
  county<- counties[k]
  
  Eval_samp[k]<- my_proc(paste0("Summer_results/ORL_TK_eval_samp-R_samp-W_test_fips_", county, ".csv"))
  Eval[k]<- my_proc(paste0("Summer_results/ORL_TK_eval_samp-R_obs-W_test_fips_", county, ".csv"))
  
  print(county)
}

results<- data.frame(Fips=counties, Eval) # Eval_samp
results[,c("Eval")]<- round(results[,c("Eval")],3)
results
write.csv(results, paste0("Fall_results/Final_eval_30_TK.csv"))

## AA or random policy with HI restriction:

# r_model<- "NC_model"
r_model<- "test"
pol<- "AA"
# pol<- "random"

HI_thresholds<- seq(0.5, 0.9, 0.05)
opt_HI_thr<- rep(0, length(counties))
Eval_samp<- rep(0, length(counties))
Eval<- rep(0, length(counties))

for(k in 1:length(counties)){
  county<- counties[k]
  
  Models<- paste0(r_model, "_Rstr-HI-", HI_thresholds)
  
  for(i in 1:length(Models)){
    model<- Models[i]
    proc_random_train_samp<- my_proc(paste0("Summer_results/ORL_", pol, "_train_samp-R_samp-W_", model, "_fips_", county, ".csv"))
    proc_random_eval<- my_proc(paste0("Summer_results/ORL_", pol, "_eval_samp-R_obs-W_", model, "_fips_", county, ".csv"))
    
    if(i == 1){
      Eval_samp[k]<- proc_random_train_samp
      Eval[k]<- proc_random_eval
      j<- 1
    }else{
      if(proc_random_train_samp > Eval_samp[k]){
        Eval_samp[k]<- proc_random_train_samp
        Eval[k]<- proc_random_eval
        j<- i
      }
    }
  }
  opt_HI_thr[k]<- HI_thresholds[j]
  print(county) 
}

results<- data.frame(Fips=counties, Eval, opt_HI_thr) # Eval_samp
results[,c("Eval")]<- round(results[,c("Eval")],3)
results
write.csv(results, paste0("Fall_results/Final_eval_30_", r_model, "_", pol, "-w-rstr-hi.csv"))


earlier<- read.csv("Fall_results/Final_eval_30_best-T7-T8.csv")
earlier$Random_QHI<- results$Eval
earlier$R_QHI_opt<- results$opt_HI_thr

x<- (earlier$Eval - earlier$Random)/ abs(earlier$Random)
y<- (earlier$Random_QHI - earlier$Random)/ abs(earlier$Random)
z<- (earlier$NWS - earlier$Random)/ abs(earlier$Random)

t.test(x,y,alternative="two.sided")
t.test(x,z,alternative="g")
t.test(y,z,alternative="g")

## No HI restriction:

r_model<- "test" # "NC_model"
prefix<- "T0" # "NC0" 
splitvar<-"_"  # "P-"

# Eval<- data.frame(matrix(ncol = 2, nrow = length(counties)))
Eval<- data.frame(matrix(ncol = 4, nrow = length(counties)))
NWS<- rep(0, length(counties))
Random<- rep(0, length(counties))

for(k in 1:length(counties)){
  county<- counties[k]
  
  folders<- list.files("logs/SB", pattern=paste0(prefix, "_fips-", county))
  # Models<- sapply(folders, function(s){strsplit(s, splitvar)[[1]][2]})
  # Models<- sapply(folders, function(s){strsplit(s, splitvar)[[1]][3]})
  Models<- sapply(folders, function(s){
    a<- strsplit(s, splitvar)[[1]][3]
    b<- strsplit(s, splitvar)[[1]][4]
    return(paste(a,b,sep=splitvar))
  })
  # Models<- paste0(splitvar, as.vector(unique(Models)))
  Models<- as.vector(unique(Models))
  
  proc_NWS_eval<- my_proc(paste0("Summer_results/ORL_NWS_eval_samp-R_obs-W_", r_model, "_fips_", county, ".csv"))
  proc_random_eval<- my_proc(paste0("Summer_results/ORL_random_eval_samp-R_obs-W_", r_model, "_fips_", county, ".csv"))
  
  NWS[k]<- proc_NWS_eval
  Random[k]<- proc_random_eval
  
  for(i in 1:length(Models)){
    model<- Models[i]
    Eval[k,i]<- my_proc(paste0("Summer_results/ORL_RL_eval_samp-R_obs-W_", prefix, "_fips-", county, "_", model, "_fips_", county, ".csv"))
  }
  print(county)
}

results<- data.frame(Fips=counties, Random, NWS, Eval)
# names(results)[4:5]<- c("Eval_P-0", "Eval_P-0.01")
# names(results)[4:5]<- c("Eval_NN-1-16", "Eval_NN-2-16")
names(results)[4:7]<- c("DQN_NN-1-16", "DQN_NN-2-16",
                        "TRPO_NN-1-16", "TRPO_NN-2-16")
results[,-1]<- apply(results[,-1], MARGIN=2, function(x){round(x,3)})
results

best_dqn<- t(apply(results, MARGIN=1, function(x){
  first<- x[4] > x[5]
  if(first){
    return(c(x[4], "NN_1-16"))
  }else{
    return(c(x[5], "NN_2-16"))
  }
}))
best_trpo<- t(apply(results, MARGIN=1, function(x){
  first<- x[6] > x[7]
  if(first){
    return(c(x[6], "NN_1-16"))
  }else{
    return(c(x[7], "NN_2-16"))
  }
}))

results$Eval_DQN<- as.numeric(best_dqn[,1])
results$Eval_TRPO<- as.numeric(best_trpo[,1])
results$DQN_best<- best_dqn[,2]
results$TRPO_best<- best_trpo[,2]

write.csv(results, paste0("Fall_results/Final_eval_30_", prefix, ".csv"))

s<- t(apply(results[,c("Random", "Eval_NN-1-16", "Eval_NN-2-16")], MARGIN=1, 
          function(x){
            a<- (x[2]-x[1])/abs(x[1])
            b<- (x[3]-x[1])/abs(x[1])
            return(c(a,b))
          }))

colMeans(s)


alerts_results<- data.frame(matrix(ncol = 10, nrow = 0))

for(k in 1:length(counties)){
  county<- counties[k]
  
  folders<- list.files("logs/SB", pattern=paste0(prefix, "_fips-", county))
  Models<- sapply(folders, function(s){strsplit(s, splitvar)[[1]][2]})
  Models<- paste0(splitvar, as.vector(unique(Models)))
  
  as_NWS_eval<- assess(paste0("Summer_results/ORL_NWS_eval_samp-R_obs-W_test_fips_", county, ".csv"))
  as_random_eval<- assess(paste0("Summer_results/ORL_random_eval_samp-R_obs-W_test_fips_", county, ".csv"))
  
  as_random_eval$Policy<- "random"
  as_NWS_eval$Policy<- "NWS"
  
  as_df<- rbind(as_random_eval, as_NWS_eval)
  
  for(m in Models){
    rl<- assess(paste0("Summer_results/ORL_RL_eval_samp-R_obs-W_", prefix, "_fips-", county, "_", m, "_fips_", county, ".csv"))
    rl$Policy<- "TRPO"
    as_df<- rbind(as_df, rl)
  }
  
  alerts_results<- rbind(alerts_results, as_df)
  print(county) 
}

alerts_results$Fips<- rep(counties, each=4)
alerts_results$Penalty<- rep(c(NA, NA, 0, 0.01), length(counties))
alerts_results[,c(12, 11, 13, 1:10)]


