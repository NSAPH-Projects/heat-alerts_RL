library(dplyr)

library(reticulate)
np<- import("numpy")

## Define functions:
n_days<- 153

counties<- c(41067, 53015, 20161, 37085, 48157,
             28049, 19153, 17167, 31153, 6071, 4013,
             34021, 19155, 17115, 29021, 29019, 5045, 40017, 21059,
             47113, 42017, 22109, 45015, 13031, 48367, 22063, 41053,
             32003, 4015, 6025)

my_proc<- function(filename){
  f<- file.exists(filename) # filename<- "Summer_results/ORL_RL_eval_samp-R_obs-W_T7_fips-6025_Rstr-HI-0.9_fips_6025.csv"
  if(f){
    df<- read.csv(filename)[,-1]
    df$Count = 1
    # df$Budget<- df$Budget/(n_days-1)
    # agg_df<- aggregate(. ~ Year, df, sum)
    df$Alert<- df$Actions
    agg_df<- aggregate(. ~ Year + Alert, df, sum)
    # agg_df$Budget<- agg_df$Budget/(n_days-1)
    # agg_df$budget_frac<- agg_df$Actions/agg_df$Budget
    agg_df$Frac<- agg_df$Count/sum(agg_df$Count)
    # estimated_reward<- sum(agg_df$Rewards*(1/nrow(agg_df))/agg_df$Frac)/1000
    avg_reward_A.0<- mean(agg_df[agg_df$Alert == 0, "Rewards"]/agg_df[agg_df$Alert == 0, "Count"])
    avg_reward_A.1<- mean(agg_df[agg_df$Alert == 1, "Rewards"]/agg_df[agg_df$Alert == 1, "Count"])
    b<- mean(df$Budget)
    estimated_reward<- b*avg_reward_A.1 + (n_days-1-b)*avg_reward_A.0
    # return(list(agg_df, estimated_reward))
    return(estimated_reward)
  }else{
    return(NA)
  }
}

compare_to_zero<- function(filename, Zero){
  
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

