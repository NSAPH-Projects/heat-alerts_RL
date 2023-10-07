
library(dplyr)
library(arrow)
library(rjson)
library(stringr)
library(cdlTools)
library(readr)
library(xtable)

## Get background info:

data<- read_csv("data/Summer23_Train_smaller-for-Python.csv")

# bayes<- read.csv("heat_alerts/bayesian_model/results/Bayesian_FullFast_8-16.csv", header=FALSE)
bayes<- read.csv("heat_alerts/bayesian_model/results/Bayesian_FF_C-M_wide-EB-prior.csv", header=FALSE)
A<- read_parquet("data/processed/actions.parquet")$alert

# Set train_years to zero so they're not included in the sum:
n_days<- 153
val_years<- c(2, 6, 10) # corresponding to 2007, 2011, 2015
train_years<- c(1:11)[-val_years]
n_counties<- length(A)/n_days/11
for(y in train_years){
  j<- n_days*(y-1)
  A[rep((j+1):(j+n_days),n_counties) + rep(seq(0, length(A), n_days*11)[1:n_counties], each=n_days)]<- 0
}

locs<- read_parquet("data/processed/location_indicator.parquet")[,1]$sind

crosswalk<- unlist(fromJSON(file="data/processed/fips2idx.json"))
fips<- names(crosswalk)

sum_alerts<- aggregate(A ~ locs, data.frame(A,locs), sum)
sd_eff<- aggregate(Eff ~ locs, data.frame(Eff=bayes$V1,locs), sd)

W<- as.data.frame(read_parquet("data/processed/spatial_feats.parquet"))
region_vars<- c("Cold", "Hot-Dry", "Marine", "Mixed-Dry", "Mixed-Humid", "Very Cold")
Region<- apply(W[,region_vars],
               MARGIN=1, function(x) region_vars[which(x == 1)])
Region[lengths(Region)==0]<- "Hot-Humid"
Region<- unlist(Region)

DF<- data.frame(County = fips[locs+1],
                Region = Region[locs+1],
                Alerts = rep(sum_alerts$A, each=length(A)/761),
                SD_Eff = rep(sd_eff$Eff, each=length(A)/761) #, 
                # Pop_density = exp(W$Log_Pop_density)[locs+1],
                # Med_HH_Inc = exp(W$Log_Med_HH_Income)[locs+1],
                # Democrat_pct = W$Democrat[locs+1],
                # Broadband_use = W$broadband_usage[locs+1],
                # PM25 = W$pm25[locs+1]
                )

counties<- str_pad(c(41067, 53015, 20161, 37085, 48157,
                     28049, 19153, 17167, 31153, 6071, 4013,
                     34021, 19155, 17115, 29021, 29019, 5045, 40017, 21059,
                     47113, 42017, 22109, 45015, 13031, 48367, 22063, 41053,
                     32003, 4015, 6025), 5, pad="0")

States<- data.frame(Fips = counties,
                    State_fips = substr(counties, 1,2))
States$State<- fips(States$State_fips, to="Abbreviation")
States$Fips<- as.numeric(States$Fips)

DF$Fips<- as.numeric(DF$County)
DF$County<- NULL
DF<- inner_join(distinct(DF), States[,c("Fips", "State")])

## Get spatial covariates on their original scales:
more_W<- aggregate(. ~ fips, data[which(data$fips %in% counties),c("fips", "Pop_density", "Med.HH.Income",  "Democrat", "broadband.usage", "pm25")], mean)
more_W$Fips<- as.numeric(more_W$fips)
more_W$fips<- NULL

DF<- inner_join(DF, more_W)

write.csv(DF, "data/Final_30_W.csv")

## Select one county from each climate region for hyperparameter tuning:

regions<- unique(DF$Region)

for(r in 1:length(regions)){
  print(regions[r])
  pos<- which(DF$Region==regions[r])
  Z_vars<- apply(DF[pos, c("Alerts", "SD_Eff", "Pop_density", "Med.HH.Income", 
                                              "Democrat", "broadband.usage", "pm25")], MARGIN=2, scale)
  i<- which.min(rowSums(abs(Z_vars)))
  print(paste0(DF$Fips[pos[i]], " (", DF$State[pos[i]], ")"))
}

# ## Add evaluation results:
# 
# results<- read.csv("Fall_results/Final_eval_30_best-T7-T8.csv")
# Results<- results[,c("Fips", "Random", "NWS", "Eval", "opt_HI_thr", "Best_Iter")]
# names(Results)[ncol(Results)-1]<- "QHI_thr"
# Results$Best_Iter<- Results$Best_Iter/1000000
# 
# Full<- inner_join(DF[,c("Fips", "Region", "State", "Alerts", "SD_Eff")], Results)
# 
# ## Add absolute HI for context:
# 
# data<- read_csv("data/Summer23_Train_smaller-for-Python.csv")
# data$fips<- as.numeric(data$fips)
# 
# HI_thr<- rep(0, nrow(Full))
# for(k in 1:nrow(Full)){
#   d<- data[which(data$fips == Full$Fips[k]),]
#   HI_thr[k]<- quantile(d$HImaxF_PopW, Full$QHI_thr[k])
# }
# 
# Full$HI_thr<- HI_thr
# 

Full<- DF

## Print for Latex:

Final<- Full[order(Full$Region, decreasing=TRUE),]
Final$Fips<- as.character(Final$Fips)
Final$Alerts<- as.integer(Final$Alerts)
# Final$HI_thr<- as.integer(Final$HI_thr)
# Final$QHI_thr<- as.integer(Final$QHI_thr*100)
# names(Final)[which(names(Final)=="Eval")]<- "TRPO"
# r<- Final$Random
# Final$Std_diff<- (Final$TRPO - Final$NWS)/abs(r)

# To save space:
orig<- c("Mixed-Humid", "Marine", "Hot-Humid", "Hot-Dry", "Cold")
ab<- c("MxHd", "Mrn", "HtHd", "HtDr", "Cold")
Final$Region<- as.vector(sapply(Final$Region, function(x){ab[match(x,orig)]}))
Final$Fips_ST<- apply(Final[,c("Fips", "State")], MARGIN=1, function(x){
  paste0(paste(x[1], x[2], sep=" ("),")")
})

# print(xtable(Final[,c("Fips_ST","Region","SD_Eff", "Alerts","Random","NWS",
#                       "TRPO", "Std_diff", "QHI_thr", "HI_thr", "Best_Iter")], 
#              digits=3, hline.after = 1:nrow(Final)), include.rownames=FALSE)

Final$SD_Eff<- round(Final$SD_Eff,3)
Final$Pop_density<- round(Final$Pop_density)
Final$Med.HH.Income<- round(Final$Med.HH.Income)
Final$Democrat<- round(100*Final$Democrat,1)
Final$broadband.usage<- round(100*Final$broadband.usage,1)
Final$pm25<- round(Final$pm25,2)

print(xtable(Final[,c("Fips_ST", "Region", "Alerts", # "SD_Eff", 
                      "Pop_density", "Med.HH.Income", "Democrat",
                      "broadband.usage", "pm25")], 
             # digits=3, 
             hline.after = 1:nrow(Final)), include.rownames=FALSE)

############################# Make benchmark comparisons tables:

n_days<- 153

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
    if(is.na(avg_reward_A.1)){
      estimated_reward<- (n_days-1)*avg_reward_A.0
    }else{
      estimated_reward<- b*avg_reward_A.1 + (n_days-1-b)*avg_reward_A.0
    }
    # return(list(agg_df, estimated_reward))
    return(estimated_reward)
  }else{
    return(NA)
  }
}

counties<- c(41067, 53015, 20161, 37085, 48157, 
             28049, 19153, 17167, 31153, 6071, 4013,
             34021, 19155, 17115, 29021, 29019, 5045, 40017, 21059,
             47113, 42017, 22109, 45015, 13031, 48367, 22063, 41053, 
             32003, 4015, 6025)

# r_model<- "NC_model"
r_model<- "test"

Zero<- rep(0,length(counties))
NWS<- rep(0,length(counties))
Random<- rep(0,length(counties))
Top_K<- rep(0,length(counties))

for(k in 1:length(counties)){
  county<- counties[k]
  
  Zero[k]<- my_proc(paste0("Summer_results/ORL_NA_eval_samp-R_obs-W_", r_model, "_fips_", county, ".csv"))
  # NWS[k]<- my_proc(paste0("Summer_results/ORL_NWS_eval_samp-R_obs-W_", r_model, "_fips_", county, ".csv"))
  # Random[k]<- my_proc(paste0("Summer_results/ORL_random_eval_samp-R_obs-W_", r_model, "_fips_", county, ".csv"))
  # Top_K[k]<- my_proc(paste0("Summer_results/ORL_TK_eval_samp-R_obs-W_", r_model, "_fips_", county, ".csv"))
  # 
  print(county)
}

wilcox.test(Zero, NWS, paired = TRUE, alternative = "greater", exact=FALSE)

results<- round(data.frame(County=counties, Zero, NWS, Random, Top_K),3)

Random_QHI<- read.csv(paste0("Fall_results/Alert-rate_Final_eval_30_", r_model, "_random-w-rstr-hi.csv"))
AA_QHI<- read.csv(paste0("Fall_results/Alert-rate_Final_eval_30_", r_model, "_AA-w-rstr-hi.csv"))

results$Random_QHI<- round(Random_QHI$Eval,3)
results$RQ_thr<- Random_QHI$opt_HI_thr

results$AA_QHI<- round(AA_QHI$Eval,3)
results$AA_thr<- AA_QHI$opt_HI_thr

if(r_model == "test"){
  RL<- read.csv("Fall_results/Alert-rate_Final_eval_30_T7-T8.csv")
  rl0<- read.csv("Fall_results/Alert-rate_Final_eval_30_T7-T8.csv")
}else if(r_model == "NC_model"){
  RL<- read.csv("Fall_results/Alert-rate_Final_eval_30_NC1.csv")
}

results$RL<- RL$Eval
results$RL_thr<- RL$opt_HI_thr
results

write.csv(results, paste0("Fall_results/Alert-rate_All_evals_", r_model, ".csv"))

# Benchmark<- c("Zero", "NWS", "Top_K", "Random_QHI", "AA_QHI", "RL")
Benchmark<- c("Zero", "Random", "Top_K", "Random_QHI", "AA_QHI", "RL")
Mean<- c()
# SD<- c()

for(y in Benchmark[-1]){
  # Mean<- append(Mean, mean((results[,y] - results$Random)/abs(results$Random)))
  # Mean<- append(Mean, mean((results[,y] - results$NWS)/abs(results$NWS)))
  # hist(results[,y] - results$NWS, main=paste(y, "plain"))
  # hist((results[,y] - results$NWS)/abs(results$NWS), main=paste(y, "standardized"))
  # Mean<- append(Mean, mean((results[,y] - results$NWS)))
  # SD<- append(SD, sd((results[,y] - results$NWS)))
  print(y)
  # print(wilcox.test(results[,y], results$NWS, paired = TRUE, alternative = "greater"))
  print(wilcox.test(results[,y], results$NWS, paired = TRUE, alternative = "greater", exact=FALSE))
}

d<- data.frame(Benchmark, Mean=round(Mean,3) #, SD=round(SD,3)
)
d

# t.test(results$RL - results$NWS, alternative="g")

print(xtable(d, digits=3),include.rownames=FALSE)

