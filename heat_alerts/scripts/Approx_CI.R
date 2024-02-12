
agg_results<- read.csv("Fall_results/December_part-2_Rstr-QHI_RL_avg_return.csv")

Agg_results<- agg_results[which(agg_results$Algo=="a2c" & agg_results$Forecast=="none"),]

## Get and scramble episode rewards:
rl_df<- matrix(0, nrow=1000, ncol=30)
nws_df<- matrix(0, nrow=1000, ncol=30)

for(i in 1:30){
  county<- Agg_results$County[i]
  
  dat<- read.csv(paste0("Summer_results/ORL_RL_eval_samp-R_obs-W_December_",
                        "a2c", "_F-", "none", "_Rstr-HI-", Agg_results$OT[i],
                        "_arch-", Agg_results$NHL[i], "-", Agg_results$NHU[i], "_ns-", Agg_results$n_steps[i],
                        "_fips-", county, "_fips_", county, ".csv"))
  nws<- read.csv(paste0("Summer_results/ORL_NWS_eval_samp-R_obs-W_", "mixed_constraints", "_fips_", county, ".csv"))
  
  dat$Index<- rep(1:1000, each=152)
  nws$Index<- rep(1:1000, each=152)
  
  rl_df[,i]<- tapply(dat$Rewards, dat$Index, sum)
  nws_df[,i]<- tapply(nws$Rewards, dat$Index, sum)

  print(i)
}

## Convert to absolute hosps:
W<- read.csv("data/Final_30_W.csv")

W<- W[match(Agg_results$County, W$Fips),]

get_hosps<- function(x){ # where x is a vector from avg_return function
  y<- (1 - x/152)*W$Offset/W$total_count
  return(y*152) # per summer
}

rl_DF<- apply(rl_df, MARGIN=1, get_hosps)*10000
nws_DF<- apply(nws_df, MARGIN=1, get_hosps)*10000

DF<- nws_DF - rl_DF
# DF<- apply(DF, MARGIN=1, function(x){sample(x)})

## Calculate medians --> approximate CI

m<- apply(DF, MARGIN=1, median)
quantile(m, probs = c(0.025, 0.975))
quantile(m, probs = c(0.025, 0.975))*4900

