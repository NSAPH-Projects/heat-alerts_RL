
library(dplyr)
library(arrow)
library(rjson)
library(stringr)
library(cdlTools)
library(readr)
library(xtable)

############################# Make table of background info for each county:

data<- read_csv("data/Summer23_Train_smaller-for-Python.csv")

bayes<- read.csv("heat_alerts/bayesian_model/results/Bayesian_FF_C-M_wide-EB-prior.csv", header=FALSE)
A<- read_parquet("data/processed/actions.parquet")$alert
offset<- read_parquet("data/processed/offset.parquet")$mean_other_hosps

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
mean_eff<- aggregate(Eff ~ locs, data.frame(Eff=bayes$V1,locs), mean)

W<- as.data.frame(read_parquet("data/processed/spatial_feats.parquet"))
region_vars<- c("Cold", "Hot-Dry", "Marine", "Mixed-Dry", "Mixed-Humid", "Very Cold")
Region<- apply(W[,region_vars],
               MARGIN=1, function(x) region_vars[which(x == 1)])
Region[lengths(Region)==0]<- "Hot-Humid"
Region<- unlist(Region)

DF<- data.frame(County = fips[locs+1],
                Region = Region[locs+1],
                Offset = offset,
                Alerts = rep(sum_alerts$A, each=length(A)/761),
                Mean_Eff = rep(mean_eff$Eff, each=length(A)/761),
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
DF<- aggregate(Offset ~ ., DF, mean)

## Get spatial covariates on their original scales:
more_W<- aggregate(. ~ fips, data[which(data$fips %in% counties),c("fips", "Pop_density", "Med.HH.Income",  "Democrat", "broadband.usage", "pm25")], mean)
more_W$Fips<- as.numeric(more_W$fips)
more_W$fips<- NULL

DF<- inner_join(DF, more_W)

## Get autocorrelation summaries:
data<- read_parquet("data/processed/states.parquet")
QHI<- data$quant_HI_county
Year<- year(data$Date)

acf_auc_1d<- rep(0, 30)
acf_auc_3d<- rep(0, 30)
acf_auc_5d<- rep(0, 30)
acf_auc_7d<- rep(0, 30)

i<- 1
for(k in DF$Fips){
  stats<- rowMeans(sapply(c(2007, 2011, 2015), function(y){
    qhi<- QHI[which(data$fips == k & Year == y)]
    autocorr<- acf(qhi, plot=FALSE)
    return(c(autocorr$acf[2], sum(autocorr$acf[2:4]), 
             sum(autocorr$acf[2:6]), sum(autocorr$acf[2:8])))
  }))

  acf_auc_1d[i]<- stats[1]
  acf_auc_3d[i]<- stats[2]
  acf_auc_5d[i]<- stats[3]
  acf_auc_7d[i]<- stats[4]
  
  i<- i+1
}

DF<- cbind(DF, acf_auc_1d, acf_auc_3d, acf_auc_5d, acf_auc_7d)

write.csv(DF, "data/Final_30_W.csv")

##### Select one county from each climate region for *preliminary* hyperparameter tuning, using the info above:

regions<- unique(DF$Region)

for(r in 1:length(regions)){
  print(regions[r])
  pos<- which(DF$Region==regions[r])
  Z_vars<- apply(DF[pos, c("Alerts", "SD_Eff", "Pop_density", "Med.HH.Income", 
                                              "Democrat", "broadband.usage", "pm25")], MARGIN=2, scale)
  i<- which.min(rowSums(abs(Z_vars)))
  print(paste0(DF$Fips[pos[i]], " (", DF$State[pos[i]], ")"))
}

## Print for Latex:
Full<- DF
Final<- Full[order(Full$Region, decreasing=TRUE),]
Final$Fips<- as.character(Final$Fips)
Final$Alerts<- as.integer(Final$Alerts)

# To save space:
orig<- c("Mixed-Humid", "Marine", "Hot-Humid", "Hot-Dry", "Cold")
ab<- c("MxHd", "Mrn", "HtHd", "HtDr", "Cold")
Final$Region<- as.vector(sapply(Final$Region, function(x){ab[match(x,orig)]}))
Final$Fips_ST<- apply(Final[,c("Fips", "State")], MARGIN=1, function(x){
  paste0(paste(x[1], x[2], sep=" ("),")")
})

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

############################# Make main results table:

WMW<- function(x, y=bench_df$NWS){
  wmw<- wilcox.test(x, y, paired = TRUE, alternative = "greater", exact=FALSE)
  metrics<- as.vector(c(round(median(x - y, na.rm=TRUE),3), 
                        wmw$statistic, round(wmw$p.value,5)))
  return(metrics)
}

source("heat_alerts/scripts/Convert_to_hosps.R")

bench_df<- read.csv(paste0("Fall_results/Benchmarks_mixed_constraints_avg_return.csv"))
plain_RL<- read.csv("Fall_results/December_plain_RL_avg_return.csv")
QHI_RL<- read.csv("Fall_results/December_Rstr-QHI_RL_avg_return.csv")
plain_RL_p2<- read.csv("Fall_results/December_part-2_plain_RL_avg_return.csv")
QHI_RL_p2<- read.csv("Fall_results/December_part-2_Rstr-QHI_RL_avg_return.csv")
det_trpo_none<- read.csv("Fall_results/NEW_Main_analysis_trpo_F-none.csv")
det_trpo_F<- read.csv("Fall_results/NEW_Main_analysis_trpo_F-Q_D10.csv")
det_newer<- read.csv("Fall_results/February_Rstr-QHI_RL_avg_return.csv")

alt_policies<- cbind(bench_df[,c("Random", "basic_NWS", 
                                 "Top_K", "Random_QHI", "AA_QHI")],
                     dqn=plain_RL[which(plain_RL$Algo == "dqn" & plain_RL$Forecast == "none"), "Eval"],
                     qrdqn=plain_RL_p2[which(plain_RL_p2$Algo == "qrdqn" & plain_RL_p2$Forecast == "none"), "Eval"],
                     trpo=plain_RL[which(plain_RL$Algo == "trpo" & plain_RL$Forecast == "none"), "Eval"],
                     a2c=plain_RL_p2[which(plain_RL_p2$Algo == "a2c" & plain_RL_p2$Forecast == "none"), "Eval"],
                     dqn.f=plain_RL[which(plain_RL$Algo == "dqn" & plain_RL$Forecast == "Q_D10"), "Eval"],
                     qrdqn.f=plain_RL_p2[which(plain_RL_p2$Algo == "qrdqn" & plain_RL_p2$Forecast == "Q_D10"), "Eval"],
                     trpo.f=plain_RL[which(plain_RL$Algo == "trpo" & plain_RL$Forecast == "Q_D10"), "Eval"],
                     a2c.f=plain_RL_p2[which(plain_RL_p2$Algo == "a2c" & plain_RL_p2$Forecast == "Q_D10"), "Eval"],
                     dqn.qhi=QHI_RL[which(QHI_RL$Algo == "dqn" & QHI_RL$Forecast == "none"), "Eval"],
                     qrdqn.qhi=QHI_RL_p2[which(QHI_RL_p2$Algo == "qrdqn" & QHI_RL_p2$Forecast == "none"), "Eval"],
                     trpo.qhi=QHI_RL[which(QHI_RL$Algo == "trpo" & QHI_RL$Forecast == "none"), "Eval"],
                     a2c.qhi=QHI_RL_p2[which(QHI_RL_p2$Algo == "a2c" & QHI_RL_p2$Forecast == "none"), "Eval"],
                     dqn.qhi.f=QHI_RL[which(QHI_RL$Algo == "dqn" & QHI_RL$Forecast == "Q_D10"), "Eval"],
                     qrdqn.qhi.f=QHI_RL_p2[which(QHI_RL_p2$Algo == "qrdqn" & QHI_RL_p2$Forecast == "Q_D10"), "Eval"],
                     trpo.qhi.f=QHI_RL[which(QHI_RL$Algo == "trpo" & QHI_RL$Forecast == "Q_D10"), "Eval"],
                     a2c.qhi.f=QHI_RL_p2[which(QHI_RL_p2$Algo == "a2c" & QHI_RL_p2$Forecast == "Q_D10"), "Eval"],
                     det.trpo.qhi=det_trpo_none$Eval,
                     det.a2c.qhi=det_newer[which(det_newer$Algo == "a2c" & det_newer$Forecast == "none"), "Eval"],
                     det.trpo.qhi.f=det_trpo_F$Eval,
                     det.a2c.qhi.f=det_newer[which(det_newer$Algo == "a2c" & det_newer$Forecast == "Q_D10"), "Eval"]
)

D<- t(apply(alt_policies, MARGIN=2, WMW))
D<- as.data.frame(D)
names(D)<- c("Median Diff.", "WMW stat", "p-value")

pos<- match(W$Fips, bench_df$County)
Pols<- cbind(Zero=bench_df$Zero, NWS=bench_df$NWS, alt_policies)[pos,]
nohr<- apply(Pols, MARGIN=2, get_hosps)

denom<- 10000
nohr<- denom*nohr
compared_to_zero<- apply(nohr[,3:ncol(nohr)], MARGIN=2, function(y){nohr[,1]-y})
compared_to_nws<- apply(nohr[,3:ncol(nohr)], MARGIN=2, function(y){nohr[,2]-y})

vs.nws<- apply(compared_to_nws, MARGIN=2, median)
vs.zero<- apply(compared_to_zero, MARGIN=2, median)
D[,"Median Annual Hosps Saved vs NWS (vs Zero) / 10,000"]<- paste0(round(vs.nws,3), " (", round(vs.zero,2), ")")
D[,"Total Estimated Annual Hosps Saved, Whole US"]<- paste0(round(vs.nws*all_medicare/denom), " (", round(vs.zero*all_medicare/denom), ")")
D[,c("Median Diff.", "WMW stat", "p-value")]<- apply(D[,c("Median Diff.", "WMW stat", "p-value")],
                                                     MARGIN=2, as.character)

print(xtable(D),include.rownames=TRUE)
