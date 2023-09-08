
library(dplyr)
library(arrow)
library(rjson)
library(stringr)
library(cdlTools)
library(readr)
library(xtable)

## Get background info:

bayes<- read.csv("heat_alerts/bayesian_model/results/Bayesian_FullFast_8-16.csv", header=FALSE)
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
                SD_Eff = rep(sd_eff$Eff, each=length(A)/761))

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
DF<- inner_join(distinct(DF[,c("Fips", "Region", "Alerts", "SD_Eff")]), States[,c("Fips", "State")])

## Add evaluation results:

results<- read.csv("Fall_results/Final_eval_30_best-T7-T8.csv")
Results<- results[,c("Fips", "Random", "NWS", "Eval", "opt_HI_thr", "Best_Iter")]
names(Results)[ncol(Results)-1]<- "QHI_thr"
Results$Best_Iter<- Results$Best_Iter/1000000

Full<- inner_join(DF[,c("Fips", "Region", "State", "Alerts", "SD_Eff")], Results)

## Add absolute HI for context:

data<- read_csv("data/Summer23_Train_smaller-for-Python.csv")
data$fips<- as.numeric(data$fips)

HI_thr<- rep(0, nrow(Full))
for(k in 1:nrow(Full)){
  d<- data[which(data$fips == Full$Fips[k]),]
  HI_thr[k]<- quantile(d$HImaxF_PopW, Full$QHI_thr[k])
}

Full$HI_thr<- HI_thr



## Print for Latex:

Final<- Full[order(Full$Region, decreasing=TRUE),]
Final$Fips<- as.character(Final$Fips)
Final$Alerts<- as.integer(Final$Alerts)
Final$HI_thr<- as.integer(Final$HI_thr)
Final$QHI_thr<- as.integer(Final$QHI_thr*100)
names(Final)[which(names(Final)=="Eval")]<- "TRPO"
r<- Final$Random
Final$Std_diff<- (Final$TRPO - Final$NWS)/abs(r)

# To save space:
orig<- c("Mixed-Humid", "Marine", "Hot-Humid", "Hot-Dry", "Cold")
ab<- c("MxHd", "Mrn", "HtHd", "HtDr", "Cold")
Final$Region<- as.vector(sapply(Final$Region, function(x){ab[match(x,orig)]}))
Final$Fips_ST<- apply(Final[,c("Fips", "State")], MARGIN=1, function(x){
  paste0(paste(x[1], x[2], sep=" ("),")")
})

print(xtable(Final[,c("Fips_ST","Region","SD_Eff", "Alerts","Random","NWS",
                      "TRPO", "Std_diff", "QHI_thr", "HI_thr", "Best_Iter")], 
             digits=3, hline.after = 1:nrow(Final)), include.rownames=FALSE)

