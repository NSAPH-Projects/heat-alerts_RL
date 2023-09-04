
library(dplyr)
library(arrow)
library(rjson)
library(stringr)
library(cdlTools)
library(readr)
library(xtable)

## Get background info:

A<- read_parquet("data/processed/actions.parquet")$alert
locs<- read_parquet("data/processed/location_indicator.parquet")[,1]$sind

crosswalk<- unlist(fromJSON(file="data/processed/fips2idx.json"))
fips<- names(crosswalk)

sum_alerts<- aggregate(A ~ locs, data.frame(A,locs), sum)

W<- as.data.frame(read_parquet("data/processed/spatial_feats.parquet"))
region_vars<- c("Cold", "Hot-Dry", "Marine", "Mixed-Dry", "Mixed-Humid", "Very Cold")
Region<- apply(W[,region_vars],
               MARGIN=1, function(x) region_vars[which(x == 1)])
Region[lengths(Region)==0]<- "Hot-Humid"
Region<- unlist(Region)

DF<- data.frame(County = fips[locs+1],
                Region = Region[locs+1],
                Alerts = rep(sum_alerts$A, each=length(A)/761))

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
DF<- inner_join(distinct(DF[,c("Fips", "Region", "Alerts")]), States[,c("Fips", "State")])

## Add evaluation results:

results<- read.csv("Fall_results/Final_eval_30_best-T7-T8.csv")
Results<- results[,c("Fips", "Random", "NWS", "Eval", "opt_HI_thr")]
names(Results)[ncol(Results)]<- "QHI_thr"

Full<- inner_join(DF[,c("Fips", "Region", "State", "Alerts")], Results)

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
Final$HI_thr<- as.integer(Final$HI_thr)
Final$QHI_thr<- as.integer(Final$QHI_thr*100)
names(Final)[which(names(Final)=="Eval")]<- "TRPO"
r<- Final$Random
Final$Std_Diff<- (Final$TRPO - Final$NWS)/abs(r)

print(xtable(Final[,c(1:7,10,8:9)], 
             digits=3, hline.after = 1:nrow(Final)), include.rownames=FALSE)

