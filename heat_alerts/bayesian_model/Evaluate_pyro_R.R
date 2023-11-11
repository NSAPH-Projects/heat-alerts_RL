
library(arrow)
library(rjson)

# bayes<- read.csv("heat_alerts/bayesian_model/results/Bayesian_FF_C-all_wide-EB-prior.csv", header=FALSE)
# bayes<- read.csv("heat_alerts/bayesian_model/results/Bayesian_FF_C-none_wide-EB-prior.csv", header=FALSE)
# bayes<- read.csv("heat_alerts/bayesian_model/results/Bayesian_FF_C-HI_wide-EB-prior.csv", header=FALSE)
# bayes<- read.csv("heat_alerts/bayesian_model/results/Bayesian_FF_C-A_wide-EB-prior.csv", header=FALSE)
bayes<- read.csv("heat_alerts/bayesian_model/results/Bayesian_FF_C-M_wide-EB-prior.csv", header=FALSE)

#### Compare to observations:
Y<- read_parquet("data/processed/outcomes.parquet")$other_hosps
A<- read_parquet("data/processed/actions.parquet")$alert
offset<- read_parquet("data/processed/offset.parquet")[,1]
denom<- read_parquet("data/processed/Medicare_denominator.parquet")[,1]

# Calculating R^2 on the daily rates of NOHR hospitalizations:
pred_Y<- bayes$V3/denom
Y<- Y/denom

cor(Y, pred_Y)^2
1 - (sum((Y - pred_Y)^2)/(((length(A)-1)*var(Y))))


#### Obtain locations with highest effectiveness:

baseline<- bayes$V2
effectiveness<- bayes$V1

locs<- read_parquet("data/processed/location_indicator.parquet")[,1]#$sind

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
                Eff = effectiveness,
                Alerts = rep(sum_alerts$A, each=length(A)/761))
# DF<- DF[A==1,]

DF[order(DF$Eff, decreasing=TRUE),][0:20,]

agg_DF<- aggregate(. ~ County + Region + Alerts, DF, mean)
var_DF<- aggregate(Eff ~ County + Region + Alerts, DF, sd)
var_DF$Mean_Eff<- agg_DF$Eff
var_DF$Eff<- round(var_DF$Eff, 4)

many_a<- var_DF[which(var_DF$Alerts >= 75),]
O<- many_a[order(many_a$Eff, decreasing=TRUE),]
O[which(O$Region == "Mixed-Humid"),] # manually run through the five main regions 

agg_DF[order(agg_DF$Eff, decreasing=TRUE),]
agg_DF[order(agg_DF$Eff, decreasing=TRUE),][0:100,]
agg_DF[order(agg_DF$Alerts, decreasing=TRUE),][0:20,]

var_DF[order(var_DF$Eff, decreasing=TRUE),][0:100,]


#### Sanity check:

library(usmap)
library(ggplot2)
library(dplyr)

my_obj<- us_map(regions = "counties")
my_obj$fips<- as.numeric(my_obj$fips)
agg_DF$County<- as.numeric(agg_DF$County)

my_obj<- inner_join(my_obj, agg_DF, by=c("fips" = "County"))

ggplot(my_obj, aes(x,y,group=group,fill=as.factor(Region))) +
  geom_polygon() + scale_fill_discrete()
