
library(arrow)
library(rjson)

# bayes<- read.csv("Bayesian_models/Bayesian_R_7-19.csv", header=FALSE)
# bayes<- read.csv("Bayesian_models/Bayesian_Full_8-4.csv", header=FALSE)
bayes<- read.csv("Bayesian_models/Bayesian_Full_8-7.csv", header=FALSE)

#### Compare to observations:
Y<- read_parquet("bayesian_model/data/processed/outcomes.parquet")$other_hosps
A<- read_parquet("bayesian_model/data/processed/actions.parquet")$alert
offset<- read_parquet("bayesian_model/data/processed/offset.parquet")[,1]
denom<- read_parquet("bayesian_model/data/processed/Medicare_denominator.parquet")[,1]

# pred_Y<- bayes$R0
# pred_Y[A==1]<- bayes$R1[A==1]
# pred_Y<- pred_Y*offset

# pred_Y<- bayes$V3#/offset
# Y<- Y/offset
pred_Y<- bayes$V3/denom
Y<- Y/denom

cor(Y, pred_Y)^2
1 - (sum((Y - pred_Y)^2)/(((length(A)-1)*var(Y))))


#### Obtain locations with highest effectiveness:

effectiveness<- bayes$V1

locs<- read_parquet("bayesian_model/data/processed/location_indicator.parquet")[,1]

crosswalk<- unlist(fromJSON(file="bayesian_model/data/processed/fips2idx.json"))
fips<- names(crosswalk)

W<- as.data.frame(read_parquet("bayesian_model/data/processed/spatial_feats.parquet"))
region_vars<- c("Cold", "Hot-Dry", "Marine", "Mixed-Dry", "Mixed-Humid", "Very Cold")
Region<- apply(W[,region_vars],
               MARGIN=1, function(x) region_vars[which(x == 1)])
Region[lengths(Region)==0]<- "Hot-Humid"
Region<- unlist(Region)

DF<- data.frame(County = fips[locs+1],
                Region = Region[locs+1],
                Eff = effectiveness)
DF<- DF[A==1,]

DF[order(DF$Eff, decreasing=TRUE),][0:30,]

agg_DF<- aggregate(Eff ~ County + Region, DF, mean)
agg_DF[order(agg_DF$Eff, decreasing=TRUE),][0:30,]


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
