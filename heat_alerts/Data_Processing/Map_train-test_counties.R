
## Get train and test sets from Q_learning.R

library(ggplot2)
library(maps)
library(stringr)
library(dplyr)

load("data/Train-Test.RData")

## Get map data:

counties<- map_data("county")
counties$county<- str_to_title(counties$subregion)
counties$state<- str_to_title(counties$region)

train_fips<- distinct(Train[,c("fips", "Population", "county", "STNAME")])
train_fips$county<- sapply(train_fips$county, function(s) strsplit(s, " ")[[1]][2])
test_fips<- distinct(Test[,c("fips", "Population", "county", "STNAME")])
test_fips$county<- sapply(test_fips$county, function(s) strsplit(s, " ")[[1]][2])
fips<- rbind(train_fips, test_fips)

Counties<- full_join(counties, fips, by = c("state" = "STNAME", "county"))
Counties$set<- "Not in dataset"
too_few<- which(Counties$Population < 65000)
Counties[which(Counties$fips %in% test_fips$fips), "set"]<- "Test"
Counties[which(Counties$fips %in% train_fips$fips), "set"]<- "Train"
Counties[which(Counties$fips %in% too_few), "set"]<- "Too few people"

## Save LL info for later:
saveRDS(Counties[,c("fips", "long", "lat", "set")], "data/County_coords.rds")

## Make map:

ggplot() + geom_polygon(data=Counties, aes(x=long, y=lat, group = group,
                                           fill = set)) + 
  scale_fill_manual("", 
                      breaks = c("Train",
                                 "Test",
                                 "Too few people",
                                 "Not in dataset"),
                      values = c("light green",  "blue", "orange", "gray")) + 
  xlab("Longitude") + ylab("Latitude")





