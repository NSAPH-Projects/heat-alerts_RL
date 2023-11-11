
#### Note: this method is not able to find all the counties we want!

## Get train and test sets from Q_learning.R

library(ggplot2)
library(maps)
library(stringr)
library(dplyr)

library(tigris, lib.loc = "~/apps/R_4.2.2")

load("data/Train-Test.RData")

## Get map data:

shp<- counties()
counties<- select(data.frame(shp[,c("GEOID", "STATEFP", "INTPTLON", "INTPTLAT")]), -geometry)
counties$STATEFP<- as.numeric(counties$STATEFP)
# counties<- map_data("county")
# counties$county<- str_to_title(counties$subregion)
# counties$state<- str_to_title(counties$region)

train_fips<- distinct(Train[,c("fips", "Population", "county", "STATE")])
train_fips$county<- sapply(train_fips$county, function(s) strsplit(s, " ")[[1]][2])
test_fips<- distinct(Test[,c("fips", "Population", "county", "STATE")])
test_fips$county<- sapply(test_fips$county, function(s) strsplit(s, " ")[[1]][2])
fips<- rbind(train_fips, test_fips)

Counties<- full_join(fips, counties, by = c("STATE" = "STATEFP", "fips" = "GEOID"))
Counties$set<- "Not in dataset"
Counties[which(Counties$fips %in% test_fips$fips), "set"]<- "Test"
Counties[which(Counties$fips %in% train_fips$fips), "set"]<- "Train"
Counties[which(Counties$Population < 65000), "set"]<- "Too few people"

## Save LL info for later:
Counties$long<- Counties$INTPTLON
Counties$lat<- Counties$INTPTLAT
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





