
## Get train and test sets from Q_learning.R

library(ggplot2)
library(maps)
library(stringr)
library(dplyr)

## Get map data:

counties<- map_data("county")
counties$county<- str_to_title(counties$subregion)
counties$state<- str_to_title(counties$region)

fips<- distinct(summer[,c("state", "GEOID", "Population", "county", "STNAME")])
fips$county<- sapply(fips$county, function(s) strsplit(s, " ")[[1]][2])

Counties<- full_join(counties, fips, by = c("state" = "STNAME", "county"))
Counties$set<- "Not in dataset"
Counties[which(Counties$GEOID %in% too_few), "set"]<- "Too few people"
Counties[which(Counties$GEOID %in% test_fips), "set"]<- "Test"
Counties[which(Counties$GEOID %in% train_fips), "set"]<- "Train"
Counties[which(Counties$GEOID %in% validation_fips), "set"]<- "Validation"

## Make map:

ggplot() + geom_polygon(data=Counties, aes(x=long, y=lat, group = group,
                                           fill = set)) + 
  scale_fill_manual("", 
                      breaks = c("Train",
                                 "Validation",
                                 "Test",
                                 "Too few people",
                                 "Not in dataset"),
                      values = c("light green", "orange", "magenta", 
                                 "blue", "gray")) + 
  xlab("Longitude") + ylab("Latitude")


