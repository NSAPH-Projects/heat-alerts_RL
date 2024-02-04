library(ggplot2)
library(maps)
library(stringr)
library(dplyr)
library(tigris)
library(arrow)
library(ggrepel)

## Get counties:
W<- read_parquet("data/processed/spatial_feats.parquet")
region_vars<- c("Cold", "Hot-Dry", "Marine", "Mixed-Dry", "Mixed-Humid", "Very Cold")
Region<- apply(W[,region_vars],
               MARGIN=1, function(x) region_vars[which(x == 1)])
Region[lengths(Region)==0]<- "Hot-Humid"
Region<- unlist(Region)

all_counties<- data.frame(Fips=unique(data$fips), Region)
rl_set_30<- c(41067, 53015, 20161, 37085, 48157,
              28049, 19153, 17167, 31153, 6071, 4013,
              34021, 19155, 17115, 29021, 29019, 5045, 40017, 21059,
              47113, 42017, 22109, 45015, 13031, 48367, 22063, 41053,
              32003, 4015, 6025)
all_counties$`RL Analysis Set`<- all_counties$Fips %in% rl_set_30


## Get map data:

shp<- counties()
# counties<- select(data.frame(shp[,c("GEOID", "STATEFP", "INTPTLON", "INTPTLAT")]), -geometry)
counties<- data.frame(shp[,c("GEOID", "STATEFP", "INTPTLON", "INTPTLAT")])
counties$GEOID<- as.numeric(counties$GEOID)

Counties<- full_join(all_counties, counties, by = c("Fips" = "GEOID"))
Counties$long<- as.numeric(Counties$INTPTLON)
Counties$lat<- as.numeric(Counties$INTPTLAT)

## Subset:

#### Make map:

p<- ggplot() + geom_sf(data=Counties, aes(geometry=geometry, fill = Region)) + 
  xlab("Longitude") + ylab("Latitude") +
  xlim(c(-125, -67)) + ylim(c(24, 50))

p + geom_text_repel(data=Counties[which(Counties$`RL Analysis Set`),],
               aes(x=long, y=lat, label=as.factor(Fips)), max.overlaps = Inf)
  
