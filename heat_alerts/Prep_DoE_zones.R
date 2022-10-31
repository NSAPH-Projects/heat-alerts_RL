
library(stringr)

setwd("C:/Users/ellen/OneDrive/MyDocs/Graduate Research/Heat alerts RL")

zones<- read.csv("DoE_climate_zones.csv") # downloaded from https://gist.github.com/philngo/d3e251040569dba67942 on 7/20/22

states<- str_pad(zones$State.FIPS, 2, pad = "0")
counties<- str_pad(zones$County.FIPS, 3, pad = "0")

GEOID<- paste0(states, counties)

Zones<- data.frame(GEOID, IECC_zone=zones$IECC.Climate.Zone, 
                   BA_zone=zones$BA.Climate.Zone)

write.csv(Zones, "Prepped_zones.csv", row.names = FALSE)
