library(yaml)
library(tidycensus)

setwd("C:/Users/ellen/OneDrive/MyDocs/Graduate Research/Heat alerts RL")

pop<- get_acs(geography = "county", year = 2013, 
                      variables = c("B01001_001"))

med.HH<- get_acs(geography = "county", year = 2013, 
                        variables = c("B19013_001"))

County_data<- data.frame(GEOID = pop$GEOID, Population = pop$estimate, 
                         Med.HH.Income = med.HH$estimate)

write.csv(County_data, "County_data.csv", row.names = FALSE)
