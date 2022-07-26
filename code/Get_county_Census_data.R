library(yaml)
library(tidycensus)

setwd("C:/Users/ellen/OneDrive/MyDocs/Graduate Research/Heat alerts RL")

v13<- load_variables(2013, "acs5", cache = TRUE)


## Only looking at Medicare-age people:

pop_65<- get_acs(geography = "county", year = 2013, 
                 variables = c("B01001_020", "B01001_021",
                               "B01001_022", "B01001_023",
                               "B01001_024", "B01001_025",
                               "B01001_044", "B01001_045",
                               "B01001_046", "B01001_047",
                               "B01001_048", "B01001_049"))

Pop_65<- data.frame(GEOID = pop_65$GEOID, Pop.65 = pop_65$estimate)

All_pop_65<- aggregate(Pop.65 ~ GEOID, data = Pop_65, sum)

write.csv(All_pop_65, "Pop_Medicare-age.csv", row.names = FALSE)

## All ages:

pop<- get_acs(geography = "county", year = 2013, 
                      variables = c("B01001_001"))

med.HH<- get_acs(geography = "county", year = 2013, 
                        variables = c("B19013_001"))

County_data<- data.frame(GEOID = pop$GEOID, Population = pop$estimate, 
                         Med.HH.Income = med.HH$estimate)

write.csv(County_data, "County_data.csv", row.names = FALSE)

