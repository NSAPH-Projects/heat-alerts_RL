library(yaml)
library(tidycensus)

setwd("C:/Users/ellen/OneDrive/MyDocs/Graduate Research/Heat alerts RL")

v13<- load_variables(2013, "acs5", cache = TRUE)


## Get population 65+ from PEP: https://www2.census.gov/programs-surveys/popest/datasets/
  ## Meta-docs:
    ## https://www2.census.gov/programs-surveys/popest/technical-documentation/file-layouts/2000-2010/intercensal/county/co-est00int-agesex-5yr.pdf
    ## https://www2.census.gov/programs-surveys/popest/technical-documentation/file-layouts/2010-2020/cc-est2020-agesex.pdf

early<- read.csv("co-est00int-agesex-5yr.csv")
early$GEOID<- paste0(early$STATE, str_pad(early$COUNTY, 3, pad="0"))
e.pop.65<- early[which(early$AGEGRP >= 14 & early$SEX == 0), c(21, 9:18, 20)]
E.pop.65<- aggregate(. ~ GEOID, e.pop.65, sum)

Pop.65<- data.frame(GEOID=E.pop.65$GEOID, Pop.65=E.pop.65$POPESTIMATE2006,
                    year=2006)
for(y in 2007:2010){
  new<- data.frame(GEOID=E.pop.65$GEOID, 
                   Pop.65=E.pop.65[,paste0("POPESTIMATE", y)], year=y)
  Pop.65<- rbind(Pop.65, new)
}

late<- read.csv("CC-EST2020-AGESEX-ALL.csv")
late$GEOID<- paste0(late$STATE, str_pad(late$COUNTY, 3, pad="0"))
late$year<- late$YEAR + 2007
l.pop.65<- late[which(late$year %in% 2011:2016), c("GEOID", "year",
                                                      "AGE6569_TOT",
                                                      "AGE7074_TOT", 
                                                      "AGE7579_TOT",
                                                      "AGE8084_TOT",
                                                      "AGE85PLUS_TOT")]
l.pop.65[,3:7]<- apply(l.pop.65[,3:7], MARGIN=2, as.numeric)
L.pop.65<- data.frame(l.pop.65[,1:2], Pop.65=rowSums(l.pop.65[,3:7]))

All.Pop.65<- rbind(L.pop.65, Pop.65[,c("GEOID", "year", "Pop.65")])

write.csv(All.Pop.65, "Pop-Medicare_county-age.csv", row.names=FALSE)

  
## All ages, other SES:

pop<- get_acs(geography = "county", year = 2013, 
                      variables = c("B01001_001"))

med.HH<- get_acs(geography = "county", year = 2013, 
                        variables = c("B19013_001"))

County_data<- data.frame(GEOID = pop$GEOID, Population = pop$estimate, 
                         Med.HH.Income = med.HH$estimate)

write.csv(County_data, "County_data.csv", row.names = FALSE)

