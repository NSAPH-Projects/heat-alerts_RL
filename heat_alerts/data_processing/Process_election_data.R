
library(dplyr)

data<- read.csv("data/countypres_2000-2020.csv") # downloaded from https://doi.org/10.7910/DVN/VOQCHQ 
all_pos<- which(data$year %in% c(2004, 2008, 2012, 2016))
all_pos<- intersect(all_pos, which(!is.na(data$county_fips)))

Rates<- (data$candidatevotes / data$totalvotes)

dem_pos<- intersect(which(data$party == "DEMOCRAT"), all_pos)
rep_pos<- intersect(which(data$party == "REPUBLICAN"), all_pos)
## Note: there isn't data for Green or Libertarian candidates in my years of interest

dem_df<- data.frame(year = data$year[dem_pos], 
                    county_fips = data$county_fips[dem_pos],
                    Democrat = Rates[dem_pos])
rep_df<- data.frame(year = data$year[rep_pos], 
                    county_fips = data$county_fips[rep_pos],
                    Republican = Rates[rep_pos])

DF<- full_join(dem_df, rep_df)

write.csv(DF, "data/Cleaned_election_data.csv", row.names=FALSE)

