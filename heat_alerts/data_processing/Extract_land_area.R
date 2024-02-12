## First, downloaded this shapefile from https://www2.census.gov/geo/tiger/TIGER2010/COUNTY/2010/

library(raster)

shp<- shapefile("tl_2010_us_county10.shp")


df<- data.frame(fips = shp$GEOID10, Area = shp$ALAND10)

df$Area<- as.numeric(df$Area)*(3.86102e-7) # converting square meters to square miles

write.csv(df, "Counties_land_area.csv", row.names = FALSE)
