library(raster)
library(stringr)
library(tidyverse)
library(sf)
library(tidycensus)
library(curl)
library(argparse)
library(arrow)

# import utilities
utils <- new.env()
source("data_processing/utils.R", local = utils)

# argparse
parser <- argparse::ArgumentParser()
parser$add_argument("--pop65", default=FALSE, action="store_true")
args <- parser$parse_args()

# ===== 1. shapefile =====
url <- "https://www2.census.gov/geo/tiger/TIGER2010/COUNTY/2010/tl_2010_us_county10.zip"
tgt_dir <- "./data/shapefile"
tgt_shpfile <- sprintf("%s/tl_2010_us_county10.shp", tgt_dir)

# unzip shapefile if not available
if (!file.exists(tgt_shpfile)) {
    utils$download_and_uncompress(url, tgt_dir)
}

# read shapefile and add land area
counties <- st_read(tgt_shpfile) %>%
    rename(GEOID = GEOID10) %>%
    mutate(area = ALAND10 * 3.86102e-7) %>% # in sq miles %>%
    mutate(lon = as.numeric(INTPTLON10),
           lat = as.numeric(INTPTLAT10)) %>%
    select(GEOID, lat, lon, area) %>%
    st_drop_geometry() %>%
    as_tibble()

# ==== 2. county census data =====
pop <- get_acs(
    geography = "county",
    year = 2013,
    variables = c("B01001_001")
)

income <- get_acs(
    geography = "county",
    year = 2013,
    variables = c("B19013_001")
)

ses_data <- data.frame(
    GEOID = pop$GEOID,
    total_pop = pop$estimate,
    med_hh_income = income$estimate
)

# merge with shapefile
counties <- counties %>%
    left_join(ses_data, by = "GEOID") %>%
    mutate(pop_density=total_pop/area)

# ==== 3. internet usage =====
# proxy from 2020

url <- paste0(
    "https://raw.githubusercontent.com/microsoft/USBroadbandUsagePercentages/",
    "c84ce57cfcc09f77c0b9803aed164a70d1655131/dataset/broadband_data_2020October.csv"
)

tgt_file <- "./data/broadband_data.csv"
if (!file.exists(tgt_file)) {
    curl_download(url, tgt_file, quiet = FALSE)
}
broadband <- read_csv(tgt_file) %>%
    mutate(
        GEOID = str_pad(`COUNTY ID`, 5, pad = "0"),
        broadband_usage = `BROADBAND USAGE`,
    ) %>%
    select(GEOID, broadband_usage) %>%
    mutate(GEOID = case_when(
        GEOID == "46102" ~ "46113",
        TRUE ~ GEOID
    ))

# save spatial confounders
write_parquet(counties, "./data/processed/spatial_confounders.parquet")

# test read
counties <- read_parquet("./data/processed/spatial_confounders.parquet")
