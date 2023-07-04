library(curl)
library(tidyverse)
library(arrow)
library(argparse)
library(lubridate)


# make path ./data/heatmetrics/ if not available
dir <- "./data/heatmetrics"
if (!dir.exists(dir)) {
    dir.create(dir)
}

# download population-weighted data if not path exists
url <- "https://ndownloader.figstatic.com/files/35070550"
tgt_file <- sprintf("%s/heatmetrics.rds", dir)
if (!file.exists(sprintf("%s/heatmetrics.rds", dir))) {
    curl_download(url, tgt_file, quiet = FALSE)
}

# read rds and transform to portable arrow format
df <- read_rds(tgt_file)

# filter dates and cols to keep
df <- df %>%
    select(StCoFIPS, Date, HImin_C, HImax_C, HImean_C) %>%
    filter(month(Date) >= 5 & month(Date) <= 9) %>%
    filter(year(Date) >= 2006 & year(Date) <= 2019)

# write to parquet
write_parquet(df, sprintf("data/processed/heatmetrics.parquet", dir))

# read parquet to test
df <- read_parquet(sprintf("data/processed/heatmetrics.parquet", dir))
print(head(df))
