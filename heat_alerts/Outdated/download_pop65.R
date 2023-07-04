url <- paste0(
    "https://www2.census.gov/programs-surveys/popest/datasets/2000-2010",
    "/intercensal/county/co-est00int-agesex-5yr.csv"
)
tgt_dir <- "./data/county_census_data"
# make dir if not exists
if (!dir.exists(tgt_dir)) {
    dir.create(tgt_dir)
}
tgt_file <- sprintf("%s/co-est00int-agesex-5yr.csv", tgt_dir)
if (!file.exists(tgt_file)) {
    curl_download(url, tgt_file, quiet = FALSE)
}

# read csv
cols <- paste0("POPESTIMATE", 2006:2010)
early <- read_csv(tgt_file) %>%
    mutate(GEOID = paste0(STATE, COUNTY)) %>%
    filter(AGEGRP >= 14 & SEX == 0) %>%
    select(GEOID, .env$cols) %>%
    # reshape to long format except GEOID
    pivot_longer(cols = .env$cols, names_to = "year", values_to = "pop_65") %>%
    mutate(year = as.integer(str_remove(year, "POPESTIMATE"))) %>%
    # add population per county
    group_by(GEOID, year) %>%
    summarise(pop_65 = sum(pop_65), .groups = "drop")

# b. Population of >65 years old for 2011-2018
url <- paste0(
    "https://www2.census.gov/programs-surveys/popest/datasets/2010-2020/counties/",
    "/asrh/CC-EST2020-AGESEX-ALL.csv"
)
tgt_file <- sprintf("%s/nc-est2020-agesex-alldata.csv", tgt_dir)
if (!file.exists(tgt_file)) {
    curl_download(url, tgt_file, quiet = FALSE)
}
# make sure STATE and COUNTY variable is read as string
late <- read_csv(tgt_file) %>%
    mutate(
        GEOID = paste0(str_pad(STATE, 2, pad = "0"), str_pad(COUNTY, 3, pad = "0")),
        pop_65 = AGE6569_TOT + AGE7074_TOT + AGE7579_TOT + AGE8084_TOT + AGE85PLUS_TOT,
        year = YEAR + 2007
    ) %>%
    filter(year %in% 2011:2018) %>%
    select(GEOID, year, pop_65)

all_pop_65 <- bind_rows(early, late)
counties <- counties %>%
    left_join(all_pop_65, by = "GEOID")