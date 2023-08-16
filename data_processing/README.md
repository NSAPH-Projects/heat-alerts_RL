# Instructions for refactored data processing

## Philosophy

* Except for health data, it should run on any computer. No absolute paths.
* It should minimize the number of files that have to be manually downloaded, preferring automatic downloads.
* The user should not need to manually edit any files or scripts.
A container and conda env for full reproducibility should be provided. Using packrat for R in addition to specifying versions in the docker file.
* Flexibility: e.g., adding and removing covariates should be easy.
* Process first, merge last. This makes it easier to add new data sources and share what is sharable.
* Intermediate and output files should be saved in portable formats. No R-specific formats.


