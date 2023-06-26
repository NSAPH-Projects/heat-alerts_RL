FROM mambaorg/micromamba:1.4.4-focal

# we need root to install libraries
USER root

# install build essentials
RUN apt update && apt install -y build-essential git wget

# install python 3.10, r, pip
RUN micromamba install -y -n base -c conda-forge python=3.10

# Make RUN commands use the new environment:
SHELL ["micromamba", "run", "-n", "base", "/bin/bash", "-c"]

# Copy env.yaml and update conda environment
COPY env.yaml /tmp/env.yaml
RUN micromamba install -n base -f /tmp/env.yaml

# # Copy over install_r_packages.R and install packages
# COPY ./install_r_packages.sh /tmp/install_r_packages.sh
# RUN bash /tmp/install_r_packages.sh

# Install additional R packages
# Add default CRAN repo to https://cran.rstudio.com/ in .Rprofile
RUN echo 'options(repos = c(CRAN = "https://cran.rstudio.com/"))' >> ~/.Rprofile
RUN Rscript -e 'install.packages("sf")'
