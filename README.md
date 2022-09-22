# heat-alerts_mortality_RL

This is code for investigating applicability of RL to environmental health, specifically issuance of heat alerts.

### Data Processing:
1. Merging mortality data and heat alerts data: Explore_merged_data.R
2. Processing county-level covariates: Extract_land_area.R, Get_county_Census_data.R, Prep_DoE_zones.R
3. Merge together: Merge-finalize_county_data.R

### Installing Pytorch on FASRC/FASSE:
1. Follow [these instructions](https://github.com/fasrc/User_Codes/tree/master/AI/PyTorch)
2. Additionally, run this from terminal:
```bash
conda install scikit-learn
conda install -c conda-forge tqdm
```
3. Might need to manually install pandas as well depending on the conda environment version...

### Installing torch for R on FASRC/FASSE:
*Note: we couldn't get torch for R to correctly use GPUs, so we switched to Pytorch*

The Harvard cluster's Linux version is too old to support this package. Thus, we need to use a container where we can install everything we need. [Here is helpful documentation on using Singularity Containers on the cluster](https://docs.rc.fas.harvard.edu/wp-content/uploads/2022/08/Containers_on_Cannon_08_22.pdf).

Use FASRC Remote Desktop (NOT Containerized version), set up using fasse_gpu partition. Then work from the remote desktop terminal. *Tip: use Clipboard functionality (from blue bar on LHS of screen) to paste text into the virtual desktop*

The first time:
1. cd [working directory path]
2. singularity pull docker://rockerdev/ml:4.0.0-cuda-10.2 # it can take a while to set up the container image
3. R
4. install.packages("torch")
5. install.packages("https://cran.r-project.org/src/contrib/Archive/rlang/rlang_0.4.12.tar.gz", repo=NULL, type="source")
 
All subsequent times:
1. cd [working directory path]
2. singularity shell --bind [working directory path]  --nv ml_4.0.0-cuda-10.2.sif --num_gpus=4
3. R
4. library("rlang", lib.loc= [path to where the newer rlang is installed, for me it's "/n/home_fasse/econsidine/R/x86_64-pc-linux-gnu-library/"])
5. library("torch")
6. cuda_is_available() # TRUE if you requested gpus on fasse_gpu and enabled gpus on the singularity container (--nv and --num_gpus arguments)

*Note on using singularity containers: beyond what's included in the FASRC docs, there is a lot of helpful documentation online, e.g. describing how "singularity cache list" shows all existing container images, and how "singularity cache clean" removes everything in the cache.*

### Analysis:
