library(torch)
# ## Had issues installing torch on FASRC, so ended up downloading the files a la this vignette: https://cran.r-project.org/web/packages/torch/vignettes/installation.html
# get_install_libs_url(type = "10.2")
# download.file("https://download.pytorch.org/libtorch/cu102/libtorch-cxx11-abi-shared-with-deps-1.11.0%2Bcu102.zip",
#               destfile="/n/home_fasse/econsidine/apps/R_4-0-2/torch/libtorch.zip")
# download.file("https://storage.googleapis.com/torch-lantern-builds/refs/heads/cran/v0.8.0/latest/Linux-gpu-102.zip",
#               destfile="/n/home_fasse/econsidine/apps/R_4-0-2/torch/liblantern.zip")
# install_torch_from_file(libtorch = "file:///n/home_fasse/econsidine/apps/R_4-0-2/torch/libtorch.zip",
#                         liblantern = "file:///n/home_fasse/econsidine/apps/R_4-0-2/torch/liblantern.zip")

library(luz)
torch_tensor(1)

