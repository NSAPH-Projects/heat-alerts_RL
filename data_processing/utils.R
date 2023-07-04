download_and_uncompress <- function(url, tgt_dir) {
    tgt_file <- sprintf("%s/%s", tgt_dir, basename(url))
    if (!file.exists(tgt_file)) {
        # make dir if not exists
        if (!dir.exists(tgt_dir)) {
            dir.create(tgt_dir)
        }

        # download
        curl::curl_download(url, tgt_file, quiet = FALSE)

        # unzip
        unzip(tgt_file, exdir = tgt_dir)

        # rm zip
        file.remove(tgt_file)
    }
}
