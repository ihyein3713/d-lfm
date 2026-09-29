# Convert R data files (.rda) to CSV with R itself.
#
# Needed for the ADNIMERGE2 tables that the python readers cannot parse (ADSL, ADQS): they use
# serialization features pyreadr/rdata do not support.
#
# Usage:
#   Rscript custom/rda_to_csv.R [SRC_DIR] [OUT_DIR] [NAME ...]
#     SRC_DIR  directory with .rda files   (default: the ADNIMERGE2 data dir below)
#     OUT_DIR  directory for the CSVs      (default: <SRC_DIR>/../csv)
#     NAME     dataset names to convert, without .rda (default: all)

args <- commandArgs(trailingOnly = TRUE)
src <- if (length(args) >= 1) args[1] else "/mnt/aix22308/data/ADNI_delta_lfm/meta/ADNIMERGE2/data"
out <- if (length(args) >= 2) args[2] else file.path(dirname(src), "csv")
only <- if (length(args) >= 3) args[-(1:2)] else character(0)

dir.create(out, showWarnings = FALSE, recursive = TRUE)
files <- list.files(src, pattern = "\\.rda$", full.names = TRUE)
if (length(only) > 0) {
  files <- files[tools::file_path_sans_ext(basename(files)) %in% sub("\\.rda$", "", only)]
}
if (length(files) == 0) stop("no .rda files found in ", src)

for (f in files) {
  env <- new.env()
  ok <- tryCatch({ load(f, envir = env); TRUE },
                 error = function(e) { cat(sprintf("%-34s FAILED  %s\n", basename(f), conditionMessage(e))); FALSE })
  if (!ok) next
  for (nm in ls(env)) {
    obj <- get(nm, envir = env)
    if (!is.data.frame(obj)) {
      cat(sprintf("%-34s SKIP    %s is %s, not a data.frame\n", basename(f), nm, class(obj)[1]))
      next
    }
    # list columns (rare in ADNI tables) cannot go into a flat CSV; collapse them
    for (cn in names(obj)) if (is.list(obj[[cn]])) obj[[cn]] <- sapply(obj[[cn]], paste, collapse = ";")
    dst <- file.path(out, paste0(nm, ".csv"))
    write.csv(obj, dst, row.names = FALSE, na = "")
    cat(sprintf("%-34s OK      %s.csv (%d x %d)\n", basename(f), nm, nrow(obj), ncol(obj)))
  }
}
cat("written to ", out, "\n", sep = "")
