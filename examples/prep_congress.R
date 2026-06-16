# Build the Congressional-bills corpus for examples/ectm_congress.py from the
# keyATM replication archive (Eshima, Imai & Sasaki 2024, AJPS),
# Harvard Dataverse doi:10.7910/DVN/RKNNVL.
#
# Writes a compact sparse corpus to examples/congress_data/:
#   counts.mtx  vocab.txt  meta.csv  (docid, congress, chamber, cap_topic)
#
# Group  = chamber (House / Senate)        Time = Congress (101..114, 1989-2017)
# cap_topic is the Comparative Agendas Project human-coded primary topic.
#
# Requires R packages: quanteda, Matrix.  Run from the repo root:
#   Rscript examples/prep_congress.R

suppressMessages({library(quanteda); library(Matrix)})

dir.create("examples/congress_data", showWarnings = FALSE, recursive = TRUE)
api <- "https://dataverse.harvard.edu/api/access/datafile/"
get <- function(id, dest) if (!file.exists(dest))
  download.file(paste0(api, id), dest, mode = "wb", quiet = TRUE)

get(6390635, "examples/congress_data/bill_dfm.rds")            # 4421-bill dfm
get(6390639, "examples/congress_data/true_topic.tab")          # CAP gold topics

dfm     <- readRDS("examples/congress_data/bill_dfm.rds")
dn      <- docnames(dfm)
cong    <- as.integer(sub("([0-9]+)th-congress.*", "\\1", dn))
chamber <- ifelse(grepl("house-bill", dn), "House", "Senate")

topic_names <- c("Macroeconomics","Civil rights","Health","Agriculture","Labor",
  "Education","Environment","Energy","Immigration","Transportation","Law and crime",
  "Social welfare","Housing","Domestic commerce","Defense","Technology","Foreign trade",
  "International affairs","Government operations","Public lands","Culture")
tt  <- read.delim("examples/congress_data/true_topic.tab", stringsAsFactors = FALSE)
M   <- as.matrix(tt[match(sub(".txt$", "", dn), tt$docid), paste0("X", 1:21)])
rs  <- rowSums(M, na.rm = TRUE)
cap <- ifelse(rs == 0 | is.na(rs), "Unlabeled",
              topic_names[max.col(replace(M, is.na(M), 0), ties.method = "first")])

Matrix::writeMM(as(dfm, "CsparseMatrix"), "examples/congress_data/counts.mtx")
writeLines(featnames(dfm), "examples/congress_data/vocab.txt")
write.csv(data.frame(docid = sub(".txt$", "", dn), congress = cong,
                     chamber = chamber, cap_topic = cap),
          "examples/congress_data/meta.csv", row.names = FALSE)

cat(sprintf("wrote %d bills x %d words across Congress %d-%d (House %d / Senate %d)\n",
            nrow(dfm), nfeat(dfm), min(cong), max(cong),
            sum(chamber == "House"), sum(chamber == "Senate")))
