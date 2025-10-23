library(clusterProfiler)
library(org.Hs.eg.db)
library(dplyr)

# convert_ids_with_bitr > Funcion que permite leer un archivo, extraer IDs y convertirlos al tipo(s) elegido(s)
convert_ids_with_bitr <- function(in_path, has_header, read_first_row, fromType, toType_list, out_path) {
  
  # 1. Read the input text file + 2. Extract IDs to vector
  if (read_first_row) {
    first_row_df <- tryCatch(
      {
        read.delim(in_path, header = FALSE, nrows = 1, sep = "\t", stringsAsFactors = FALSE)
      },
      error = function(e) {
        stop(paste("Error reading input file (first row):", e$message))
      }
    )
    original_ids <- unlist(first_row_df)
    names(original_ids) <- NULL
    rm(first_row_df)
    
    if (has_header) {  # Drop "case_id" or any other first item
      original_ids <- original_ids[-1]
    }
  } else {
    ids_df <- tryCatch(
      {
        read.delim(in_path, header = has_header, sep = "\t", stringsAsFactors = FALSE)
      },
      error = function(e) {
        stop(paste("Error reading input file:", e$message))
      }
    )
    original_ids <- ids_df[[1]]
    rm(ids_df)
  }
  
  # (Aux) Check IDs available for organism
  all_from_ids_in_db <- keys(org.Hs.eg.db, keytype = fromType)
  matched_ids_count <- sum(original_ids %in% all_from_ids_in_db)
  print('IDs in organism:')
  print(matched_ids_count)
  
  # 3. Convert IDs from fromType to each toType(s) + Reshape
  result_df <- tryCatch(
    {
      bitr(
        geneID = original_ids,
        fromType = fromType,
        toType = toType_list,
        OrgDb = org.Hs.eg.db,
        drop = FALSE
      )
    },
    error = function(e) {
      warning(paste0("Conversion failed: ", e$message))
      return(data.frame())
    }
  )
  
  print('BITR results:')
  print(nrow(result_df))
  
  result_df <- result_df[, c('SYMBOL', 'ENTREZID', 'ENSEMBL')]
  result_df <- result_df[order(result_df$SYMBOL),]
  
  # 4. Store the final DataFrame in the out_path
  tryCatch(
    {
      write.table(result_df, out_path, sep = "\t", row.names = FALSE, quote = FALSE)
      cat(paste0("Successfully saved converted IDs to: ", out_path, "\n"))
    },
    error = function(e) {
      stop(paste("Error writing output file:", e$message))
    }
  )
  
  return(result_df)
}

# =======================

# ==== UNIVERSE DATA ====
# Convert SNP IDs
snp_in <- ''  # input file (all genes in model)
snp_out <- ''  # output file
snp_ids <- convert_ids_with_bitr(snp_in, TRUE, TRUE, 'SYMBOL', c('ENTREZID', 'ENSEMBL'), snp_out)

nrow(snp_ids[!is.na(snp_ids$ENTREZID),])
length(unique(snp_ids$ENTREZID))

# Convert RNA IDs
rna_in <- ''  # input file (all genes in model)
rna_out <- ''  # output file
rna_ids <- convert_ids_with_bitr(rna_in, TRUE, TRUE, 'ENSEMBL', c('SYMBOL', 'ENTREZID'), rna_out)

length(unique(rna_ids$ENTREZID))

# Convert CNV IDs
cnv_in <- ''  # input file (all genes in model)
cnv_out <- ''  # output file
cnv_ids <- convert_ids_with_bitr(cnv_in, TRUE, TRUE, 'ENSEMBL', c('SYMBOL', 'ENTREZID'), cnv_out)

length(unique(cnv_ids$ENTREZID))

# Convert MIR IDs
mir_in <- ''  # input file (all mirs in model)
mir_out <- ''  # output file
mir_ids <- convert_ids_with_bitr(mir_in, FALSE, FALSE, 'SYMBOL', c('ENTREZID', 'ENSEMBL'), mir_out)

mir_ids[is.na(mir_ids$ENTREZID),]
nrow(mir_ids[is.na(mir_ids$ENTREZID),])
length(unique(mir_ids$ENTREZID))

# ==== TOP DATA ====
# Convert SNP IDs
top_snp_in <- ''  # input file (top X from SHAP)
top_snp_out <- ''  # output file
top_snp_ids <- convert_ids_with_bitr(top_snp_in, TRUE, FALSE, 'SYMBOL', c('ENTREZID', 'ENSEMBL'), top_snp_out)

# Convert RNA IDs
top_rna_in <- ''  # input file (top X from SHAP)
top_rna_out <- ''  # output file
top_rna_ids <- convert_ids_with_bitr(top_rna_in, TRUE, FALSE, 'ENSEMBL', c('SYMBOL', 'ENTREZID'), top_rna_out)

# Convert CNV IDs
top_cnv_in <- ''  # input file (top X from SHAP)
top_cnv_out <- ''  # output file
top_cnv_ids <- convert_ids_with_bitr(top_cnv_in, TRUE, FALSE, 'ENSEMBL', c('SYMBOL', 'ENTREZID'), top_cnv_out)

# Convert MIR IDs
top_mir_in <- ''  # input file (top X from SHAP)
top_mir_out <- ''  # output file
top_mir_ids <- convert_ids_with_bitr(top_mir_in, FALSE, FALSE, 'SYMBOL', c('ENTREZID', 'ENSEMBL'), top_mir_out)
