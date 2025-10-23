# ======== Prep environment ========
if (!requireNamespace("BiocManager", quietly = TRUE)) {
  install.packages("BiocManager")
}

BiocManager::install(c("clusterProfiler", "GOSemSim", "org.Hs.eg.db", "AnnotationDbi", "enrichplot"))

library(clusterProfiler)
library(org.Hs.eg.db)
library(AnnotationDbi)
library(ggplot2)
library(enrichplot)
library(GOSemSim)

# ==================================


# ============ Prep data ===========
# Define directories
shap_dir <- ''  # main dir
out_dir <- paste0(shap_dir, '/OUTCOME')

# Read input data (Define gene and universe lists)
rna_top_genes <- unique(readLines(paste0(shap_dir, '/RNA_top10_genes.tsv')))
rna_universe <- unique(readLines(paste0(shap_dir, '/RNA_universe.tsv')))
mir_top_genes <- unique(readLines(paste0(shap_dir, '/MIR_top10_target_genes.tsv')))

# Map gene symbols to ENTREZ IDs (more accurate GO anotation)
rna_top_genes_match <- bitr(rna_top_genes,
                             fromType = "SYMBOL",
                             toType = "ENTREZID",
                             OrgDb = org.Hs.eg.db,
                             drop = TRUE)  # drops unmatched genes
rna_top_genes_entrez <- rna_top_genes_match$ENTREZID

mir_top_genes_match <- bitr(mir_top_genes,
                             fromType = "SYMBOL",
                             toType = "ENTREZID",
                             OrgDb = org.Hs.eg.db,
                             drop = TRUE)  # drops unmatched genes
mir_top_genes_entrez <- mir_top_genes_match$ENTREZID

rna_universe_match <- bitr(rna_universe,
                           fromType = "SYMBOL",
                           toType = "ENTREZID",
                           OrgDb = org.Hs.eg.db,
                           drop = TRUE)  # drops unmatched genes
rna_universe_entrez <- rna_universe_match$ENTREZID

# Check unmapped genes
cat('RNA TOP: ', length(rna_top_genes), ' (In) | ', length(unique(rna_top_genes_match$SYMBOL)), ' (MAPPED)')
cat('MIR TOP: ', length(mir_top_genes), ' (In) | ', length(unique(mir_top_genes_match$SYMBOL)), ' (MAPPED)')
cat('RNA UNIVERSE: ', length(rna_universe), ' (In) | ', length(unique(rna_universe_match$SYMBOL)), ' (MAPPED)')

cat('Missing MIR: ', setdiff(mir_top_genes, mir_top_genes_match$SYMBOL))
cat('Missing UNIVERSE: ', setdiff(rna_universe, rna_universe_match$SYMBOL))

# Get missing universe for miRNA data (Dont have from SRC. Use all genes in HSA)
mir_universe_entrez <- keys(org.Hs.eg.db, keytype = "ENTREZID")
# ==================================


# =========== Preliminary analysis ==========
# Explore if top 100 RNA are found in miRNA target list 
common_genes <- intersect(rna_top_genes, mir_top_genes)
# ==================================


# =========== GO analysis ==========
# Perform GO enrichment analysis (3 per dataset)
set_list <- c('RNA', 'MIR')
go_list <- c('BP', 'MF', 'CC')

gene_per_set <- list(
  RNA = rna_top_genes_entrez,
  MIR = mir_top_genes_entrez
)
universe_per_set <- list(
  RNA = rna_universe_entrez,
  MIR = mir_universe_entrez
)

go_result_list <- list()

for (set in set_list) {
  for (ont in go_list) {
    cat(paste0('\nRunning GO analysis (', ont, ') for ', set, '...\n'))
    
    go_result <- enrichGO(gene = gene_per_set[[set]],
                          OrgDb = org.Hs.eg.db,
                          keyType = "ENTREZID",
                          ont = ont,
                          pAdjustMethod = "BH",
                          pvalueCutoff = 0.05,
                          qvalueCutoff = 0.2,
                          readable = TRUE,
                          universe = universe_per_set[[set]])
    
    go_result_list[[set]][[ont]] = go_result
    
    # Print summary + Store results in file
    if (!is.null(go_result) && nrow(as.data.frame(go_result)) > 0) {
      result_df <- as.data.frame(go_result)
      file_name <- paste0(out_dir, "/GO_results_", set, '_', ont, ".csv")
      write.table(result_df, 
                  file = file_name, 
                  row.names = FALSE, 
                  col.names = TRUE,
                  sep = "\t", 
                  quote = FALSE)
      cat(paste0(" >> GO results for ", set, " (", ont, ") stored in file: ", file_name, "\n"))
    } else {
      cat(paste("  >> No significantly enriched GO terms were found for ", set, " (", ont, ")\n"))
    }
  }
}

print('======== GO enrichment analysis DONE ========')
# ==================================

# =========== KEGG analysis ==========
# Perform KEGG enrichment analysis (1 per dataset)
kegg_result_list <- list()
# aux <- list()
for (set in set_list) {
  cat(paste0('\nRunning KEGG analysis for ', set, '...\n'))
  
  kegg_result <- enrichKEGG(gene = gene_per_set[[set]],
                        organism = "hsa",
                        keyType = "kegg",  # refers to result, not genes
                        pAdjustMethod = "BH",
                        pvalueCutoff = 0.05,
                        qvalueCutoff = 0.2,
                        universe = universe_per_set[[set]])
  kegg_result <- setReadable(kegg_result, OrgDb = org.Hs.eg.db, keyType="ENTREZID") # Gene IDs to symbols

  kegg_result_list[[set]][['kegg']] = kegg_result
  
  # Print summary + Store results in file
  if (!is.null(kegg_result) && nrow(as.data.frame(kegg_result)) > 0) {
    result_df <- as.data.frame(kegg_result)
    file_name <- paste0(out_dir, "/KEGG_results_", set, ".csv")
    write.table(result_df,
                file = file_name,
                row.names = FALSE,
                col.names = TRUE,
                sep = "\t",
                quote = FALSE)
    cat(paste0(" >> KEGG results for ", set, " stored in file: ", file_name, "\n"))
  } else {
    cat(paste("  >> No significantly enriched KEGG terms were found for ", set, "\n"))
  }
}

# kegg_result_list = aux

print('======== KEGG enrichment analysis DONE ========')
# ==================================


# =========== Disease-enrichment analysis ==========
# Perform DOSE analysis (1 per dataset)
dose_result_list <- list()
do_list <- c('HDO', 'HPO')

for (set in set_list) {
  for (ont in do_list) {
    cat(paste0('\nRunning DOSE analysis (', ont, ') for ', set, '...\n'))

    dose_result <- enrichDO(gene = gene_per_set[[set]],
                            organism = 'hsa',
                            ont = ont,
                            pAdjustMethod = "BH",
                            pvalueCutoff = 0.05,
                            qvalueCutoff = 0.2,
                            readable = TRUE,
                            universe = universe_per_set[[set]])

    dose_result_list[[set]][[ont]] = dose_result

    # Print summary + Store results in file
    if (!is.null(dose_result) && nrow(as.data.frame(dose_result)) > 0) {
      result_df <- as.data.frame(dose_result)
      file_name <- paste0(out_dir, "/DOSE_results_", set, '_', ont, ".csv")
      write.table(result_df,
                  file = file_name,
                  row.names = FALSE,
                  col.names = TRUE,
                  sep = "\t",
                  quote = FALSE)
      cat(paste0(" >> DOSE results for ", set, " (", ont, ") stored in file: ", file_name, "\n"))
    } else {
      cat(paste("  >> No significantly enriched DOSE terms were found for ", set, " (", ont, ")\n"))
    }
  }
  
  # NETWORK OF CANCER GENE ANNOTATION
  
  ont = "NCG"
  cat(paste0('\nRunning DOSE analysis (', ont, ') for ', set, '...\n'))
  
  dose_result <- DOSE::enrichNCG(gene = gene_per_set[[set]],
                                 pAdjustMethod = "BH",
                                 pvalueCutoff = 0.05,
                                 qvalueCutoff = 0.2,
                                 readable = TRUE,
                                 universe = universe_per_set[[set]],
                                 # minGSSize=1)  # En RNA no hay anyway
                                 )
  
  dose_result_list[[set]][[ont]] = dose_result
  
  # Print summary + Store results in file
  if (!is.null(dose_result) && nrow(as.data.frame(dose_result)) > 0) {
    result_df <- as.data.frame(dose_result)
    file_name <- paste0(out_dir, "/DOSE_results_", set, '_', ont, ".csv")
    write.table(result_df, 
                file = file_name, 
                row.names = FALSE, 
                col.names = TRUE,
                sep = "\t", 
                quote = FALSE)
    cat(paste0(" >> DOSE results for ", set, " (", ont, ") stored in file: ", file_name, "\n"))
  } else {
    cat(paste("  >> No significantly enriched DOSE terms were found for ", set, " (", ont, ")\n"))
  }
  
  # DISGENET ANNOTATION

  ont = "DGN"
  cat(paste0('\nRunning DOSE analysis (', ont, ') for ', set, '...\n'))

  dose_result <- DOSE::enrichDGN(gene = gene_per_set[[set]],
                           pAdjustMethod = "BH",
                           pvalueCutoff = 0.05,
                           qvalueCutoff = 0.2,
                           readable = TRUE,
                           universe = universe_per_set[[set]])

  dose_result_list[[set]][[ont]] = dose_result

  # Print summary + Store results in file
  if (!is.null(dose_result) && nrow(as.data.frame(dose_result)) > 0) {
    result_df <- as.data.frame(dose_result)
    file_name <- paste0(out_dir, "/DOSE_results_", set, '_', ont, ".csv")
    write.table(result_df,
                file = file_name,
                row.names = FALSE,
                col.names = TRUE,
                sep = "\t",
                quote = FALSE)
    cat(paste0(" >> DOSE results for ", set, " (", ont, ") stored in file: ", file_name, "\n"))
  } else {
    cat(paste("  >> No significantly enriched DOSE terms were found for ", set, " (", ont, ")\n"))
  }
}

print('======== DOSE analysis DONE ========')
# ==================================


# ====== Result simplification ======
# Simplify lists (merge similar terms)
go_simplified_list <- list()

for (set in set_list) {
  for (ont in go_list) {
    go_simplified_list[[set]][[ont]] <- simplify(go_result_list[[set]][[ont]],
                                                 cutoff = 0.7,   # Umbral de similitud semántica (con 0.8 perdiamos matches esperables)
                                                 by = "p.adjust", # Criterio para seleccionar el término representativo (aquí el p-valor ajustado más bajo)
                                                 select_fun = min) # Función para seleccionar el término si hay varios con la misma similitud
  }
}

results_list_of_list = list(GO_ALL = go_result_list,
                            GO_SIMPLE = go_simplified_list,
                            KEGG = kegg_result_list,
                            DOSE = dose_result_list)
for (set in set_list) {
  for (ont in go_list) {
    result_df <- as.data.frame(go_simplified_list[[set]][[ont]])
    file_name <- paste0(out_dir, "/GO_results_", set, '_', ont, "_SIMPLIFIED.csv")
    write.table(result_df, 
                file = file_name, 
                row.names = FALSE, 
                col.names = TRUE,
                sep = "\t", 
                quote = FALSE)
  }
}

# ==================================


# ====== Result visualization ======
first_list = list(GO_ALL = "GO",
                  GO_SIMPLE = "GO",
                  KEGG = "KEGG",
                  DOSE = "DOSE")
last_list = list(GO_ALL = "[ALL]",
                 GO_SIMPLE = "[SIMPLE]",
                 KEGG = "",
                 DOSE = "")
term_list = list(GO_ALL = go_list,
                 GO_SIMPLE = go_list,
                 KEGG = c('kegg'),
                 DOSE = c('HDO', 'HPO', 'NCG', 'DGN'))


for (k in c('GO_ALL', 'GO_SIMPLE', 'KEGG', 'DOSE')) {
# for (k in c('DOSE')) {
  # Dotplot > Each dot is a GO term. The dot size shows the gene count and the color, the p-adjust
  cat("===== DOT PLOTS =====\n")
  for (set in set_list) {
    for (ont in term_list[[k]]) {
      if (!is.null(results_list_of_list[[k]][[set]][[ont]]) && nrow(as.data.frame(results_list_of_list[[k]][[set]][[ont]])) > 0) {
        cat(paste("  >> Plotting dot plot for ", set, " (", ont, ")...\n"))
        # Show top 50 terms
        p <- dotplot(results_list_of_list[[k]][[set]][[ont]],
                     showCategory = 50,
                     title = paste0(first_list[[k]], " Enrichment for ", set, " (", ont, ") ", last_list[[k]]))
        p <- p + theme(axis.text.y = element_text(size = 7, margin = unit(c(1.1,1.1,0,-10), "pt")),
                       plot.margin = unit(c(0.5, 0.5, 0.5, 3), "lines"))

        png(paste0(out_dir, "/PLOTS/", k, "_dotplot_", set, "_", ont, ".png"),
            width = 900,
            height = 1200,
            res = 100)
        print(p)
        dev.off()

      } else {
        cat(paste0("  >> No significantly enriched ", first_list[[k]], " terms were found for ", set, " (", ont, ")\n"))
      }

    }
  }

  # Barplot > Each bar is a GO term. The  size shows the gene count and the color, the p-adjust
  cat("===== BAR PLOTS =====\n")
  for (set in set_list) {
    for (ont in term_list[[k]]) {
      if (!is.null(results_list_of_list[[k]][[set]][[ont]]) && nrow(as.data.frame(results_list_of_list[[k]][[set]][[ont]])) > 0) {
        cat(paste("  >> Plotting bar plot for ", set, " (", ont, ")...\n"))
        # Show top 50 terms
        p <- barplot(results_list_of_list[[k]][[set]][[ont]],
                     showCategory = 50,
                     title = paste0(first_list[[k]], " Enrichment for ", set, " (", ont, ") ", last_list[[k]]))
        p <- p + theme(axis.text.y = element_text(size = 7, margin = unit(c(1.1,1.1,0,-10), "pt")),
                       plot.margin = unit(c(0.5, 0.5, 0.5, 3), "lines"))

        png(paste0(out_dir, "/PLOTS/", k, "_barplot_", set, "_", ont, ".png"),
            width = 900,
            height = 1200,
            res = 100)
        print(p)
        dev.off()

      } else {
        cat(paste0("  >> No significantly enriched ", first_list[[k]], " terms were found for ", set, " (", ont, ")\n"))
      }

    }
  }

  # Enrichment map > Each node is a GO term and the lines describe relationship between them.
  cat("===== ENRICHMENT MAPS =====\n")
  for (set in set_list) {
    for (ont in term_list[[k]]) {
      if (!is.null(results_list_of_list[[k]][[set]][[ont]]) && nrow(as.data.frame(results_list_of_list[[k]][[set]][[ont]])) > 0) {
        cat(paste("  >> Plotting enrichment plot for ", set, " (", ont, ")...\n"))
        data <- pairwise_termsim(results_list_of_list[[k]][[set]][[ont]])
        p <- emapplot(data, showCategory=20)
        p <- p +
          ggtitle(paste0(first_list[[k]], " Enrichment for ", set, " (", ont, ") ", last_list[[k]])) +
          theme(plot.title = element_text(hjust = 0.5))

        png(paste0(out_dir, "/PLOTS/", k, "_emap_", set, "_", ont, ".png"),
            width = 1000,
            height = 1000,
            res = 100)
        print(p)
        dev.off()
      } else {
        cat(paste0("  >> No significantly enriched ", first_list[[k]], " terms were found for ", set, " (", ont, ")\n"))
      }

    }
  }

  # Gene-Concept Network > Shows relationship between genes and terms. Helps identifying central genes.
  cat("===== GENE-CONCEPT NETWORKS =====\n")
  for (set in set_list) {
    for (ont in term_list[[k]]) {
      if (!is.null(results_list_of_list[[k]][[set]][[ont]]) && nrow(as.data.frame(results_list_of_list[[k]][[set]][[ont]])) > 0) {
        cat(paste("  >> Plotting CNET plot for ", set, " (", ont, ")...\n"))
        # Show top 10 terms
        p <- cnetplot(results_list_of_list[[k]][[set]][[ont]],
                      showCategory = 10,
                      foldChange = NULL,
                      color.params = list(edge = TRUE, node = TRUE),
                      color_category = "#0000be",
                      color_item = "#ffc0c0",
                      node_label = "share"
                      )
        p <- p +
          ggtitle(paste0(first_list[[k]], " Enrichment for ", set, " (", ont, ") ", last_list[[k]])) +
          theme(plot.title = element_text(hjust = 0.5))

        png(paste0(out_dir, "/PLOTS/", k, "_cnet_", set, "_", ont, ".png"),
            width = 1000,
            height = 1000,
            res = 100)
        print(p)
        dev.off()
      } else {
        cat(paste0("  >> No significantly enriched ", first_list[[k]], " terms were found for ", set, " (", ont, ")\n"))
      }

    }
  }
}

# ==================================
# ==================================
