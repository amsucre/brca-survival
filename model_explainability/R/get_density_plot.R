# PREP ENVIRONMENT

library(ggplot2)
library(dplyr)
library(tidyr)

# ==========
# ==========

# MAIN FUNCTION
draw_density_plot <- function(df, 
                              stat, 
                              stat_name,
                              set_levels,
                              set_labels,
                              model_levels,
                              model_labels,
                              ref,
                              model_colors=NULL,
                              keep=NULL) {
  
  # --- Transform DF to long format
  df_long <- df %>%
    pivot_longer(
      cols = -MODEL, 
      names_to = c("SET", "FOLD"), 
      names_pattern = paste0(stat, "_([A-Z]{2})_(\\d+)$"),
      values_to = 'VALUE'
    ) %>%
    
    {
      if (!is.null(keep)) {
        filter(., SET == keep)
      } else {
        .
      }
    } %>%  # if only one set to keep, filter the DF
    
    mutate(
      SET = factor(SET,
                   levels = set_levels,
                   labels = set_labels),
      
      VALUE = as.numeric(VALUE),  # just in case
      
      MODEL = factor(MODEL,
                     levels = model_levels,
                     labels = model_labels)
    )
    
  # --- Plot density plot
  
  plot_density <- ggplot(df_long, 
                         aes(x = VALUE, fill = MODEL, color = MODEL)) +
    
    # geom_density to plot curve
    geom_density(alpha = 0.3, linewidth = 1) +
    
    # facet to split the 3 sets (Train/Val/Test)
    facet_wrap(~ SET, scales = "free_y", ncol = 1) + 
    
    # Ref line
    geom_vline(xintercept = ref, linetype = "dashed", color = "black", alpha = 1) +
    
    # Labels
    labs(
      title = paste(stat_name, "density plot"),
      x = stat_name,
      y = "Density",
      fill = "Model",
      color = "Model"
    ) +
    
    # Style
    scale_x_continuous(limits = c(0, 1.0)) +
    theme_minimal() +
    theme(
      plot.title = element_text(face = "bold", hjust = 0.5),
      strip.text = element_text(size = 12, face = "bold"), 
      legend.position = "bottom"
    )
  
  # Define colors manually
  if (!is.null(model_colors)) {
    plot_density <- plot_density + 
      scale_fill_manual(values = model_colors) +
      scale_color_manual(values = model_colors)
  }
  
  return(plot_density)
}

# ==========
# ==========

# DEFINE INPUT DATA
home_dir <- ''  # results dir

target_columns <- c("CI_TR", "CI_VA", "CI_TE")

model_files_omi <- c(paste0(home_dir, 'OMI-SINGLE/', '5_fold_results_all.tsv'))
model_names_omi <- c('SNP', 'MIR', 'CNV', 'RNA', 'CLIN')

model_files_int <- c(paste0(home_dir, 'INT/', '5_fold_results_OMI_4Q_raw.tsv'),
                     paste0(home_dir, 'INT/', '5_fold_results_OMI_CLIN_4Q_raw.tsv'),
                     paste0(home_dir, 'INT/', '5_fold_results_OMI_RESNET_4Q_raw.tsv'),
                     paste0(home_dir, 'INT/', '5_fold_results_CLIN_RESNET_4Q_raw.tsv'),
                     paste0(home_dir, 'INT/', '5_fold_results_OMI_CLIN_RESNET_4Q_raw.tsv'),
                     paste0(home_dir, 'INT/', '5_fold_results_OMI_EARLY.tsv'),
                     paste0(home_dir, 'INT/', '5_fold_results_OMI_CLIN_EARLY.tsv'),
                     paste0(home_dir, 'INT/', '5_fold_results_OMI_CLIN_RESNET_MEAN_EARLY.tsv')
)

model_files_img_soa <- c(paste0(home_dir, 'IMG/', 'summary_final_amil.csv'),
                         paste0(home_dir, 'SOA/', 'summary_final_porpoise_mmf.csv'),
                         paste0(home_dir, 'SOA/', 'summary_final_mcat.csv'),
                         paste0(home_dir, 'SOA/', 'summary_final_mgct.csv')
)
model_names_img_soa <- c('RESNET', 'PORPOISE', 'MCAT', 'MGCT')

# GET C-INDEX VALUES
values_list <- list()

# # ALL OMICS IN SAME FILE
for (fpath in model_files_omi) {
  # Read file
  df <- read.delim(fpath, header = TRUE, sep = "\t")
  
  for (model_name in model_names_omi) {
    # Get DF subset
    df_omi <- df %>% filter(OMI == model_name)
    
    # Init stat holder
    model_values <- c(MODEL = model_name)
    
    # Get c-index values
    for (col_name in target_columns) {
      # Check if column exists
      if (col_name %in% colnames(df_omi)) {
        # Get column values
        values <- df_omi[[col_name]]
        
        # Rename the values vector elements to include the column name (e.g., CI_TR_AVG)
        names(values) <- paste(col_name, 1:5, sep = "_")
        
        # Append the model name and values 
        model_values <- c(model_values, values)
      } else {
        warning(paste("Column", col_name, "not found in file:", model_name))
      }
    }
    
    # Store values (temp)
    values_list[[model_name]] <- model_values
  }
}

# # INTEGRATION MODELS
for (fpath in model_files_int) {
  # Get model name
  model_name <- gsub("^5_fold_results_|_4Q_raw|\\.tsv$", "", basename(fpath))
  print(model_name)
  
  # Read file
  df <- read.delim(fpath, header = TRUE, sep = "\t")
  
  # Init stat holder
  model_values <- c(MODEL = model_name)
  
  # Get c-index values
  for (col_name in target_columns) {
    # Check if column exists
    if (col_name %in% colnames(df)) {
      # Get column values
      values <- df[[col_name]]

      # Rename the values vector elements to include the column name (e.g., CI_TR_AVG)
      names(values) <- paste(col_name, 1:5, sep = "_")
      
      # Append the model name and values 
      model_values <- c(model_values, values)
    } else {
      warning(paste("Column", col_name, "not found in file:", model_name))
    }
  }
  
  # Store values (temp)
  values_list[[model_name]] <- model_values
}

# # IMAGE AND SOA MODELS
i = 1
for (fpath in model_files_img_soa) {
  # Get model name
  model_name <- model_names_img_soa[i]
  print(model_name)
  
  # Read file
  df <- read.delim(fpath, header = TRUE, sep = ",")
  df <- df %>%
    filter(!X %in% c("mean", "std")) %>%
    select(train_cindex, val_cindex, test_cindex)
  
  # print(colnames(df))
  colnames(df) <- target_columns 
  
  # Init stat holder
  model_values <- c(MODEL = model_name)
  
  # Get c-index values
  for (col_name in target_columns) {
    # Check if column exists
    if (col_name %in% colnames(df)) {
      # Get column values
      values <- df[[col_name]]
      
      # Rename the values vector elements to include the column name (e.g., CI_TR_AVG)
      names(values) <- paste(col_name, 1:5, sep = "_")
      
      # Append the model name and values 
      model_values <- c(model_values, values)
    } else {
      warning(paste("Column", col_name, "not found in file:", model_name))
    }
  }
  
  # Store stats (temp)
  values_list[[model_name]] <- model_values
  
  i = i+1
}

# STORE RESULTS
ci_df <- dplyr::bind_rows(values_list)

output_file = paste0(home_dir, 'CINDEX_ALL_MODELS.tsv')
write.table(
  ci_df,
  file = output_file,
  sep = "\t",
  row.names = FALSE,
  quote = FALSE,
  col.names = TRUE
)

ci_df <- read.delim(output_file, header = TRUE, sep = "\t")

# ============
# ============

# GET IBS VALUES
ibs_path <- ''  # IBS results path
ibs_df <- read.delim(ibs_path, header = TRUE, sep = "\t")
ibs_df <- ibs_df[, c('MODEL', 'IBS_TE_1', 'IBS_TE_2', 'IBS_TE_3', 'IBS_TE_4', 'IBS_TE_5')]

# ============
# ============

# PREP DATA FOR DENSITY PLOTS + PLOT DENSITY PLOTS
ci_model_lev <- c("MCAT", "PORPOISE", "MGCT",
                  "CLIN", "SNP", "RNA", "CNV", "MIR", "RESNET",  
                  "OMI", "OMI_CLIN", "OMI_RESNET", "CLIN_RESNET", "OMI_CLIN_RESNET", 
                  "OMI_EARLY", "OMI_CLIN_EARLY", "OMI_CLIN_RESNET_MEAN_EARLY" 
)
ci_model_lbl <- c("SOA: MCAT", "SOA: PORPOISE", "SOA: MGCT",
                  "CLIN", "SNV", "RNA", "CNV", "MIRNA", "IMAGE",  
                  "OMI (L)", "OMI+CLIN (L)", "OMI+IMG (L)", "CLIN+IMG (L)", "OMI+CLIN+IMG (L)",
                  "OMI (E)", "OMI+CLIN (E)", "OMI+CLIN+IMG (E)"
)
model_colors <-  c("SOA: MCAT" = "#BF0F0F", "SOA: PORPOISE" = "#EC1313", "SOA: MGCT" = "#F36868",
                   "CLIN" = "#120FBF", "SNV" = "#413EEF", "RNA" = "#BEBDFA", "CNV" = "#0C6D97", "MIRNA" ="#13ABEC", "IMAGE" = "#68C9F3",
                   "OMI (L)" = "#0B4205", "OMI+CLIN (L)" = "#136D09", "OMI+IMG (L)" = "#21BF0F", "CLIN+IMG (L)" = "#4FEF3E", "OMI+CLIN+IMG (L)"= "#9CF692",
                   "OMI (E)" = "#CBA406", "OMI+CLIN (E)" = "#F7C708", "OMI+CLIN+IMG (E)" = "#FADB61")

ci_plot <- draw_density_plot(df=ci_df,
                             stat='CI', 
                             stat_name='C-Index',
                             set_levels=c("TR", "VA", "TE"),
                             set_labels=c("Training", "Validation", "Test"),
                             model_levels=ci_model_lev,
                             model_labels=ci_model_lbl, 
                             ref=0.5,
                             model_colors=model_colors)

ci_TE_plot <- draw_density_plot(df=ci_df,
                             stat='CI', 
                             stat_name='C-Index (Test)',
                             set_levels=c("TR", "VA", "TE"),
                             set_labels=c("Training", "Validation", "Test"),
                             model_levels=ci_model_lev,
                             model_labels=ci_model_lbl, 
                             ref=0.5,
                             model_colors=model_colors,
                             keep='TE')

ibs_model_lev <- c("MCAT", "PORPOISE", "MGCT",
                    "CLIN", "SNP", "EXP", "CNV", "MIR", "IMG",  
                    "OM", "OM_CL", "OM_IM", "CL_IM", "OM_CL_IM", 
                    "OM_E", "OM_CL_E", "OM_CL_IM_E" 
)
ibs_model_lbl <- c("SOA: MCAT", "SOA: PORPOISE", "SOA: MGCT",
                   "CLIN", "SNV", "RNA", "CNV", "MIRNA", "IMAGE",  
                   "OMI (L)", "OMI+CLIN (L)", "OMI+IMG (L)", "CLIN+IMG (L)", "OMI+CLIN+IMG (L)",
                   "OMI (E)", "OMI+CLIN (E)", "OMI+CLIN+IMG (E)"
)

ibs_plot <- draw_density_plot(df=ibs_df,
                             stat='IBS', 
                             stat_name='IBS (Test)',
                             set_levels=c("TE"),
                             set_labels=c("Test"),
                             model_levels=ibs_model_lev,
                             model_labels=ibs_model_lbl, 
                             ref=0,
                             model_colors=model_colors)

# SAVE PLOTS
plot_dir = ''  # output dir

ggsave(
  filename = paste0(plot_dir, 'density_CI_3_sets.png'),
  plot = ci_plot, 
  width = 12,
  height = 20,
  units = "cm",
  dpi = 300 
)

ggsave(
  filename = paste0(plot_dir, 'density_CI.png'),
  plot = ci_TE_plot, 
  width = 12,
  height = 12,
  units = "cm",
  dpi = 300 
)

ggsave(
  filename = paste0(plot_dir, 'density_IBS.png'),
  plot = ibs_plot, 
  width = 12,
  height = 12,
  units = "cm",
  dpi = 300 
)

