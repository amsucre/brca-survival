# PREP ENVIRONMENT

library(ggplot2)
library(dplyr)
library(tidyr)

# ==========
# ==========

# MAIN FUNCTION
draw_violin_plot <- function(df, 
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
  
  plot_violin <- ggplot(df_long, 
                         aes(x = MODEL, y = VALUE, fill = MODEL, color = MODEL)) +
    
    # geom_violin to plot curve
    geom_violin(alpha = 0.3, linewidth = 0.8) +
    
    # geom_jitter to plot individual points
    geom_jitter(width = 0.1, size = 0.6, alpha = 0.5, color = "black") +
    
    # facet to split the 3 sets (Train/Val/Test) - side by side
    facet_wrap(~ SET, scales = "free_y", nrow = 1) + 
    
    # Ref line
    geom_hline(yintercept = ref, linetype = "dashed", color = "black", alpha = 1) +
    
    # Violins in vertical axis
    coord_flip() +
    
    # Labels
    labs(
      title = paste(stat_name, "violin plot"),
      y = stat_name,
      x = "Model",
      fill = "Model",
      color = "Model"
    ) +
    
    # Style
    scale_y_continuous(limits = c(0, 1.0)) +
    theme_minimal() +
    theme(
      plot.title = element_text(face = "bold", hjust = 0.5),
      strip.text = element_text(size = 12, face = "bold"), 
      legend.position = "none"
    ) +
    
    # Swap Y axis sort order (IND models on top)
    scale_x_discrete(limits = rev) 
  
  # Define colors manually
  if (!is.null(model_colors)) {
    plot_violin <- plot_violin + 
      scale_fill_manual(values = model_colors, limits = model_labels) +
      scale_color_manual(values = model_colors, limits = model_labels)
  }
  
  return(plot_violin)
}

# ==========
# ==========

# DEFINE INPUT DATA
home_dir <- '' # INPUT DIR
ci_file = paste0(home_dir, 'CINDEX_ALL_MODELS.tsv') # file generated for density plot
ci_df <- read.delim(ci_file, header = TRUE, sep = "\t")

ibs_path <- paste0(home_dir, 'IBS_ALL_5fold.tsv')
ibs_df <- read.delim(ibs_path, header = TRUE, sep = "\t")
ibs_df <- ibs_df[, c('MODEL', 'IBS_TE_1', 'IBS_TE_2', 'IBS_TE_3', 'IBS_TE_4', 'IBS_TE_5')]

# ============
# ============

# PREP DATA FOR VIOLIN PLOTS + PLOT VIOLIN PLOTS
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

ci_violin_plot <- draw_violin_plot(df=ci_df,
                             stat='CI', 
                             stat_name='C-Index',
                             set_levels=c("TR", "VA", "TE"),
                             set_labels=c("Training", "Validation", "Test"),
                             model_levels=ci_model_lev,
                             model_labels=ci_model_lbl, 
                             ref=0.5,
                             model_colors=model_colors)

ci_TE_violin_plot <- draw_violin_plot(df=ci_df,
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

ibs_violin_plot <- draw_violin_plot(df=ibs_df,
                              stat='IBS', 
                              stat_name='IBS (Test)',
                              set_levels=c("TE"),
                              set_labels=c("Test"),
                              model_levels=ibs_model_lev,
                              model_labels=ibs_model_lbl, 
                              ref=0,
                              model_colors=model_colors)

# SAVE PLOTS
plot_dir = ''  # out dir

ggsave(
  filename = paste0(plot_dir, 'violin_CI_3_sets.png'),
  plot = ci_violin_plot, 
  width = 20,
  height = 10,
  units = "cm",
  dpi = 300 
)

ggsave(
  filename = paste0(plot_dir, 'violin_CI.png'),
  plot = ci_TE_violin_plot, 
  width = 12,
  height = 10,
  units = "cm",
  dpi = 300 
)

ggsave(
  filename = paste0(plot_dir, 'violin_IBS.png'),
  plot = ibs_violin_plot, 
  width = 12,
  height = 10,
  units = "cm",
  dpi = 300 
)



