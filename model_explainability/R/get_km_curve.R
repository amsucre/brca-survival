# PREP ENVIRONMENT
library(survival)
library(survminer)
library(dplyr)

# ==========

# MAIN FUNCTION
get_kaplan_curve_age <- function(df, thr, set) {
  # Define groups
  surv_df <- df %>%
    mutate(
      GROUP = ifelse(AGE_DIAGNOSIS > thr, 
                     paste(">", thr, " years"),
                     paste("<=", thr, "years"))
    )    
  
  # Create survival object
  surv_object <- Surv(time = surv_df$MONTHS, 
                      event = 1 - surv_df$IS_CENSORED)
  
  # Fit curve based on age group
  km_fit <- surv_fit(surv_object ~ GROUP, data = surv_df)
  
  # Test if curves are significantly different
  log_rank_test <- survdiff(surv_object ~ GROUP, data = surv_df)

  # Create plot
  km_plot <- ggsurvplot(
    km_fit, 
    data = surv_df, 
    
    # Style
    title = paste0("Kaplan-Meier Curve by Age (Set:", set, ")"),
    xlab = "Survival Months", 
    ylab = "Survival Probability",
    legend.title = "Age Group",
    
    # Extra data, curves, colors
    risk.table = TRUE,
    cumevents = TRUE, 
    pval = TRUE,
    conf.int = FALSE,
    risk.table.col = "strata",
    cumevents.col = "strata",
    linetype = "strata",
    ggtheme = theme_bw(),
    palette = c("#EC1313", "#413EEF"),
    
    # Highlight censored points
    censor = TRUE,
    censor.shape = "|",
    
    # Legend
    legend.labs = c(paste0("Age > ", thr), 
                    paste0("Age <= ", thr))
  )
  
  km_plot$plot <- km_plot$plot + 
    theme(plot.title = element_text(hjust = 0.5))
  
  return (list(plot=km_plot, test=log_rank_test))
  
}

get_kaplan_curve <- function(surv_df, set) {
  # Create survival object
  surv_object <- Surv(time = surv_df$MONTHS, 
                      event = 1 - surv_df$IS_CENSORED)
  
  # Fit curve 
  km_fit <- surv_fit(surv_object ~ 1, data = surv_df)
  
  # Create plot
  km_plot <- ggsurvplot(
    km_fit, 
    data = surv_df, 
    
    # Style
    title = paste0("Kaplan-Meier Curve (Set:", set, ")"),
    xlab = "Survival Months", 
    ylab = "Survival Probability",

    # Extra data, curves, colors
    risk.table = TRUE,
    cumevents = TRUE, 
    conf.int = FALSE,
    risk.table.col = "strata",
    cumevents.col = "strata",
    linetype = "strata",
    ggtheme = theme_bw(),
    legend = 'none',
    
    # Highlight censored points
    censor = TRUE,
    censor.shape = "|",
  )
  
  km_plot$plot <- km_plot$plot + 
    theme(plot.title = element_text(hjust = 0.5))
  
  return (list(plot=km_plot))
  
}

# ==========

# PREP DATA
in_dir <- ""  # Raw data dir
f_all <- paste0(in_dir, 'BRCA_CLIN_all.csv')
f_train <- paste0(in_dir, 'BRCA_CLIN_train.csv')
f_test <- paste0(in_dir, 'BRCA_CLIN_test.csv')

df_all <- read.delim(f_all, header = TRUE, sep = "\t")
df_train <- read.delim(f_train, header = TRUE, sep = "\t")
df_test <- read.delim(f_test, header = TRUE, sep = "\t")

age_thr <- 59  # MEDIAN (BASED ON COHORT)

# SURVIVAL ANALYSIS (BY AGE)
result_train <- get_kaplan_curve_age(df_train, age_thr, 'Train')
result_test <- get_kaplan_curve_age(df_test, age_thr, 'Test')


# SURVIVAL ANALYSIS (OVERALL)
result_all <- get_kaplan_curve(df_all, 'Full Cohort')
result_test <- get_kaplan_curve(df_test, 'Test')

print(result_all$plot)
print(result_test$plot)


# SAVE PLOTS
plot_dir = '' # OUTPUT DIR

ggsave(
  filename = paste0(plot_dir, 'km_age_TEST.png'),
  plot = survminer::arrange_ggsurvplots(
    x = list(result_test$plot), 
    print = FALSE,
    ncol=1
  ),
  width = 12,
  height = 15,
  units = "cm",
  dpi = 300 
)

ggsave(
  filename = paste0(plot_dir, 'km_age_TRAIN.png'),
  plot = survminer::arrange_ggsurvplots(
    x = list(result_train$plot), 
    print = FALSE,
    ncol=1
  ),
  width = 12,
  height =15,
  units = "cm",
  dpi = 300 
)

ggsave(
  filename = paste0(plot_dir, 'km_all_TEST.png'),
  plot = survminer::arrange_ggsurvplots(
    x = list(result_test$plot), 
    print = FALSE,
    ncol=1
  ),
  width = 12,
  height = 15,
  units = "cm",
  dpi = 300 
)

ggsave(
  filename = paste0(plot_dir, 'km_all_COHORT.png'),
  plot = survminer::arrange_ggsurvplots(
    x = list(result_all$plot), 
    print = FALSE,
    ncol=1
  ),
  width = 12,
  height =15,
  units = "cm",
  dpi = 300 
)
