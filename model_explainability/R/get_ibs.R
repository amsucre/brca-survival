# INSTALL PACKAGES
install.packages("survival")
install.packages("SurvMetrics")

# LOAD PACKAGES 
library(survival)
library(SurvMetrics)

# ==================================

# DEF MAIN FUNCTION

get_ibs <- function(path_real, path_pred, thr_list, model_name) {
  # Read input data
  df_real <- read.delim(path_real, header = TRUE)
  df_pred <- read.delim(path_pred, header = TRUE)
  
  # Merge Dfs (Only to ensure same samples are included, in the same order)
  df_merged <- merge(df_real, df_pred, by = "SAMPLE")
  
  # Get Surv object  (Needed for PEC) >> FOR CENSORING WE NEED 1 EVENT, 0 ALIVE
  surv_object <- with(df_merged, Surv(MONTHS, 1 - IS_CENSORED))
  
  # Prep prediction data
  pred_matrix <- as.matrix(df_merged[, c("Q1", "Q2", "Q3", "Q4")])
  
  # Calculate IBS
  ibs_value <- IBS(
    object = surv_object,
    sp_matrix = pred_matrix,
    IBSrange = thr_list
  )
  
  # Show results
  cat("IBS for ", model_name, ": ", ibs_value, "\n")

  return(ibs_value)
}

# ===================

# ANALYSIS

dir <- '' # IBS dir
time_thr <- c(20.155, 37.87, 76.335, 245.1)  # define based on cohort (survival bins thresholds)
path_real <- paste0(dir, "IN/real_survival_TEST.tsv")

### INDIVIDUAL MODELS (5-FOLD)
ibs_mean <- list()
ibs_sd <- list()
ibs_ci_lower <- list()
ibs_ci_upper <- list()
ibs_folds_list <- list()

for (omi in c('SNP', 'EXP', 'CNV', 'MIR', 'CLIN', 'IMG')) {
  ibs_folds <- numeric(5)
  for (i in 0:4) {
    path_pred <- paste0(dir, "IN/IND_EACH/", omi, "/pred_surv_cum_", omi, "_", i, "_TEST.tsv")
    ibs_folds[i+1] <- get_ibs(path_real, path_pred, time_thr, omi)
  }
  
  ibs_folds_list[[omi]] <- ibs_folds
  
  ibs_mean[[omi]] <- mean(ibs_folds)
  ibs_sd[[omi]] <- sd(ibs_folds)
  
  ci_results <- get_conf_int(ibs_folds)
  ibs_ci_lower[[omi]] <- ci_results['CI_LOWER'] 
  ibs_ci_upper[[omi]] <- ci_results['CI_UPPER'] 
}

folds_df <- as.data.frame(do.call(rbind, ibs_folds_list))
colnames(folds_df) <- paste0("IBS_", 1:5)

ibs_df_5fold <- data.frame(
  MODEL = names(ibs_mean),
  IBS_MEAN = unlist(ibs_mean),
  IBS_SD = unlist(ibs_sd),
  IBS_CI_LOWER = unlist(ibs_ci_lower),
  IBS_CI_UPPER = unlist(ibs_ci_upper)
)

ibs_df_5fold <- cbind(ibs_df_5fold, folds_df)

write.table(
  ibs_df_5fold,
  file = paste0(dir, "OUT/IBS_IND_5fold.tsv"),
  sep = "\t",
  row.names = FALSE,
  quote = FALSE 
)

### INTEGRATION MODELS (5-FOLD)
ibs_int_mean <- list()
ibs_int_sd <- list()
ibs_int_ci_lower <- list()
ibs_int_ci_upper <- list()
ibs_folds_list <- list()

for (model in c('OM', 'OM_CL', 'OM_IM', 'CL_IM', 'OM_CL_IM', 'OM_E', 'OM_CL_E', 'OM_CL_IM_E')) {
  ibs_int_folds <- numeric(5)
  for (i in 0:4) {
    path_pred <- paste0(dir, "IN/INT_EACH/", model, "/pred_surv_cum_", model, "_", i, "_TEST.tsv")
    ibs_int_folds[i+1] <- get_ibs(path_real, path_pred, time_thr, model)
  }
  
  ibs_folds_list[[model]] <- ibs_int_folds
  
  ibs_int_mean[[model]] <- mean(ibs_int_folds)
  ibs_int_sd[[model]] <- sd(ibs_int_folds)
  
  ci_results <- get_conf_int(ibs_int_folds)
  ibs_int_ci_lower[[model]] <- ci_results['CI_LOWER']
  ibs_int_ci_upper[[model]] <- ci_results['CI_UPPER']
}

folds_df <- as.data.frame(do.call(rbind, ibs_folds_list))
colnames(folds_df) <- paste0("IBS_", 1:5)
  
ibs_int_df_5fold <- data.frame(
  MODEL = names(ibs_int_mean),
  IBS_MEAN = unlist(ibs_int_mean),
  IBS_SD = unlist(ibs_int_sd),
  IBS_CI_LOWER = unlist(ibs_int_ci_lower),
  IBS_CI_UPPER = unlist(ibs_int_ci_upper)
)

ibs_int_df_5fold <- cbind(ibs_int_df_5fold, folds_df)
  
write.table(
  ibs_int_df_5fold,
  file = paste0(dir, "OUT/IBS_INT_5fold.tsv"),
  sep = "\t",
  row.names = FALSE,
  quote = FALSE 
)

### SOA MODELS (5-FOLD)
ibs_soa_mean <- list()
ibs_soa_sd <- list()
ibs_soa_ci_lower <- list()
ibs_soa_ci_upper <- list()
ibs_folds_list <- list()

for (model in c('PORPOISE', 'MGCT', 'MCAT')) {
  ibs_soa_folds <- numeric(5)
  for (i in 0:4) {
    path_pred <- paste0(dir, "IN/SOA_EACH/", model, "/pred_surv_cum_", model, "_", i, "_TEST.tsv")
    if (file.exists(path_pred)) {
      ibs_soa_folds[i+1] <- get_ibs(path_real, path_pred, time_thr, model)
    }
  }
  
  ibs_folds_list[[model]] <- ibs_soa_folds
  
  ibs_soa_mean[[model]] <- mean(ibs_soa_folds)
  ibs_soa_sd[[model]] <- sd(ibs_soa_folds)
  
  ci_results <- get_conf_int(ibs_soa_folds)
  ibs_soa_ci_lower[[model]] <- ci_results['CI_LOWER'] 
  ibs_soa_ci_upper[[model]] <- ci_results['CI_UPPER'] 
}

folds_df <- as.data.frame(do.call(rbind, ibs_folds_list))
colnames(folds_df) <- paste0("IBS_", 1:5)

ibs_soa_df_5fold <- data.frame(
  MODEL = names(ibs_soa_mean),
  IBS_MEAN = unlist(ibs_soa_mean),
  IBS_SD = unlist(ibs_soa_sd),
  IBS_CI_LOWER = unlist(ibs_soa_ci_lower),
  IBS_CI_UPPER = unlist(ibs_soa_ci_upper)
)

ibs_soa_df_5fold <- cbind(ibs_soa_df_5fold, folds_df)

write.table(
  ibs_soa_df_5fold,
  file = paste0(dir, "OUT/IBS_SOA_5fold.tsv"),
  sep = "\t",
  row.names = FALSE,
  quote = FALSE 
)