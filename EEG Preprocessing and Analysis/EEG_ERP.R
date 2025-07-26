# ===================================================================
# R SCRIPT FOR ERP SIGNIFICANCE TESTS (LATENCY & AMPLITUDE)
# ===================================================================

# 1. SETUP
# -------------------------------------------------------------------
# Install packages if you haven't already
# install.packages(c("tidyverse", "lme4", "lmerTest", "emmeans", "ggrepel", "rstatix", "afex"))
library(tidyverse)
library(lme4)
library(lmerTest)
library(emmeans)
library(ggrepel)
library(ggplot2)
library(rstatix) # Still useful for data manipulation
library(afex)    # For efficient ANOVA (replacing BayesFactor)

# --- Configuration ---
input_csv_path <- ""
output_dir <- ""
dir.create(output_dir, showWarnings = FALSE)

set.seed(123)


  # 2. LOAD AND PREPARE DATA
  # -------------------------------------------------------------------
if (!file.exists(input_csv_path)) {
  stop(paste("Error: Input CSV file not found at:", input_csv_path,
             "\nPlease re-run the Python script to generate it."))
}
metrics_df <- read_csv(input_csv_path, show_col_types = FALSE)

metrics_df <- metrics_df %>%
  mutate(
    group = factor(group, levels = c("non-ADHD", "ADHD")),
    condition = as.factor(condition),
    participant = as.factor(participant)
  )

# --- 2B. PARTICIPANT EXCLUSION --- # if needed
participants_to_exclude <- c()

cat(sprintf("\nData rows before exclusion: %d\n", nrow(metrics_df)))

metrics_df_filtered <- metrics_df %>%
  filter(!participant %in% participants_to_exclude)

cat(sprintf("Excluding participants: %s\n", paste(participants_to_exclude, collapse = ", ")))
cat(sprintf("Data rows after exclusion: %d\n\n", nrow(metrics_df_filtered)))


  # ===================================================================
# PRIMARY ANALYSIS: LINEAR MIXED-EFFECTS MODELS (lmer)
# ===================================================================
cat("\n\n--- Running Primary Statistical Models (lmer) on Filtered Data ---\n")

run_lmer_model <- function(data, metric) {
  if (metric %in% names(data)) {
    formula <- as.formula(paste(metric, "~ group * condition + (1 | participant)"))
    model <- lmer(formula, data = data %>% filter(!is.na(.data[[metric]])))
    cat(paste("\n---", toupper(metric), "LMER Model Summary ---\n"))
    print(summary(model))
  }
}

metrics_to_analyze <- c("gfp_peak_latency_ms", "gfp_peak_amplitude_uv",
                        "oz_peak_latency_ms", "oz_peak_amplitude_uv",
                        "fz_peak_latency_ms", "fz_peak_amplitude_uv",
                        "fz_neg_200_400_peak_latency_ms", "fz_neg_200_400_peak_amplitude_uv",
                        "oz_pos_100_200_peak_latency_ms", "oz_pos_100_200_peak_amplitude_uv")

for (metric in metrics_to_analyze) {
  run_lmer_model(metrics_df_filtered, metric)
}

  # ===================================================================
# COMPLEMENTARY ANALYSIS: REGULAR ANOVA (using afex)
# - Provides traditional ANOVA results with sphericity correction.
# ===================================================================
cat("\n\n--- Running Regular ANOVAs (using afex::aov_ez) on Filtered Data ---\n")
cat("Note: This uses the afex package for ANOVA with sphericity correction.\n")

run_afex_anova <- function(data, metric, run_posthocs = FALSE) {
  if (!(metric %in% names(data))) {
    cat(sprintf("\nMetric '%s' not found. Skipping afex ANOVA.\n", metric))
    return()
  }
  
  model_data <- data %>% filter(!is.na(.data[[metric]]))
  
  if(nrow(model_data) == 0) {
    cat(sprintf("Skipping afex ANOVA for %s: no complete data after NA removal.\n", metric))
    return()
  }
  
  cat(paste("\n\n======================================================\n"))
  cat(sprintf("       ANOVA (afex::aov_ez): %s\n", toupper(metric)))
  cat(paste("======================================================\n"))
  
  tryCatch({
    afex_result <- aov_ez(
      id = "participant", dv = metric, data = model_data,
      within = "condition", between = "group",
      anova_table = list(es = "pes")
    )
    print(afex_result)
    
    # --- Post-hoc tests (controlled by the 'run_posthocs' argument) ---
    if (run_posthocs) {
      cat(sprintf("\n--- Post-hoc tests for %s ---\n", toupper(metric)))
      if ("group" %in% names(model_data) && "condition" %in% names(model_data)) {
        cat("\n--- Post-hoc tests for interaction ---\n")
        post_hoc_interaction <- emmeans(afex_result, ~ condition | group)
        print(pairs(post_hoc_interaction))
      }
      if ("group" %in% names(model_data) && length(unique(model_data$group)) > 1) {
        cat("\n--- Post-hoc tests for main effect of group ---\n")
        post_hoc_group <- emmeans(afex_result, ~ group)
        print(pairs(post_hoc_group))
      }
      if ("condition" %in% names(model_data) && length(unique(model_data$condition)) > 1) {
        cat("\n--- Post-hoc tests for main effect of condition ---\n")
        post_hoc_condition <- emmeans(afex_result, ~ condition)
        print(pairs(post_hoc_condition))
      }
    } else {
      cat(sprintf("\nPost-hoc tests were not run for %s.\n", toupper(metric)))
    }
    
  }, error = function(e) {
    cat(sprintf("Error running afex ANOVA for %s: %s\n", metric, e$message))
  })
}

# Run the afex ANOVA for all our key metrics
for (metric in metrics_to_analyze) {
  
  # ==========================================================
  # ✅ EDIT THIS VARIABLE TO CONTROL ALL POST-HOC TESTS
  # ==========================================================
  #
  # `FALSE` = All post-hocs are OFF.
  # `TRUE`  = All post-hocs are ON.
  # `grepl("^oz_", metric)` = ON for specific variables only.
  #
  run_posthocs_for_this_metric <- FALSE
  #
  # ==========================================================
  
  run_afex_anova(
    data = metrics_df_filtered, 
    metric = metric, 
    run_posthocs = run_posthocs_for_this_metric
  )
}

  # ===================================================================
# VISUALIZATION
# ===================================================================
cat("\n\n=======================================================\n")
cat("               GENERATING PLOTS \n")
cat("=======================================================\n\n")

# Generic plotting function to avoid repetition
create_labeled_plot <- function(data, metric, title_prefix) {
  if (!(metric %in% names(data))) { return(NULL) }
  
  y_label <- ifelse(grepl("amplitude", metric), "Peak Amplitude (µV)", "Peak Latency (ms)")
  
  p <- ggplot(data %>% filter(!is.na(.data[[metric]])),
              aes(x = group, y = .data[[metric]], fill = group)) +
    geom_boxplot(alpha = 0.4, outlier.shape = NA) +
    geom_jitter(width = 0.2, alpha = 0.1, height = 0) +
    geom_text_repel(
      aes(label = participant), size = 2.5, box.padding = 0.1, point.padding = 0.1,
      max.overlaps = Inf, segment.color = 'grey70', segment.alpha = 0.7
    ) +
    labs(
      title = paste(title_prefix, y_label, "Distribution"),
      subtitle = "Using filtered data. Each label represents one participant in one condition.",
      x = "Group", y = y_label
    ) +
    theme_bw(base_size = 14) +
    guides(fill = "none") +
    scale_fill_manual(values = c("non-ADHD" = "royalblue", "ADHD" = "darkorange")) +
    facet_wrap(~condition)
  
  print(p)
  ggsave(file.path(output_dir, paste0("plot_", metric, "_all_labels.png")), p, width = 14, height = 9)
}

# Create plots for all metrics
for (metric in metrics_to_analyze) {
  title_prefix <- toupper(str_extract(metric, "^[a-z]+"))
  create_labeled_plot(metrics_df_filtered, metric, title_prefix)
}


cat("\n\n✅ R script finished. All models and plots have been generated.\n")
