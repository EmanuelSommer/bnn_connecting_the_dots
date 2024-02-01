library(tidyverse)
# load the data +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
aggregated_data <- read_csv("")

main_hyperparams <- c(
  "data",
  "activation",
  "hidden_structure",
  "n_chains",
  "n_samples",
  "keep_warmup",
  "sampler",
  "prior_sd",
  "prior_dist"
)

numeric_result_cols <- c(
  "rmse_linear",
  "rmse_rf",
  "n_bad_chains",
  "n_good_chains",
  "rmse_good_chains",
  "rmse_good_chains_100",
  "acc_90hpdi",
  "acc_90hpdi_100",
  "runtime"
)
# aggregate the results wrt replications +++++++++++++++++++++++++++++++++++
aggregated_data <- aggregated_data |>
  group_by(across(all_of(main_hyperparams))) |>
  # average all numeric_result_cols and add a count of replicates
  summarise(
    across(all_of(numeric_result_cols), mean),
    n_replicates = n(),
    .groups = "keep"
  ) |>
  ungroup()

# runtime analysis +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# use histogram + kde plots one for each dataset
aggregated_data |>
  ggplot(aes(x = runtime)) +
  geom_density(aes(y = ..density..), color = "red") +
  # rugplot
  geom_rug() +
  facet_wrap(~data, scales = "free") +
  labs(
    x = "Runtime (minutes)",
    y = "Density",
    title = "Runtime analysis"
  ) +
  theme_bw()

aggregated_data |>
  filter(data == "airfoil.data") |>
  mutate(across(all_of(main_hyperparams), as.character)) |>
  pivot_longer(
    cols = c(
      "activation", "hidden_structure", "sampler", "prior_sd", "prior_dist"
    ),
    names_to = "hyperparam",
    values_to = "value"
  ) |>
  ggplot(aes(x = runtime, y = value)) +
  geom_boxplot(aes(color = data)) +
  scale_color_brewer(palette = "Set1") +
  scale_fill_brewer(palette = "Set1") +
  facet_wrap(~hyperparam, scales = "free") +
  labs(
    x = "Runtime (minutes)",
    y = "Hyperparameter value",
    title = "Runtime analysis"
  ) +
  theme_bw()

# Bad chains analysis ++++++++++++++++++++++++++++++++++++++++++++++++++++++

aggregated_data |>
  ggplot(aes(x = n_bad_chains)) +
  geom_density(aes(y = ..density..), color = "red") +
  # rugplot
  geom_rug() +
  facet_wrap(~data, scales = "free") +
  labs(
    x = "Number of bad chains",
    y = "Density",
    title = "Bad chains analysis"
  ) +
  theme_bw()

aggregated_data |>
  mutate(
    hidden_structure = factor(
      hidden_structure,
      levels = c("2", "8", "16-16", "64", "32-32-32")
    )
  ) |>
  group_by(activation, hidden_structure) |>
  summarise(
    n_bad_chains = mean(n_bad_chains),
    n_good_chains = mean(n_good_chains),
    .groups = "keep"
  ) |>
  ggplot(aes(x = hidden_structure, y = n_good_chains, group = activation)) +
  geom_line(aes(color = activation)) +
  geom_point(aes(color = activation)) +
  labs(
    x = "Hidden structure",
    y = "Number of BLM chains",
    color = "Activation",
    title = "Better than LM comparison across architectures"
  ) +
  scale_color_brewer(palette = "Set1") +
  scale_y_continuous(breaks = seq(12)) +
  # facet_wrap(~prior_sd, scales = "free") +
  theme_bw()


aggregated_data |>
  # all columns to strings
  mutate(across(all_of(main_hyperparams), as.character)) |>
  pivot_longer(
    cols = c(
      "activation", "hidden_structure", "sampler", "prior_sd", "prior_dist"
    ),
    names_to = "hyperparam",
    values_to = "value"
  ) |>
  ggplot(aes(x = n_bad_chains, y = value)) +
  geom_violin(aes(color = data, fill = data)) +
  scale_color_brewer(palette = "Set1") +
  scale_fill_brewer(palette = "Set1") +
  facet_wrap(~hyperparam, scales = "free") +
  labs(
    x = "Number of bad chains",
    y = "Hyperparameter value",
    title = "Bad chains analysis"
  ) +
  theme_bw()

# RMSE analysis ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
aggregated_data |>
  ggplot(aes(x = rmse_good_chains)) +
  geom_density(aes(y = ..density..), color = "blue") +
  # rugplot
  geom_rug(aes(color = hidden_structure), size = 1, length = unit(0.1, "npc")) +
  geom_vline(
    aes(xintercept = rmse_linear), color = "black", linetype = "dashed"
    ) +
  geom_vline(aes(xintercept = rmse_rf), color = "gray", linetype = "dashed") +
  facet_wrap(~data, scales = "free") +
  labs(
    x = "RMSE linear",
    y = "Density",
    title = "RMSE analysis",
    subtitle = paste(
      "The baseline RMSEs are shown as",
      "dashed lines (black = LM, gray = RF)"
    )
  ) +
  theme_bw()

# now do the same but with the 100 samples
aggregated_data |>
  ggplot(aes(x = rmse_good_chains_100)) +
  geom_density(aes(y = ..density..), color = "blue") +
  # rugplot
  geom_rug(aes(color = hidden_structure)) +
  geom_vline(
    aes(xintercept = rmse_linear), color = "black", linetype = "dashed"
  ) +
  geom_vline(aes(xintercept = rmse_rf), color = "gray", linetype = "dashed") +
  facet_wrap(~data, scales = "free") +
  labs(
    x = "RMSE linear",
    y = "Density",
    title = "RMSE analysis (Only 100 samples)",
    subtitle = paste(
      "The baseline RMSEs are shown as",
      "dashed lines (black = LM, gray = RF)"
    )
  ) +
  theme_bw()

  # now the scatter plots
aggregated_data |>
  mutate(across(all_of(main_hyperparams), as.character)) |>
  pivot_longer(
    cols = c(
      "activation", "hidden_structure", "sampler", "prior_sd", "prior_dist"
    ),
    names_to = "hyperparam",
    values_to = "value"
  ) |>
  ggplot(aes(x = rmse_good_chains, y = value)) +
  geom_boxplot(aes(color = data)) +
  scale_color_brewer(palette = "Set1") +
  scale_fill_brewer(palette = "Set1") +
  facet_wrap(~hyperparam, scales = "free") +
  labs(
    x = "RMSE linear",
    y = "Hyperparameter value",
    title = "RMSE analysis"
  ) +
  theme_bw()

aggregated_data |>
  mutate(across(all_of(main_hyperparams), as.character)) |>
  pivot_longer(
    cols = c(
      "activation", "hidden_structure", "sampler", "prior_sd", "prior_dist"
      ),
    names_to = "hyperparam",
    values_to = "value"
  ) |>
  ggplot(aes(x = rmse_good_chains_100, y = value)) +
  geom_violin(aes(color = data, fill = data)) +
  scale_color_brewer(palette = "Set1") +
  scale_fill_brewer(palette = "Set1") +
  facet_wrap(~hyperparam, scales = "free") +
  labs(
    x = "RMSE linear",
    y = "Hyperparameter value",
    title = "RMSE analysis (Only 100 samples)"
  ) +
  theme_bw()


# Calibration analysis +++++++++++++++++++++++++++++++++++++++++++++++++++++
# same as RMSE basically
aggregated_data |>
  ggplot(aes(x = acc_90hpdi)) +
  geom_density(aes(y = ..density..), color = "blue") +
  # rugplot
  geom_rug(aes(color = hidden_structure)) +
  geom_vline(xintercept = 0.9, color = "black", linetype = "dashed") +
  facet_wrap(~data, scales = "free") +
  labs(
    x = "Calibration linear",
    y = "Density",
    title = "Accuracy of 90% HPDI",
    subtitle = paste(
      "The baseline calibrations are shown as",
      "dashed lines (black = LM, gray = RF)"
    )
  ) +
  theme_bw()

# now do the same but with the 100 samples
aggregated_data |>
  ggplot(aes(x = acc_90hpdi_100)) +
  geom_density(aes(y = ..density..), color = "blue") +
  # rugplot
  geom_rug(aes(color = hidden_structure)) +
  geom_vline(xintercept = 0.9, color = "black", linetype = "dashed") +
  facet_wrap(~data, scales = "free") +
  labs(
    x = "Calibration linear",
    y = "Density",
    title = "Accuracy of 90% HPDI, (Only 100 samples)",
    subtitle = paste(
      "The baseline calibrations are shown as dashed lines",
      "(black = LM, gray = RF)"
    )
  ) +
  theme_bw()

# now the scatter plots
aggregated_data |>
  mutate(across(all_of(main_hyperparams), as.character)) |>
  pivot_longer(
    cols = c(
      "activation", "hidden_structure",
      "sampler", "prior_sd", "prior_dist"
    ),
    names_to = "hyperparam",
    values_to = "value"
  ) |>
  ggplot(aes(x = acc_90hpdi, y = value)) +
  geom_violin(aes(color = data, fill = data)) +
  geom_vline(xintercept = 0.9, color = "black", linetype = "dashed") +
  scale_color_brewer(palette = "Set1") +
  scale_fill_brewer(palette = "Set1") +
  facet_wrap(~hyperparam, scales = "free") +
  labs(
    x = "Calibration linear",
    y = "Hyperparameter value",
    title = "Accuracy of 90% HPDI",
  ) +
  theme_bw()

aggregated_data |>
  mutate(across(all_of(main_hyperparams), as.character)) |>
  pivot_longer(
    cols = c(
      "activation", "hidden_structure", "sampler", "prior_sd", "prior_dist"
    ),
    names_to = "hyperparam",
    values_to = "value"
  ) |>
  ggplot(aes(x = acc_90hpdi_100, y = value)) +
  geom_boxplot(aes(color = data)) +
  geom_vline(xintercept = 0.9, color = "black", linetype = "dashed") +
  scale_color_brewer(palette = "Set1") +
  scale_fill_brewer(palette = "Set1") +
  facet_wrap(~hyperparam, scales = "free") +
  labs(
    x = "Calibration linear",
    y = "Hyperparameter value",
    title = "Accuracy of 90% HPDI (Only 100 samples)",
  ) +
  theme_bw()


# look at the best models in terms of RMSE
aggregated_data |>
  arrange(data, rmse_good_chains_100) |>
  group_by(data) |>
  slice(1:3) |>
  select(
    data, activation, hidden_structure, sampler, prior_sd, prior_dist,
    rmse_good_chains, rmse_good_chains_100, acc_90hpdi, rmse_rf, n_good_chains
  )

aggregated_data |>
  group_by(activation) |>
  summarise(
    isna_rmse = mean(is.na(rmse_good_chains_100)),
    mean_good_chains = mean(n_good_chains),
    n_good_chains_smaller_half = mean(n_good_chains < 6)
  )

aggregated_data |>
  filter(n_good_chains > 1) |>
  filter(rmse_good_chains < rmse_linear) |>
  group_by(activation, hidden_structure, data) |>
  summarise(
    n = n(),
    runtimes = mean(runtime),
    rmse = mean(rmse_good_chains),
  ) |>
  arrange(data, hidden_structure, activation) |>
  View()


aggregated_data |>
  filter(
    hidden_structure == "16-16"
  )
