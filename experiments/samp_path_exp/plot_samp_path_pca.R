RESULT_DIR <- 'results/'
EXP <- 'fcn_bnns/'
DATE <- '2024-01-16-13-57-52/'
path <- paste0(RESULT_DIR, EXP, DATE, "pca/pca.csv")

library(tidyverse)
df <- read_csv(path)
df_aggr <- df |>
  mutate(
    block = as.integer(str_remove(block, "[A-Za-z]")),
    type = if_else(str_detect(type, "W"), "Weights", "Biases")
  ) |>
  group_by(dataset, type, block) |>
  summarise(
    mean_var_expl = mean(explained_variance),
    mean = mean(pcomp),
    sd = sd(pcomp),
    .groups = "keep"
  ) |>
  ungroup() |>
  mutate(dataset = paste0(dataset, " (", round(mean_var_expl, 2), ")"))

df_aggr
write_csv(df_aggr, paste0(RESULT_DIR, EXP, DATE, "pca/pca_aggr.csv"))

avg_explained_var <- df_aggr |>
  group_by(dataset) |>
  summarise(
    mean = mean(mean_var_expl),
    .groups = "keep"
  )
avg_explained_var

plot_aggr <- df_aggr |>
  ggplot() +
  geom_point(
    aes(x = block, y = mean, color = type),
    size = 2,
    position = position_dodge(0.5)
  ) +
  geom_line(
    aes(x = block, y = mean, color = type),
    position = position_dodge(0.5)
  ) +
  geom_errorbar(
    aes(x = block, ymin = mean - sd, ymax = mean + sd, color = type),
    width = 0.1,
    position = position_dodge(0.5)
  ) +
  scale_x_continuous(breaks = seq(1, 7, 1)) +
  scale_color_brewer(palette = "Set1") +
  facet_wrap(~dataset, nrow = 1, ncol = 3) +
  labs(
    x = "Layer", y = "Average absolute PCA loadings (k=3)", color = "",
    caption = paste(
      "Average explained variance by the first 3",
      "principal components in brackets."
    )
  ) +
  theme_bw() +
  theme(
    strip.background = element_rect(fill = "white", colour = "white"),
    legend.position = "bottom",
    text = element_text(size = 14),
    axis.text.x = element_text(size = 12),
    axis.text.y = element_text(size = 12),
    legend.text = element_text(size = 12)
  )

plot_aggr

library(here)
ggsave(
  here(RESULT_DIR, EXP, DATE, "pca", "pca_plot.png"),
  plot_aggr,
  height = 5,
  width = 12
)
