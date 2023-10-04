library(tidyverse)
library(latex2exp)

width <- 8.33*1000
height <- 5.79*1000

fname1 <- "full_w1_1.0_w2_5.0_sentence_41785"
fname2 <- "full_w1_2.0_w2_1.0_sentence_41785"

df1 <- read_csv(paste0("results/ner_ss_attraction/", fname1, ".csv") )
df1 <- df1 %>% mutate(problem = "State Attraction", impact = p_att - p_clean) %>% 
  select(seconds, problem, n_exp, w1, w2, d2original, impact, exp_util) %>% 
  mutate(exp_util =  exp_util - w1 - 30*(w2) - 1)
df2 <- read_csv(paste0("results/ner_ss_attraction/", fname2, ".csv") )
df2 <- df2 %>% mutate(problem = "State Attraction", impact = p_att - p_clean) %>% 
  select(seconds, problem,  n_exp, w1, w2, d2original, impact, exp_util) %>% 
  mutate(exp_util =  exp_util - w1 - 30*(w2) - 1)

df3 <- read_csv(paste0("results/ner_ss_repulsion/", fname1, ".csv") )
df3 <- df3 %>% mutate(problem = "State Repulsion", impact = p_att - p_clean) %>% 
  select(seconds, problem, n_exp, w1, w2, d2original, impact, exp_util) %>% 
  mutate(exp_util =  exp_util - w1 - 30*(w2) - 1)
df4 <- read_csv(paste0("results/ner_ss_repulsion/", fname2, ".csv") )
df4 <- df4 %>% mutate(problem = "State Repulsion", impact = p_att - p_clean) %>% 
  select(seconds, problem,  n_exp, w1, w2, d2original, impact, exp_util) %>% 
  mutate(exp_util =  exp_util - w1 - 30*(w2) - 1)

df5 <- read_csv(paste0("results/ner_sd/", fname1, ".csv") )
df5 <- df5 %>% mutate(problem = "Distribution Disruption", impact = `kl-d`) %>% 
  select(seconds, problem,  n_exp, w1, w2, d2original, impact, exp_util) %>% 
  mutate(exp_util =  exp_util - 30*(w2) - 1)
df6 <- read_csv(paste0("results/ner_sd/", fname2, ".csv") )
df6 <- df6 %>% mutate(problem = "Distribution Disruption", impact = `kl-d`) %>% 
  select(seconds, problem,  n_exp, w1, w2, d2original, impact, exp_util) %>% 
  mutate(exp_util =  exp_util - 30*(w2) - 1)

df7 <- read_csv(paste0("results/ner_path_attraction/", fname1, ".csv") )
df7 <- df7 %>% mutate(problem = "Path Attraction", impact = hamming_d2target) %>% 
  select(seconds, problem,  n_exp, w1, w2, d2original, impact, exp_util)  %>% 
  mutate(exp_util =  exp_util - 30*(w1+w2) - 1)
df8 <- read_csv(paste0("results/ner_path_attraction/", fname2, ".csv") )
df8 <- df8 %>% mutate(problem = "Path Attraction", impact = hamming_d2target) %>% 
  select(seconds, problem, n_exp, w1, w2, d2original, impact, exp_util)  %>% 
  mutate(exp_util =  exp_util - 30*(w1+w2) - 1)

df <- df1 %>% bind_rows(df2, df3, df4, df5, df6, df7, df8)

df_sum <- df %>% filter(seconds==9000) %>% group_by(problem, w1, w2) %>% 
  summarise("w1" = mean(w1),
            "w2" = mean(w2),
            "mean_impact" = mean(impact),
            "std_impact" = sd(impact),
            "mean_HD2original" = mean(d2original),
            "std_HD2original" = sd(d2original),
            "mean_EU" = mean(exp_util),
            "std_EU" = sd(exp_util)
  ) %>%  select(-starts_with("std")) 

knitr::kable(df_sum, "latex")

