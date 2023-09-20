library(tidyverse)
library(latex2exp)

fname <- "full_w1_1.0_w2_5.0_sentence_41785"
df <- read_csv(paste0("results/", fname, ".csv") )

df_sum <- df %>% select(-c(original_phrase, attacked_phrase)) %>% 
  group_by(seconds) %>% 
  reframe("w1" = mean(w1),
            "w2" = mean(w2),
            "|X|" = mean(n_obs),
            "|T|" = mean(T),
            "|Q|" = mean(n_hidden),
            "mean_EU" = mean(exp_util),
            "std_EU" = sd(exp_util),
            "mean_HD2target" = mean(hamming_d2target),
            "std_HD2target" = sd(hamming_d2target),
            "mean_HD2original" = mean(d2original),
            "std_HD2original" = sd(d2original),
            "mean_originalacc" = mean(original_acc),
            "std_originalacc" = sd(original_acc),
            "mean_attacc" = mean(attacked_acc),
            "std_attacc" = sd(attacked_acc))

df_pivot <- df_sum %>%  pivot_longer(
  cols = starts_with("mean_") | starts_with("std_"),
  names_to = c(".value", "name"),
  names_sep = "_",
  values_to = c("Mean", "Std")
) %>%
  mutate(name = case_when(
    name == "EU" ~ "Expected Utility",
    name == "HD2target" ~ "Hamming Distance to Target",
    name == "HD2original" ~ "Hamming Distance to Original",
    name == "originalacc" ~ "Original Accuracy",
    name == "attacc" ~ "Attacked Accuracy",
    TRUE ~ name
  ))

subt <- TeX(paste("$w_1 =", as.character(unique(df_pivot$w1)),"$,", 
                  "$w_2 =", as.character(unique(df_pivot$w2)),"$,", 
                  "$|T| =", as.character(unique(df_pivot$`|T|`)),"$,", 
                  "$|Q| =", as.character(unique(df_pivot$`|Q|`)),"$,",
                  "$|X| =", as.character(unique(df_pivot$`|X|`)),"$"))


p <- df_pivot %>% ggplot(aes(x = seconds, y = mean)) + 
  geom_point() + geom_errorbar(aes(ymin=mean-std, 
                                   ymax=mean+std), width =0.9) + 
  facet_wrap(name~., scales="free") + theme_minimal() +
  labs(title = "Attack performance metrics", subtitle =subt, x = 
         "Running time (seconds)", y = "Mean plus/minus one standard deviation") 

ggsave(p, path = "fig.png", dpi=400)
