library(tidyverse)
library(latex2exp)

width <- 8.33*1000
height <- 5.79*1000


fname2 <- "full_w1_2.0_w2_1.0_sentence_41785"
fname1 <- "full_w1_1.0_w2_5.0_sentence_41785"
df1 <- read_csv(paste0("results/ner_sd/", fname1, ".csv") )
df1 <- df1 %>% mutate(exp_util =  exp_util - 30*(w2) - 1)
df2 <- read_csv(paste0("results/ner_sd/", fname2, ".csv") )
df2 <- df2 %>% mutate(exp_util =  exp_util - 30*(w2) - 1)


df_sum1 <- df1 %>% select(-c(original_phrase, attacked_phrase)) %>% 
  group_by(seconds) %>% 
  summarise("w1" = mean(w1),
            "w2" = mean(w2),
            "|X|" = mean(n_obs),
            "|T|" = mean(T),
            "|Q|" = mean(n_hidden),
            "mean_EU" = mean(exp_util),
            "std_EU" = sd(exp_util),
            "mean_kl-d" = mean(`kl-d`),
            "std_kl-d" = sd(`kl-d`),
            "mean_HD2original" = mean(d2original),
            "std_HD2original" = sd(d2original)
           ) %>% mutate(comb = "1")

df_sum2 <- df2 %>% select(-c(original_phrase, attacked_phrase)) %>% 
  group_by(seconds) %>% 
  summarise("w1" = mean(w1),
            "w2" = mean(w2),
            "|X|" = mean(n_obs),
            "|T|" = mean(T),
            "|Q|" = mean(n_hidden),
            "mean_EU" = mean(exp_util),
            "std_EU" = sd(exp_util),
            "mean_kl-d" = mean(`kl-d`),
            "std_kl-d" = sd(`kl-d`),
            "mean_HD2original" = mean(d2original),
            "std_HD2original" = sd(d2original)
  ) %>% mutate(comb = "2")



df <- df_sum1 %>% bind_rows(df_sum2) %>% select(seconds, w1, w2, `|X|`, 
                                                `|T|`, `|Q|`, mean_EU,
                                                std_EU, comb) 


df$comb <- as.factor(df$comb)
levels(df$comb) <- c("1" = TeX("$w_1 = 1, w_2 = 5$"), "2" =  TeX("$w_1 = 2, w_2 = 1$"))


subt <- TeX(paste( 
  "$|T| =", as.character(30),"$,", 
  "$|Q| =", as.character(unique(df$`|Q|`)),"$,",
  "$|X| =", as.character(unique(df$`|X|`)),"$"))

p <- df %>% ggplot(aes(x = seconds, y = mean_EU)) + 
  geom_point() + geom_errorbar(aes(ymin=mean_EU-std_EU, 
                                   ymax=mean_EU+std_EU), width =0.9) + 
  facet_wrap(comb~., labeller = label_parsed) +
  theme_minimal() +
  #labs(title = "Attack performance. Distribution Disruption.", subtitle =subt, x = 
  #       "Running time (seconds)", y = "Expected Utility")  +
  theme(text = element_text(size=30)) + theme(panel.margin = unit(2, "lines"))

png( paste0("dist_disr_EU", ".png"), width = width , height = height,
     type = "quartz", pointsize = 15, res = 800)
print(p)
dev.off()


if(FALSE){

  #####
  df_pivot <- df_sum %>%  pivot_longer(
    cols = starts_with("mean_") | starts_with("std_"),
    names_to = c(".value", "name"),
    names_sep = "_",
    values_to = c("Mean", "Std")
  ) %>%
    mutate(name = case_when(
      name == "EU" ~ "Expected Utility",
      name == "KL-D" ~ "KL Divergence",
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
    labs(title = "Attack performance metrics. Distribution Disruption.", subtitle =subt, x = 
           "Running time (seconds)", y = "Mean plus/minus one standard deviation") 
  
  png( paste0(fname, ".png"), width = 6800 , height = 5648, units = "px", 
       type = "quartz", pointsize = 15, res = 800)
  print(p)
  dev.off()
  
}
