setwd("~/Research/HMM Poisoning")
#install.packages('latex2exp')

library(tidyverse)
library(mosaic)
library(latex2exp)

######################################
####### State Attraction Plots########
######################################
#State-Attraction Low Uncertainty
df = read_csv('att_low_unc_ratio.csv')
#df$res_std_plus3 =  df$res + 3/2 * (df$res_std_plus - df$res) 
#df$res_std_minus3 =  df$res - 3/2 * (df$res-df$res_std_minus)

pdf('att_low_unc_ratio.pdf')
gf_line(df$res ~ df$ratio, color = 'black' ) %>%
  #gf_line(df$res_std_plus ~ df$ratio, color = 'grey5'  ) %>%
  #gf_line(df$res_std_minus ~ df$ratio, color = 'grey5' ) %>%
  gf_theme(theme_bw())  %>%
  gf_labs( x = TeX("$w_1 /w_2$"),
           y = TeX("Probability of $Q_3 =1$" )) + 
     geom_ribbon(aes(ymin =df$res_std_minus , ymax = df$res_std_plus), fill = 'grey40', alpha = 0.5) + 
     scale_y_continuous(breaks=seq(0,1,0.1), limits = c(0,1)) +
     scale_x_continuous(breaks=seq(0,100,10), limits = c(0,55) ) +
     theme(panel.border = element_blank(), panel.grid.major = element_blank(),
            panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"), text = element_text(size = 22))
dev.off()

#State-Attraction High Uncertainty
df = read_csv('att_high_unc_ratio.csv')
#df$res_std_plus3 =  df$res + 3/2 * (df$res_std_plus - df$res) 
#df$res_std_minus3 =  df$res - 3/2 * (df$res-df$res_std_minus)

pdf('att_high_unc_ratio.pdf')
gf_line(df$res ~ df$ratio, color = 'black' ) %>%
  #gf_line(df$res_std_plus ~ df$ratio, color = 'grey5'  ) %>%
  #gf_line(df$res_std_minus ~ df$ratio, color = 'grey5' ) %>%
  gf_theme(theme_bw())  %>%
  gf_labs( x = TeX("$w_1 /w_2$"),
           y = TeX("Probability of $Q_3 =1$" )) + 
  geom_ribbon(aes(ymin =df$res_std_minus , ymax = df$res_std_plus), fill = 'grey40', alpha = 0.5) + 
  scale_y_continuous(breaks=seq(0,1,0.1), limits = c(0,1)) +
  scale_x_continuous(breaks=seq(0,100,10), limits = c(0,55)) +
  theme(panel.border = element_blank(), panel.grid.major = element_blank(),
                panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"), text = element_text(size = 22))
dev.off()


######################################
####### State Repulsion Plots########
######################################
#State-Repulsion Low Uncertainty
df = read_csv('rep_low_unc_ratio.csv')
#df$res_std_plus3 =  df$res + 3/2 * (df$res_std_plus - df$res) 
#df$res_std_minus3 =  df$res - 3/2 * (df$res-df$res_std_minus)

pdf('rep_low_unc_ratio.pdf')
gf_line(df$res ~ df$ratio, color = 'black' ) %>%
  #gf_line(df$res_std_plus ~ df$ratio, color = 'grey5'  ) %>%
  #gf_line(df$res_std_minus ~ df$ratio, color = 'grey5' ) %>%
  gf_theme(theme_bw())  %>%
  gf_labs( x = TeX("$w_1 /w_2$"),
           y = TeX("Probability of $Q_3 =2$" )) + 
  geom_ribbon(aes(ymin =df$res_std_minus , ymax = df$res_std_plus), fill = 'grey40', alpha = 0.5) + 
  scale_y_continuous(breaks=seq(0,1,0.1), limits = c(0,1)) +
  scale_x_continuous(breaks=seq(0,100,10), limits = c(0,80)) +
  theme(panel.border = element_blank(), panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"), text = element_text(size = 22))
dev.off()

#State-Repulsion High Uncertainty
df = read_csv('rep_high_unc_ratio.csv')
#df$res_std_plus3 =  df$res + 3/2 * (df$res_std_plus - df$res) 
#df$res_std_minus3 =  df$res - 3/2 * (df$res-df$res_std_minus)

pdf('rep_high_unc_ratio.pdf')
gf_line(df$res ~ df$ratio, color = 'black' ) %>%
  #gf_line(df$res_std_plus ~ df$ratio, color = 'grey5'  ) %>%
  #gf_line(df$res_std_minus ~ df$ratio, color = 'grey5' ) %>%
  gf_theme(theme_bw())  %>%
  gf_labs( x = TeX("$w_1 /w_2$"),
           y = TeX("Probability of $Q_3 =2$" )) + 
  geom_ribbon(aes(ymin =df$res_std_minus , ymax = df$res_std_plus), fill = 'grey40', alpha = 0.5) + 
  scale_y_continuous(breaks=seq(0,1,0.1), limits = c(0,1)) +
  scale_x_continuous(breaks=seq(0,100,10), limits = c(0,80)) +
  theme(panel.border = element_blank(), panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"), text = element_text(size = 22))
dev.off()

######################################
####### Dist Disruption Plots########
######################################
#Dist Disrupt Low Uncertainty
df = read_csv('dd_low_unc_ratio.csv')
#df$res_std_plus3 =  df$res + 3/2 * (df$res_std_plus - df$res) 
#df$res_std_minus3 =  df$res - 3/2 * (df$res-df$res_std_minus)

pdf('dd_low_unc_ratio.pdf')
gf_line(df$res ~ df$ratio, color = 'black' ) %>%
  #gf_line(df$res_std_plus ~ df$ratio, color = 'grey5'  ) %>%
  #gf_line(df$res_std_minus ~ df$ratio, color = 'grey5' ) %>%
  gf_theme(theme_bw())  %>%
  gf_labs( x = TeX("$w_1 /w_2$"),
           y = TeX("Kullback-Leibler Divergence" )) + 
  geom_ribbon(aes(ymin =df$res_std_minus , ymax = df$res_std_plus), fill = 'grey40', alpha = 0.5) + 
  scale_y_continuous(breaks=seq(0,11,1), limits = c(0,11)) +
  scale_x_continuous(breaks=seq(0,100,1), limits = c(0,10)) +
  theme(panel.border = element_blank(), panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"), text = element_text(size = 22))
dev.off()

#Dist-Disrupt High Uncertainty
df = read_csv('dd_high_unc_ratio.csv')
#df$res_std_plus3 =  df$res + 3/2 * (df$res_std_plus - df$res) 
#df$res_std_minus3 =  df$res - 3/2 * (df$res-df$res_std_minus)

pdf('dd_high_unc_ratio.pdf')
gf_line(df$res ~ df$ratio, color = 'black' ) %>%
  #gf_line(df$res_std_plus ~ df$ratio, color = 'grey5'  ) %>%
  #gf_line(df$res_std_minus ~ df$ratio, color = 'grey5' ) %>%
  gf_theme(theme_bw())  %>%
  gf_labs( x = TeX("$w_1 /w_2$"),
           y = TeX("Kullback-Leibler Divergence" )) + 
  geom_ribbon(aes(ymin =df$res_std_minus , ymax = df$res_std_plus), fill = 'grey40', alpha = 0.5) + 
  scale_y_continuous(breaks=seq(0,11,1), limits = c(0,11)) +
  scale_x_continuous(breaks=seq(0,100,1), limits = c(0,10)) +
  theme(panel.border = element_blank(), panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"), text = element_text(size = 22))
dev.off()

######################################
####### Path Attract Plots########
######################################
#Path Attract Low Uncertainty
df = read_csv('pd_low_unc_ratio.csv')
#df$res_std_plus3 =  df$res + 3/2 * (df$res_std_plus - df$res) 
#df$res_std_minus3 =  df$res - 3/2 * (df$res-df$res_std_minus)

pdf('pd_low_unc_ratio.pdf')
gf_line(df$res ~ df$ratio, color = 'black' ) %>%
  #gf_line(df$res_std_plus ~ df$ratio, color = 'grey5'  ) %>%
  #gf_line(df$res_std_minus ~ df$ratio, color = 'grey5' ) %>%
  gf_theme(theme_bw())  %>%
  gf_labs( x = TeX("$w_1 /w_2$"),
           y = TeX("Jaccard Index" )) + 
  geom_ribbon(aes(ymin =df$res_std_minus , ymax = df$res_std_plus), fill = 'grey40', alpha = 0.5) + 
  scale_y_continuous(breaks=seq(0,1,0.1), limits = c(0,1)) +
  scale_x_continuous(breaks=seq(0,350,50), limits = c(0,350)) +
  theme(panel.border = element_blank(), panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"), text = element_text(size = 22))
dev.off()

#Path Attract High Uncertainty
df = read_csv('pd_high_unc_ratio.csv')
#df$res_std_plus3 =  df$res + 3/2 * (df$res_std_plus - df$res) 
#df$res_std_minus3 =  df$res - 3/2 * (df$res-df$res_std_minus)

pdf('pd_high_unc_ratio.pdf')
gf_line(df$res ~ df$ratio, color = 'black' ) %>%
  #gf_line(df$res_std_plus ~ df$ratio, color = 'grey5'  ) %>%
  #gf_line(df$res_std_minus ~ df$ratio, color = 'grey5' ) %>%
  gf_theme(theme_bw())  %>%
  gf_labs( x = TeX("$w_1 /w_2$"),
           y = TeX("Jaccard Index" )) + 
  geom_ribbon(aes(ymin =df$res_std_minus , ymax = df$res_std_plus), fill = 'grey40', alpha = 0.5) + 
  scale_y_continuous(breaks=seq(0,1,0.1), limits = c(0,1)) +
  scale_x_continuous(breaks=seq(0,350,50), limits = c(0,350)) +
  theme(panel.border = element_blank(), panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"), text = element_text(size = 22))
dev.off()

