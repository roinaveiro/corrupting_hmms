#setwd("C:/Users/Chema Camacho/Desktop/ICMAT/HMM/experiments/experiment_17/v5_19_05_2022_right/")
#install.packages('latex2exp')

library(tidyverse)
library(mosaic)
library(latex2exp)

######################################
####### State Attraction Plots########
######################################
#State-Attraction Low Uncertainty
df = read_csv('att_low_unc_box.csv')

pdf('att_low_unc_box.pdf')
gf_boxplot(res~as.factor(diff_n_comp), data =df ) %>%
  gf_theme(theme_bw())  %>%
  gf_labs( x = TeX("No. Poisoned Observations"),
           y = TeX("Probability of $Q_3 =1$" )) + 
  scale_y_continuous(breaks=seq(0,1,0.1), limits = c(0,1)) +
  theme(panel.border = element_blank(), panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"), text = element_text(size = 22))
dev.off()

#State-Attraction high Uncertainty
df = read_csv('att_high_unc_box.csv')

pdf('att_high_unc_box.pdf')
gf_boxplot(res~as.factor(diff_n_comp), data =df ) %>%
  gf_theme(theme_bw())  %>%
  gf_labs( x = TeX("No. Poisoned Observations"),
           y = TeX("Probability of $Q_3 =1$" )) + 
  scale_y_continuous(breaks=seq(0,1,0.1), limits = c(0,1)) +
  theme(panel.border = element_blank(), panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"), text = element_text(size = 22))
dev.off()


######################################
####### State Repulsion Plots########
######################################
#State-Repulsion Low Uncertainty
df = read_csv('rep_low_unc_box.csv')

pdf('rep_low_unc_box.pdf')
gf_boxplot(res~as.factor(diff_n_comp), data =df ) %>%
  gf_theme(theme_bw())  %>%
  gf_labs( x = TeX("No. Poisoned Observations"),
           y = TeX("Probability of $Q_3 =2$" )) + 
  scale_y_continuous(breaks=seq(0,1,0.1), limits = c(0,1)) +
  theme(panel.border = element_blank(), panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"), text = element_text(size = 22))
dev.off()

#State-Attraction high Uncertainty
df = read_csv('rep_high_unc_box.csv')

pdf('rep_high_unc_box.pdf')
gf_boxplot(res~as.factor(diff_n_comp), data =df ) %>%
  gf_theme(theme_bw())  %>%
  gf_labs( x = TeX("No. Poisoned Observations"),
           y = TeX("Probability of $Q_3 =2$" )) + 
  scale_y_continuous(breaks=seq(0,1,0.1), limits = c(0,1)) +
  theme(panel.border = element_blank(), panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"), text = element_text(size = 22))
dev.off()



######################################
####### Dist Disruption Plots########
######################################
#Dist Disrupt Low Uncertainty
df = read_csv('dd_low_unc_box.csv')
#df$res_std_plus3 =  df$res + 3/2 * (df$res_std_plus - df$res) 
#df$res_std_minus3 =  df$res - 3/2 * (df$res-df$res_std_minus)

pdf('dd_low_unc_box.pdf')
gf_boxplot(res~as.factor(diff_n_comp), data =df ) %>%
  gf_theme(theme_bw())  %>%
  gf_labs( x = TeX("No. Poisoned Observations"),
           y = TeX("Kullback-Leibler Divergence" )) + 
  scale_y_continuous(breaks=seq(0,11,1), limits = c(0,11)) +
  theme(panel.border = element_blank(), panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"), text = element_text(size = 22))        
dev.off()

#Dist-Disrupt High Uncertainty
df = read_csv('dd_high_unc_box.csv')
#df$res_std_plus3 =  df$res + 3/2 * (df$res_std_plus - df$res) 
#df$res_std_minus3 =  df$res - 3/2 * (df$res-df$res_std_minus)

pdf('dd_high_unc_box.pdf')
gf_boxplot(res~as.factor(diff_n_comp), data =df ) %>%
  gf_theme(theme_bw())  %>%
  gf_labs( x = TeX("No. Poisoned Observations"),
           y = TeX("Kullback-Leibler Divergence" )) + 
  scale_y_continuous(breaks=seq(0,11,1), limits = c(0,11)) +
  theme(panel.border = element_blank(), panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"), text = element_text(size = 22))        
dev.off()

######################################
####### Path Attract Plots########
######################################
#Path Attract Low Uncertainty
df = read_csv('pd_low_unc_box.csv')

pdf('pd_low_unc_box.pdf')
gf_boxplot(res~as.factor(diff_n_comp), data =df ) %>%
  gf_theme(theme_bw())  %>%
  gf_labs( x = TeX("No. Poisoned Observations"),
           y = TeX("Normalized Hamming Distance" )) + 
  scale_y_continuous(breaks=seq(0,1,0.1), limits = c(0,1)) +
  theme(panel.border = element_blank(), panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"), text = element_text(size = 22))        
dev.off()

#Path Attract High Uncertainty
df = read_csv('pd_high_unc_box.csv')

pdf('pd_high_unc_box.pdf')
gf_boxplot(res~as.factor(diff_n_comp), data =df ) %>%
  gf_theme(theme_bw())  %>%
  gf_labs( x = TeX("No. Poisoned Observations"),
           y = TeX("Normalized Hamming Distance" )) + 
  scale_y_continuous(breaks=seq(0,1,0.1), limits = c(0,1)) +
  theme(panel.border = element_blank(), panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"), text = element_text(size = 22))        
dev.off()

