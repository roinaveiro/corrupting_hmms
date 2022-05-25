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
df1 = read_csv('att_high_unc_ratio.csv')


pdf('att_lowandhigh_unc_ratio.pdf')
fig = ggplot(data = df, aes(x=ratio)) +
   geom_line(aes(y=df$res, colour = 'Low Uncertainty'), linetype = 'solid')+
   geom_ribbon(aes(ymin =df$res_std_minus , ymax = df$res_std_plus), fill = 'grey40', alpha = 0.5) + 
  scale_y_continuous(name =TeX("Probability of $Q_3 =1$" ), breaks=seq(0,1,0.1), limits = c(0,1)) +
  scale_x_continuous(name = TeX("$w_1 /w_2$"), breaks=seq(0,100,10), limits = c(0,55) ) +
  theme(panel.background = element_blank(),panel.border = element_blank(), panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"), text = element_text(size = 22))

#State-Attraction High Uncertainty
fig = fig + geom_line(aes(y=df1$res, colour = 'High Uncertainty'), linetype = 'solid') + 
  geom_ribbon(aes(ymin =df1$res_std_minus , ymax = df1$res_std_plus), fill = 'grey40', alpha = 0.5) +
  theme(axis.line.y = element_line(color = "black"),
        axis.text.y = element_text(color = "black"),
        axis.text.x = element_text(color = "black"),
        legend.text = element_text(size = 18),
        legend.position = 'bottom', legend.title = element_blank()) +
  scale_color_manual(breaks = c('Low Uncertainty', 'High Uncertainty'),
                     values = c('blue','black'))

print(fig)

dev.off()





######################################
####### State Repulsion Plots########
######################################
#State-Repulsion Low Uncertainty
df = read_csv('rep_low_unc_ratio.csv')
df1 = read_csv('rep_high_unc_ratio.csv')

pdf('rep_lowandhigh_unc_ratio.pdf')
fig2 = ggplot(data = df, aes(x=ratio)) +
  geom_line(aes(y=df$res, colour = 'Low Uncertainty'), linetype = 'solid')+
  geom_ribbon(aes(ymin =df$res_std_minus , ymax = df$res_std_plus), fill = 'grey40', alpha = 0.5) + 
  scale_y_continuous(name = TeX("Probability of $Q_3 =2$" ), breaks=seq(0,1,0.1), limits = c(0,1)) +
  scale_x_continuous(name = TeX("$w_1 /w_2$"), breaks=seq(0,100,10), limits = c(0,80)) +
  theme(panel.border = element_blank(), panel.background = element_blank() ,panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"), text = element_text(size = 22))


#State-Repulsion High Uncertainty
fig2 = fig2 + geom_line(aes(y=df1$res, colour = 'High Uncertainty'), linetype = 'solid') + 
  geom_ribbon(aes(ymin =df1$res_std_minus , ymax = df1$res_std_plus), fill = 'grey40', alpha = 0.5) +
  theme(axis.line.y = element_line(color = "black"),
        axis.text.y = element_text(color = "black"),
        axis.text.x = element_text(color = "black"),
        legend.text = element_text(size = 18),
        legend.position = 'bottom', legend.title = element_blank()) +
  scale_color_manual(breaks = c('Low Uncertainty', 'High Uncertainty'),
                     values = c('blue','black'))

print(fig2)

dev.off()




######################################
####### Dist Disruption Plots########
######################################
#Dist Disrupt Low Uncertainty
df = read_csv('dd_low_unc_ratio.csv')
df1 = read_csv('dd_high_unc_ratio.csv')

pdf('dd_lowandhigh_unc_ratio.pdf')
fig3 = ggplot(data = df, aes(x=ratio)) +
  geom_line(aes(y=df$res, colour = 'Low Uncertainty'), linetype = 'solid')+
  geom_ribbon(aes(ymin =df$res_std_minus , ymax = df$res_std_plus), fill = 'grey40', alpha = 0.5) + 
  scale_y_continuous(name =TeX("Kullback-Leibler Divergence" ) , breaks=seq(0,11,1), limits = c(0,11)) +
  scale_x_continuous(name = TeX("$w_1 /w_2$"), breaks=seq(0,100,1), limits = c(0,10)) +
  theme(panel.border = element_blank(), panel.background = element_blank(),panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"), text = element_text(size = 22))

#Dist-Disrupt High Uncertainty
fig3 = fig3 + geom_line(aes(y=df1$res, colour = 'High Uncertainty'), linetype = 'solid') + 
  geom_ribbon(aes(ymin =df1$res_std_minus , ymax = df1$res_std_plus), fill = 'grey40', alpha = 0.5) +
  theme(axis.line.y = element_line(color = "black"),
        axis.text.y = element_text(color = "black"),
        axis.text.x = element_text(color = "black"),
        legend.text = element_text(size = 18),
        legend.position = 'bottom', legend.title = element_blank()) +
  scale_color_manual(breaks = c('Low Uncertainty', 'High Uncertainty'),
                     values = c('blue','black'))

print(fig3)

dev.off()



######################################
####### Path Attract Plots########
######################################
#Path Attract Low Uncertainty
df = read_csv('pd_low_unc_ratio.csv')

pdf('pd_lowandhigh_unc_ratio.pdf')
fig4 =ggplot(data = df, aes(x=ratio)) +
  geom_line(aes(y=df$res, colour = 'Low Uncertainty'), linetype = 'solid') +
  geom_ribbon(aes(ymin =df$res_std_minus , ymax = df$res_std_plus), fill = 'grey40', alpha = 0.5) + 
  scale_y_continuous(name = TeX("Jaccard Index" ), breaks=seq(0,1,0.1), limits = c(0,1)) +
  scale_x_continuous(name = TeX("$w_1 /w_2$"), breaks=seq(0,350,50), limits = c(0,350)) +
  theme(panel.border = element_blank(), panel.background =element_blank(),panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"), text = element_text(size = 22))

#Path Attract High Uncertainty
df1 = read_csv('pd_high_unc_ratio.csv')
fig4 = fig4 + geom_line(aes(y=df1$res, colour = 'High Uncertainty'), linetype = 'solid') + 
  geom_ribbon(aes(ymin =df1$res_std_minus , ymax = df1$res_std_plus), fill = 'grey40', alpha = 0.5) +
  theme(axis.line.y = element_line(color = "black"),
        axis.text.y = element_text(color = "black"),
        axis.text.x = element_text(color = "black"),
        legend.text = element_text(size = 18),
        legend.position = 'bottom', legend.title = element_blank()) +
  scale_color_manual(breaks = c('Low Uncertainty', 'High Uncertainty'),
                     values = c('blue','black'))

print(fig4)

dev.off()

