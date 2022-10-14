library(tidyverse)
library(latex2exp)
library(greekLetters)
dpi <- 300
width <- 8.33
height <- 5.79

### directory for data
data <- read_csv("results/full_data.csv")  

XX = 20
QQ = 20
TT = 20

## uncomment for colors
#branded_colors <- c(
#  "APS-A" = "#edae49",
#  "APS-B"  = "#66a182",
#  "R&S-A"   = "#00798c",
#  "R&S-B"    = "#d1495b",
#  "RME"   = "#2e4057"
#)



comb_names <- as_labeller( c(
  "A" = "Design Point 1",
  "B" = "Design Point 2",
  "C" = "Design Point 3",
  "D" = "Design Point 4")
)

rep_el <- theme_minimal() %+replace%
  theme(  strip.text  = element_text(face = "bold", size = 10, vjust=1))

subtitle = paste0("|X| = ", as.character(XX), ", ",
                  "|Q| = ", as.character(QQ), ", ",
                  "|T| = ", as.character(TT) )

data %>% filter(time %% 10 == 0) %>% 
  mutate(
  combination = case_when(
        rho == 0.95 & k == 10000 ~ "A",
        rho == 0.95 & k == 100 ~ "B",
        rho == 0.75 & k == 10000 ~ "C",
        rho == 0.75 & k == 100 ~ "D"
  ) #atr:st.attraction, rep:st.repulsion, dd:distribution disruption, pd: path attraction
) %>% filter(attacker == "att",
             `|Q|` == QQ, `|X|` == XX, `|T|` == TT) %>%  ## write best performing MCTS and SA
  filter( (combination == "A" & solver %in% c("APS_1", "APS_500", "RME", "SA_E", "MCTS_C") ) |
          (combination == "B" & solver %in% c("APS_1", "APS_500", "RME", "SA_A", "MCTS_C") ) |
          (combination == "C" & solver %in% c("APS_1", "APS_500", "RME", "SA_B", "MCTS_E") ) |
          (combination == "D" & solver %in% c("APS_1", "APS_500", "RME", "SA_A", "MCTS_E") ) 
          ) %>%  
  mutate(solver=replace(solver, 
                        solver %in% c("SA_A", "SA_B", "SA_C", "SA_D", "SA_E", "SA_F"), "R&S-B")) %>%
  mutate(solver=replace(solver, 
                        solver %in% c("MCTS_A", "MCTS_B", "MCTS_C", "MCTS_D", "MCTS_E", "MCTS_F"), "R&S-A")) %>%
  mutate(solver=replace(solver, 
                        solver %in% c("APS_1"), "APS-A")) %>%
  mutate(solver=replace(solver, 
                        solver %in% c("APS_500"), "APS-B")) %>%
  mutate(Algorithm=solver) %>%
  ggplot(aes(x=time, y=EU, color=Algorithm)) + geom_line(size=0.5) +
  geom_errorbar(aes(ymin=Lower, ymax=Upper), size=0.5, alpha=0.25,
                position=position_dodge(0.05)) +
  facet_wrap(combination~., labeller = comb_names) +
  rep_el + 
  labs(x = expression(bold("Allotted Computation Time")),
       y = expression(bold("Expected Utility"))) +
  theme(axis.text.x=element_text(angle=-90, hjust=0, vjust=1)) +
  theme(plot.title=element_text(size=15, hjust=0.5, face="bold", vjust=-1)) +
  theme(plot.subtitle=element_text(size=12, hjust=0.5, vjust=-1)) +
  theme(text = element_text(size=12)) +
  theme(legend.text = element_text( size = 10))+
  theme(legend.title = element_text(face = "bold" ,size = 11))

ggsave(filename = "experiment_plots_r/exp_2_attr.png",   #directory to save
       device = "png", 
       dpi = dpi, width = width, height = height)


         