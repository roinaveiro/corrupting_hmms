library(tidyverse)
library(latex2exp)
library(greekLetters)
dpi <- 300
width <- 8.33
height <- 5.79


data <- read_csv("results/full_data.csv")

XX = 20
QQ = 20
TT = 20

# Uncomment for colors
#branded_colors <- c(
#  "MCTS-A" = "#edae49",
#  "MCTS-B"  = "#66a182",
#  "MCTS-C"   = "#00798c",
#  "MCTS-D"    = "#d1495b",
#  "MCTS-E"   = "#2e4057",
#  "MCTS-F" = "darkorchid4"
#)



comb_names <- as_labeller( c(
  "A" = "Design Point 1",
  "B" = "Design Point 2",
  "C" = "Design Point 3",
  "D" = "Design Point 4")
)

rep_el <- theme_minimal() %+replace%
  theme(  strip.text  = element_text(face = "bold", size = 10, vjust=1))

data %>% filter(time %% 10 == 0) %>% 
  mutate(
  combination = case_when(
        rho == 0.95 & k == 10000 ~ "A",
        rho == 0.95 & k == 100 ~ "B",
        rho == 0.75 & k == 10000 ~ "C",
        rho == 0.75 & k == 100 ~ "D"
  ) # att: state attraction, rep: state repulsion, dd: distribution disruption, pd: path attraction
) %>% filter(attacker == "att", 
             `|Q|` == QQ, `|X|` == XX, `|T|` == TT) %>% # change for SA for Simulated Annealing
  filter( (combination == "A" & solver %in% c("MCTS_A", "MCTS_B", "MCTS_C", "MCTS_D", "MCTS_E", "MCTS_F") ) |
          (combination == "B" & solver %in% c("MCTS_A", "MCTS_B", "MCTS_C", "MCTS_D", "MCTS_E", "MCTS_F") ) |
          (combination == "C" & solver %in% c("MCTS_A", "MCTS_B", "MCTS_C", "MCTS_D", "MCTS_E", "MCTS_F") ) |
          (combination == "D" & solver %in% c("MCTS_A", "MCTS_B", "MCTS_C", "MCTS_D", "MCTS_E", "MCTS_F") ) 
          ) %>%
  mutate(solver=replace(solver, 
                        solver %in% c("SA_A"), "SA-A")) %>%
  mutate(solver=replace(solver, 
                        solver %in% c("SA_B"), "SA-B"))%>%
  mutate(solver=replace(solver, 
                        solver %in% c("SA_C"), "SA-C"))%>%
  mutate(solver=replace(solver, 
                        solver %in% c("SA_D"), "SA-D"))%>%
  mutate(solver=replace(solver, 
                        solver %in% c("SA_E"), "SA-E"))%>%
  mutate(solver=replace(solver, 
                        solver %in% c("SA_F"), "SA-F")) %>%
  mutate(solver=replace(solver, 
                        solver %in% c("MCTS_A"), "MCTS-A")) %>%
  mutate(solver=replace(solver, 
                        solver %in% c("MCTS_B"), "MCTS-B")) %>%
  mutate(solver=replace(solver, 
                        solver %in% c("MCTS_C"), "MCTS-C")) %>%
  mutate(solver=replace(solver, 
                        solver %in% c("MCTS_D"), "MCTS-D")) %>%
  mutate(solver=replace(solver, 
                        solver %in% c("MCTS_E"), "MCTS-E")) %>%
  mutate(solver=replace(solver, 
                        solver %in% c("MCTS_F"), "MCTS-F")) %>%
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
  theme(legend.title = element_text(face = "bold" ,size = 11))#+  Uncommment for colors
  #scale_color_manual(values=branded_colors)      Uncomment for colors

ggsave(filename = "experiment_plots_d/exp_2_app_MCTS_attr.png",  ## dir to save
       device = "png", 
      dpi = dpi, width = width, height = height)


         