library(tidyverse)
source("utils.R")
# Experiments uncertainty structure
### Fill with the directories with the results for each experiment



## APS - 500
path_APS = "results/experiment_2_scaled_csv/APS"  
data_APS_500 <- process_path(path_APS, "/cs_500", "APS_500")

## APS - 1
path_APS = "results/experiment_2_scaled_csv/APS"
data_APS_1 <- process_path(path_APS, "/cs_1", "APS_1")

################################################################################
################################################################################

## RME
path_RME = "results/experiment_2_scaled_csv/RME"
data_RME <- process_path(path_RME, subpath = NULL, label = "RME")

################################################################################
################################################################################

## RS SA - A
path_SA_A = "results/experiment_2_scaled_csv/SA"
data_SA_A <- process_path(path_SA_A, subpath = "/A", label = "SA_A")

## RS SA - B
path_SA_B = "results/experiment_2_scaled_csv/SA"
data_SA_B <- process_path(path_SA_B, subpath = "/B", label = "SA_B")

## RS SA - C
path_SA_C = "results/experiment_2_scaled_csv/SA"
data_SA_C <- process_path(path_SA_C, subpath = "/C", label = "SA_C")

## RS SA - D
path_SA_D = "results/experiment_2_scaled_csv/SA"
data_SA_D <- process_path(path_SA_D, subpath = "/D", label = "SA_D")

## RS SA - E
path_SA_E = "results/experiment_2_scaled_csv/SA"
data_SA_E <- process_path(path_SA_E, subpath = "/E", label = "SA_E")

## RS SA - F
path_SA_F = "results/experiment_2_scaled_csv/SA"
data_SA_F <- process_path(path_SA_F, subpath = "/F", label = "SA_F")

################################################################################
################################################################################

## RS MCTS - A
path_MCTS_A = "results/experiment_2_scaled_csv/MCTS"
data_MCTS_A <- process_path(path_MCTS_A, subpath = "/A", label = "MCTS_A")

## RS MCTS - B
path_MCTS_B = "results/experiment_2_scaled_csv/MCTS"
data_MCTS_B <- process_path(path_MCTS_B, subpath = "/B", label = "MCTS_B")

## RS MCTS - C
path_MCTS_C = "results/experiment_2_scaled_csv/MCTS"
data_MCTS_C <- process_path(path_MCTS_C, subpath = "/C", label = "MCTS_C")

## RS MCTS - D
path_MCTS_D = "results/experiment_2_scaled_csv/MCTS"
data_MCTS_D <- process_path(path_MCTS_D, subpath = "/D", label = "MCTS_D")

## RS MCTS - E
path_MCTS_E = "results/experiment_2_scaled_csv/MCTS"
data_MCTS_E <- process_path(path_MCTS_E, subpath = "/E", label = "MCTS_E")

## RS MCTS - F
path_MCTS_F = "results/experiment_2_scaled_csv/MCTS"
data_MCTS_F <- process_path(path_MCTS_F, subpath = "/F", label = "MCTS_F")

################################################################################
################################################################################


data_RME %>% bind_rows(data_APS_1) %>% bind_rows(data_APS_500) %>%
  bind_rows(data_SA_A) %>% bind_rows(data_SA_B) %>% bind_rows(data_SA_C) %>%
  bind_rows(data_SA_D) %>% bind_rows(data_SA_E) %>% bind_rows(data_SA_F) %>%
  bind_rows(data_MCTS_A) %>% bind_rows(data_MCTS_B) %>% 
  bind_rows(data_MCTS_C) %>% bind_rows(data_MCTS_D) %>% 
  bind_rows(data_MCTS_E) %>% bind_rows(data_MCTS_F) %>%
  write_csv("results/full_data.csv")
