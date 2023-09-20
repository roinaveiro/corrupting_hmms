library(tidyverse)


process_path <- function(main_path, subpath, label){
  
  if(is.null(subpath)){
    paths <- list.files(path=main_path, pattern=NULL, all.files=FALSE,
                        full.names=TRUE)
    
    all_files <- c()
    for(dname in paths){
      tmp <- list.files(path=dname, pattern=".csv", all.files=FALSE,
                        full.names=TRUE)
      
      all_files <- c(all_files, tmp)
    }
    
    full_data  <- read_csv(all_files[1])
    full_data <- full_data  %>% mutate(solver = label) %>%
      group_by(solver, time, attacker, `|Q|`, `|X|`, `|T|`, rho, k) %>% 
      summarise("EU" = mean(scaled_eu), 
                "Lower" = mean(scaled_eu) - 2*sd(scaled_eu),
                "Upper" = mean(scaled_eu) + 2*sd(scaled_eu)
      ) 
    
    for(fname in all_files[2:length(all_files)]){
      tmp_data  <- read_csv(fname)
      tmp_data <- tmp_data  %>% mutate(solver = label) %>%
        group_by(solver, time, attacker, `|Q|`, `|X|`, `|T|`, rho, k) %>% 
        summarise("EU" = mean(scaled_eu), 
                  "Lower" = mean(scaled_eu) - 2*sd(scaled_eu),
                  "Upper" = mean(scaled_eu) + 2*sd(scaled_eu)
        ) 
      
      full_data <- full_data %>% 
        bind_rows(tmp_data)
    }
    return(full_data)
  }
  
  else{
    paths <- list.files(path=main_path, pattern=NULL, all.files=FALSE,
                        full.names=TRUE)
    
    paths <- paste0(paths, subpath)
    
    all_files <- c()
    for(dname in paths){
      tmp <- list.files(path=dname, pattern=".csv", all.files=FALSE,
                        full.names=TRUE)
      
      all_files <- c(all_files, tmp)
    }
    

    full_data  <- read_csv(all_files[1])
    full_data <- full_data %>% mutate(solver = label) %>%
      group_by(solver, time, attacker, `|Q|`, `|X|`, `|T|`, rho, k) %>% 
      summarise("EU" = mean(scaled_eu), 
                "Lower" = mean(scaled_eu) - 2*sd(scaled_eu),
                "Upper" = mean(scaled_eu) + 2*sd(scaled_eu)
      ) 
    
    for(fname in all_files[2:length(all_files)]){
      tmp_data  <- read_csv(fname)
      tmp_data <- tmp_data %>% mutate(solver = label) %>%
        group_by(solver, time, attacker, `|Q|`, `|X|`, `|T|`, rho, k) %>% 
        summarise("EU" = mean(scaled_eu), 
                  "Lower" = mean(scaled_eu) - 2*sd(scaled_eu),
                  "Upper" = mean(scaled_eu) + 2*sd(scaled_eu)
        ) 
      
      full_data <- full_data %>% 
        bind_rows(tmp_data)
    }
    return(full_data)
  }
    
}







