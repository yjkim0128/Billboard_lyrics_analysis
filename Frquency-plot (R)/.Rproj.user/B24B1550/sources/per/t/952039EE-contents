# R implentation of word frequency analysis & interactive plot generation
#
# By Yongju Kim | Last update Mar 12 2020

library(ggplot2)
library(plotly)
library(dplyr)

split = function(string){
  result = strsplit(string, "[[:space:]]")[[1]]
  return(result)
}

plotting = function(term){
  
  term <- tolower(term)
  
  data <- read.csv(file = "processed_lyrics.csv")
  
  year <- data %>% arrange(year) %>% select(year)
  year <- c(unique(year))
  year <- as.numeric(unlist(year))
  
  avg_freq = c()
  index = 1
  
  for (yr in year){
    
    counter = 0
    yr_data <- data %>% filter(year == yr)
    lyric_split <- yr_data %>% select(filtered_lyrics)
    lyric_split <- apply(lyric_split, 1, split)
    
    for (lyric in lyric_split){
      
      for (word in lyric){
        
        if (word == term){
          counter <- counter + 1
        }
        
      }
  
    }
    
    value = counter / nrow(yr_data)
    avg_freq[index] = value
    index = index + 1
    
  }
  
  temp = data.frame(year, avg_freq)
  graph <- ggplot(data = temp, aes(x = year, y = avg_freq))+geom_line()
  inter <- ggplotly(graph)
  return(inter)
}

#Can put any word of interest instead of "love"
plotting("love")

