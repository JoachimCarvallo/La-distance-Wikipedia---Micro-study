# Librairies
library(dplyr)
library(rvest)

# Liste de mots 
mots = c("red", "green", "blue", "white", "black", "yellow", "orange", "purple", "grey", "pink", "brown",
         "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
         "dog", "cat", "lion","wolf", "cow", "goat","sheep", "tiger", "dolphin","shark","salmon",
         "actuary", "doctor", "pharmacist", "nurse", "accountant", "engineer", "baker", "farmer", "plumber", "electrician", "builder",
         "pants", "gloves", "dress", "t-shirt", "shirt", "hat", "shoes", "socks", "sweater", "underwears", "coat",
         "kitchen", "bathroom", "living room", "dining room", "toilets", "hallway", "entrance", "bedroom", "attic", "cellar", "staircase",
         "apple", "pear", "peach", "apricot", "melon", "watermelon", "banana", "kiwi", "mango", "grape", "pineapple",
         "salad", "avocado", "leek", "cabbage", "carrot", "potato", "zucchini", "spinach", "radish", "pepper", "tomato",
         "joy", "excitement", "surprise", "sadness", "anger", "disgust", "contempt", "fear", "shame", "guilt", "shyness",
         "Michael", "William", "James", "Benjamin", "John", "Emma", "Olivia", "Mary", "Jennifer", "Elizabeth", "Sarah")

# Web scaping de exalead 
n_search = matrix(0, nrow = length(mots), ncol = length(mots), dimnames = list(mots, mots))
count = 0

for (i in 1:2){
  for(i in 1:length(mots)){
    for(j in i:length(mots)){
      
      if (n_search[i,j] == 0 | is.na(n_search[i,j])){
        if (i == j) {url = paste("https://www.exalead.com/search/wikipedia/results/?q=", mots[i], sep = "")}
        else {url = paste(c("https://www.exalead.com/search/wikipedia/results/?q=", mots[i], "+", mots[j]), collapse = "")}
        
        ok <- FALSE
        counter <- 0
        while (ok == FALSE & counter <= 5) {
          counter <- counter + 1
          out <- tryCatch({                  
            text <- url %>% read_html() %>% html_node(xpath = '/html/body/div[3]/form/div/div/small') %>% html_text()
            text <- gsub("results", "", text)
            text <- gsub("[[:space:]]", "", text)
            text <- as.numeric(gsub(",", "", text))
          },
          error = function(e) {
            Sys.sleep(2)
            e
          }
          )
          if ("error" %in% class(out)) {
            cat(".")
          } else {
            ok <- TRUE
          }
        }
        
        n_search[i,j] <- text
        count <- count + 1
        if(count %% 100 == 0){print(c(i,j))}
      }
    }
  }
}

# Completion de la matrice 
n_search <-  n_search + t(n_search) - diag(diag(n_search))

# Enregistrement des données
write.csv(n_search, "D:/Scolarité/MS IA Télécom Paris/Complexity & Intelligence/n_search_10_classes.csv")
