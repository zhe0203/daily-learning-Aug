findruns <- function(x,k){
  n <- length(x)
  runs <- NULL
  for (i in 1:(n-k+1)){
    if (all(x[i:(i+k-1)])) runs <- c(runs,i)
  }
  return(runs)
}
x <- c(TRUE,FALSE,FALSE,TRUE,TRUE,TRUE,FALSE,TRUE,TRUE,TRUE,TRUE,TRUE,TRUE)
result <- c()
for (j in 1:length(x)){
  get_num <- findruns(x,j)
  if (length(get_num) > 0) result <- c(result,j)
}
max(result)
