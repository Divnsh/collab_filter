library(NMF)
x1 = c(5,4,1,1) # get ratings for 5 users on 4 movies
x2 = c(4,5,1,1)
x3 = c(1,1,5,5)
x4 = c(1,1,4,5)
x5 = c(1,1,5,4)
R = as.matrix(rbind(x1,x2,x3,x4,x5)) # 5 rows, 4 columns
set.seed(12345)

res = nmf(R,2,"lee") # lee & seung method
V.hat = fitted(res)
print(V.hat) # estimated target matrix

w=basis(res) # W user feature matrix
dim(w)
print(w)
h=coef(res) # H movie feature matrix
dim(h)
print(h)

# recommendor system via clustering based on vectors in H
movies = data.frame(t(h))
features = cbind(movies$X1, movies$X2)
plot(movies$X1, movies$X2)

## 
library(recommenderlab) ##Collaborative filtering
library(ggplot2) ## Visualization of results
library(reshape2) ## Alteration of data
library(SNFtool)

# Read training file along with header
tr = read.csv('train_v2.csv')
tr = tr[,-1]

# Check, if removed
tr[tr$user==1,]
g = acast(tr,user~movie)
class(g)
R=as.matrix(g)
# Convert R into realRatingMatrix data structure
# realRatingMatrix is a recommenderlab sparse-matrix like data-structure

r = as(R,"realRatingMatrix")
# View r in other possible ways
as(r,"list") # as a list
as(r, "matrix") # as a sparse matrix
as(r,"dasta.frame") # as a data frame

# normalise the rating matrix
r_m = normalize(r)
as(r_m, "list")

# Create a recommender object (model)
rec1 = Recommender(r[1:nrow(r)], method = "UBCF", param = list(normalize = "Z-score", method ="Cosine",nn=5))
rec2 = Recommender(r[1:nrow(r)], method ="UBCF", param = list(normalize="Z-score", method ="Jaccard",nn=5))
rec = Recommender(r[1:nrow(r)], method="POPULAR")

# Depending upon your selection, examine what you got
print(rec2)
names(getModel(rec2))
getModel(rec2)

## Model in Action
recommended.items.u1022 = predict(rec2, r["1022",], n=5)
as(recommended.items.u1022,"list")
# obtain top 3
recommended.items.u1022 = bestN(recommended.items.u1022, n=3)
as(recommended.items.u1022, "list")


## Create predictions for the same user
# to predict affinity to all related items
recom = predict(rec2, r["1022",], type="ratings")
# Convert all your recommendations to list structure
rec_list = as(recom, "list")
# Access this list for user 1, item at index 2
rec_list[[1]][2]
# Convert to data frame
u1 = as.data.frame(rec_list[[1]])
attributes(u1)
class(u1)
# Create column by name of id in data frame u1 and populate it with row names
u1$id = row.names(u1)
# Now access movie ratings in column 1 for u1
u1[u1$id==3952,1]

## Evaluation ##
e = evaluationScheme(r[1:100], method = "split", train = .7, given = 15, goodRating=4)
e
algorithms = list("random items"=list(name="RANDOM"), "popular items" = list(name="POPULAR")
                  , "user-based CF" = list(name="UBCF"), "svd" = list(name="SVD"))
results = evaluate(e, algorithms, n=c(1,3,5,10,15,20,25))
plot(results, annotate = 1:4, legend="topleft")
