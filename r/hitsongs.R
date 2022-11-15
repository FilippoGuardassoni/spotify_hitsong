#Time Span: 2007 - 2021

'#install.packages ("ggplot2")
install.packages("gridExtra")
install.packages("dplyr")
install.packages("pastecs")
install.packages ("reshape")
install.packages("reshape2")
install.packages("factoextra")
install.packages("tidyverse")
install.packages("corrplot")
install.packages("caTools")
install.packages("pROC")
install.packages("caret")
install.packages("repr")
install.packages("glmnet")
install.packages("Rcpp")
install.packages("readr")
install.packages("randomForest")
install.packages("parameters")
install.packages("see")
install.packages("scales")
install.packages("e1071")#'

library (ggplot2)
library (purrr)
library(gridExtra)
library(dplyr)
select<-dplyr::select
library(pastecs)
library (reshape)
library(reshape2)
library(tidyverse)
library(corrplot)
library(MASS)
library(caTools)
library(pROC)
library(caret)
library(repr)
library(glmnet)
library(Rcpp)
library(readr)
library(randomForest)
library(factoextra)
library(cluster)
library(parameters)
library(RColorBrewer)
.rs.restartR()
library(scales)
library(dendextend)
library(e1071)

set.seed(1234)

songs <- read.csv("...")
attach(songs)

songs <- select(songs, -c(X))

#EDA AND CLEANING
#Some overview statistics
summary(songs)
str(songs)
stat.desc(songs)

songs$rel_date <- as.Date(songs$rel_date, "%Y-%m-%d")
songs$rel_year <- as.integer(format(as.Date(songs$rel_date, format="%Y-%m-%d"),"%Y"))

yearcount <- songs %>%
  group_by(rel_year) %>%
  summarize(Number_of_Songs = n())

ggplot(yearcount, aes(x = rel_year, y = Number_of_Songs)) + 
  geom_bar(stat="identity", fill="orange") + theme_minimal() + 
  scale_x_continuous(breaks=pretty_breaks(10))

ggplot(songs, aes(x = rel_year, y = hit)) + 
  geom_bar(stat="identity", fill="red") + theme_minimal() + 
  scale_x_continuous(breaks=pretty_breaks(10))

songs.time <- filter(songs, rel_year >= 2007)

#Histograms
h1 <- ggplot(data = songs, aes(x = acousticness)) + geom_histogram(bins = 5)
h2 <- ggplot(data = songs, aes(x = danceability)) + geom_histogram(bins = 40)
h3 <- ggplot(data = songs, aes(x = duration_ms)) + geom_histogram(bins = 40)
h4 <- ggplot(data = songs, aes(x = energy)) + geom_histogram(bins = 40)
h5 <- ggplot(data = songs, aes(x = instrumentalness)) + geom_histogram(bins = 5)
h6 <- ggplot(data = songs, aes(x = liveness)) + geom_histogram(bins = 5)
h7 <- ggplot(data = songs, aes(x = loudness)) + geom_histogram(bins = 5)
h8 <- ggplot(data = songs, aes(x = speechiness)) + geom_histogram(bins = 5)
h9 <- ggplot(data = songs, aes(x = tempo)) + geom_histogram(bins = 5)
h10 <- ggplot(data = songs, aes(x = time_signature)) + geom_histogram(bins = 5)
h11 <- ggplot(data = songs, aes(x = valence)) + geom_histogram(bins = 5)

grid.arrange(grobs = list(h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11), ncol = 3, top = "Histograms")

#Outliers
meltData <- melt(songs)
outliers <- ggplot(meltData, aes(factor(variable), value))
outliers + geom_boxplot() + facet_wrap(~variable, scale="free")

#Distribution of hits
ggplot(songs, aes(factor(hit), fill=factor(hit))) + geom_bar(stat="count", width=0.7) + theme_minimal()
ggplot(songs, aes(factor(featuring), fill=factor(hit))) + geom_bar(stat="count",position ="dodge", width=0.7) + theme_minimal()

#Distribution of songs by decades
songs.time$decade <- paste0(substr(songs.time$rel_date, start = 1, stop = 3),0)

decadecount <- songs.time %>%
  group_by(decade) %>%
  summarize(Number_of_Songs = n())

ggplot(decadecount, aes(x = decade, y = Number_of_Songs, fill = decade)) +
  geom_bar(stat="identity")

ggplot(songs.time, aes(x = rel_year, y = hit)) + 
  geom_bar(stat="identity", fill="blue") + theme_minimal() + 
  scale_x_continuous(breaks=pretty_breaks(10))


#Distribution of music features over time
subset.songs <- select(songs.time, -c(pop_artist,tot_followers,avail_mark,pop_track,search,rel_date,decade,rel_day,rel_month,week_day_out,featuring,hit))
songs.tidy <- unite(subset.songs, id, c(artist_name,track_name), sep="_", remove = FALSE)
songs.tidy$artist_name <- NULL
songs.tidy$track_name <- NULL
songs.tidy.time <- gather(songs.tidy, cell, value,-rel_year,-id)

#Distribution of music features mean over time with confidence intervals
songs.tidy.time.mean <- songs.tidy.time %>%
  group_by(rel_year,cell) %>%
  summarise(n = n(), 
            mean=mean(value),
            sd = sd(value)) %>%
  mutate(sem = sd / sqrt(n - 1),
         CI_lower = mean + qt((1-0.95)/2, n - 1) * sem,
         CI_upper = mean - qt((1-0.95)/2, n - 1) * sem)

ggplot(songs.tidy.time.mean[songs.tidy.time.mean$cell == 'acousticness', ], aes(x=rel_year, y=mean, color = cell)) +
  geom_line(aes(x=rel_year, y=mean, color=cell)) +
  geom_ribbon(aes(ymin=CI_lower,ymax=CI_upper,fill=cell),color="grey70",alpha=0.4)
ggplot(songs.tidy.time.mean[songs.tidy.time.mean$cell == 'danceability', ], aes(x=rel_year, y=mean, color = cell)) +
  geom_line(aes(x=rel_year, y=mean, color=cell)) +
  geom_ribbon(aes(ymin=CI_lower,ymax=CI_upper,fill=cell),color="grey70",alpha=0.4)
ggplot(songs.tidy.time.mean[songs.tidy.time.mean$cell == 'duration_ms', ], aes(x=rel_year, y=mean, color = cell)) +
  geom_line(aes(x=rel_year, y=mean, color=cell)) +
  geom_ribbon(aes(ymin=CI_lower,ymax=CI_upper,fill=cell),color="grey70",alpha=0.4)
ggplot(songs.tidy.time.mean[songs.tidy.time.mean$cell == 'energy', ], aes(x=rel_year, y=mean, color = cell)) +
  geom_line(aes(x=rel_year, y=mean, color=cell)) +
  geom_ribbon(aes(ymin=CI_lower,ymax=CI_upper,fill=cell),color="grey70",alpha=0.4)
ggplot(songs.tidy.time.mean[songs.tidy.time.mean$cell == 'instrumentalness', ], aes(x=rel_year, y=mean, color = cell)) +
  geom_line(aes(x=rel_year, y=mean, color=cell)) +
  geom_ribbon(aes(ymin=CI_lower,ymax=CI_upper,fill=cell),color="grey70",alpha=0.4)
ggplot(songs.tidy.time.mean[songs.tidy.time.mean$cell == 'liveness', ], aes(x=rel_year, y=mean, color = cell)) +
  geom_line(aes(x=rel_year, y=mean, color=cell)) +
  geom_ribbon(aes(ymin=CI_lower,ymax=CI_upper,fill=cell),color="grey70",alpha=0.4)
ggplot(songs.tidy.time.mean[songs.tidy.time.mean$cell == 'loudness', ], aes(x=rel_year, y=mean, color = cell)) +
  geom_line(aes(x=rel_year, y=mean, color=cell)) +
  geom_ribbon(aes(ymin=CI_lower,ymax=CI_upper,fill=cell),color="grey70",alpha=0.4)
ggplot(songs.tidy.time.mean[songs.tidy.time.mean$cell == 'speechiness', ], aes(x=rel_year, y=mean, color = cell)) +
  geom_line(aes(x=rel_year, y=mean, color=cell)) +
  geom_ribbon(aes(ymin=CI_lower,ymax=CI_upper,fill=cell),color="grey70",alpha=0.4)
ggplot(songs.tidy.time.mean[songs.tidy.time.mean$cell == 'tempo', ], aes(x=rel_year, y=mean, color = cell)) +
  geom_line(aes(x=rel_year, y=mean, color=cell)) +
  geom_ribbon(aes(ymin=CI_lower,ymax=CI_upper,fill=cell),color="grey70",alpha=0.4)
ggplot(songs.tidy.time.mean[songs.tidy.time.mean$cell == 'valence', ], aes(x=rel_year, y=mean, color = cell)) +
  geom_line(aes(x=rel_year, y=mean, color=cell)) +
  geom_ribbon(aes(ymin=CI_lower,ymax=CI_upper,fill=cell),color="grey70",alpha=0.4)

qplot(songs.time$acousticness, songs.time$danceability, group = as.factor(songs.time$hit), colour = as.factor(songs.time$hit)) + theme_minimal()
qplot(songs.time$acousticness, songs.time$loudness, group = as.factor(songs.time$hit), colour = as.factor(songs.time$hit)) + theme_minimal()
qplot(songs.time$acousticness, songs.time$energy, group = as.factor(songs.time$hit), colour = as.factor(songs.time$hit)) + theme_minimal()
qplot(songs.time$acousticness, songs.time$liveness, group = as.factor(songs.time$hit), colour = as.factor(songs.time$hit)) + theme_minimal()
qplot(songs.time$acousticness, songs.time$loudness, group = as.factor(songs.time$hit), colour = as.factor(songs.time$hit)) + theme_minimal()
qplot(songs.time$acousticness, songs.time$speechiness, group = as.factor(songs.time$hit), colour = as.factor(songs.time$hit)) + theme_minimal()


#Count of hits by artist
songs.count <- songs.time %>%
  group_by(hit, artist_name) %>%
  summarise(count = n()) %>%
  top_n(n = 10, wt = count)

ggplot(songs.count[songs.count$hit == 1, ], aes(x = reorder(artist_name, -count), count, fill = artist_name)) +
  geom_col() +
  facet_grid(~hit, scales = "free_x") + theme_minimal()

songs.normalized <- as.data.frame(songs.time)
songs.normalized <- songs.time %>% mutate_at(10, funs((.-min(.))/max(.-min(.))))
songs.normalized <- songs.normalized %>% mutate_at(14, funs((.-min(.))/max(.-min(.))))
songs.normalized <- songs.normalized %>% mutate_at(16:17, funs((.-min(.))/max(.-min(.))))

songs.features <- songs.normalized %>% 
  group_by(hit, ) %>%
  summarise(across(acousticness:valence, mean))

songs.features.long<-melt(songs.features,id.vars="hit")

ggplot(songs.features.long,aes(x=variable,y=value,fill=factor(hit)))+
  geom_bar(stat="identity",position="dodge")+ theme_minimal()

#DATA MODELLING

#Correlation matrix
songs.correlation <- select(songs.time, -c(artist_name, track_name, rel_date, search, rel_day, rel_month, week_day_out, rel_year, decade))
songs.correlation <- songs.correlation[,c(1:17)]
songs.correlation <- songs.correlation[,c(16, 17, 1:15)]

corr <- cor(songs.correlation)
round(corr, 2)

corrplot.mixed(corr, order = "hclust", 
               tl.col = "black", tl.srt = 45)

dataset <- select(songs.correlation, -c(7))

#Distribution of features
plot(dataset[,7:16], pch=20 , cex=1.5 , col=dataset$hit)

# UNSUPERVISED LEARNING
rescale.dataset.hit <- songs.correlation %>%
  filter(songs.correlation$hit == 1) %>%
  select(-c(hit, featuring))
rescale.dataset.hit <- scale(rescale.dataset.hit, center = TRUE, scale = TRUE)

cluster.dataset <- songs.correlation %>%
  filter(songs.correlation$hit == 1) %>%
  select(-c(hit, featuring))


# K-MEANS CLUSTERING
set.seed(1234)

fviz_nbclust(rescale.dataset.hit, kmeans, method = "wss")
fviz_nbclust(rescale.dataset.hit, kmeans, method = "silhouette")
gap.stat <- clusGap(rescale.dataset.hit, FUN = kmeans, nstart = 25,
                    K.max = 10, B = 50)
fviz_gap_stat(gap.stat)

'# n.clust <- n_clusters(rescale.dataset.hit,
                      package = c("cluster", "NbClust", "mclust"),
                      standardize = FALSE)
n.clust

ggplot(n.clust, aes(x = n_Clusters, fill= ifelse(n_Clusters == 3, "Highlighted","Normal"))) +
  geom_bar() +
  scale_fill_manual(name = "Number of Clusters", values=c("red","grey50")) +
  scale_x_continuous(limits = c(0,11), breaks = seq(0,11,1)) #'

# Comparing number of k
k3 <- kmeans(rescale.dataset.hit, centers = 2, nstart = 25)
k4 <- kmeans(rescale.dataset.hit, centers = 3, nstart = 25)
k5 <- kmeans(rescale.dataset.hit, centers = 4, nstart = 25)
k6 <- kmeans(rescale.dataset.hit, centers = 5, nstart = 25)

p1 <- fviz_cluster(k3, geom = "point", data = rescale.dataset.hit) + ggtitle("k = 2")
p2 <- fviz_cluster(k4, geom = "point", data = rescale.dataset.hit) + ggtitle("k = 3")
p3 <- fviz_cluster(k5, geom = "point", data = rescale.dataset.hit) + ggtitle("k = 4")
p4 <- fviz_cluster(k6, geom = "point", data = rescale.dataset.hit) + ggtitle("k = 5")

grid.arrange(p1, p2, p3, p4, nrow = 2)

kmean <- kmeans(rescale.dataset.hit, centers = 3, nstart = 25)
kmean

center <- kmean$centers
center

(BSS <- kmean$betweenss)
(TSS <- kmean$totss)
BSS / TSS * 100

kmean$cluster <- as.factor(kmean$cluster)
kmc <- kmean$cluster
kmc

kmean.dataset <- data.frame(cluster = kmc, rescale.dataset.hit)
head(kmean.dataset)

fviz_cluster(kmean, rescale.dataset.hit, geom = c("point"), ggtheme = theme_bw())

# create dataset with the cluster number
cluster <- c(1:3)
center.dataset <- data.frame(cluster, center)

center.reshape <- gather(center.dataset, features, values, acousticness: time_signature_scal)
head(center.reshape)

# Create the palette
hm.palette <-colorRampPalette(rev(brewer.pal(10, 'RdYlGn')),space='Lab')
ggplot(data = center.reshape, aes(x = features, y = cluster, fill = values)) +
  scale_y_continuous(breaks = seq(1, 7, by = 1)) +
  geom_tile() +
  coord_equal() +
  scale_fill_gradientn(colours = hm.palette(90)) +
  theme_classic()


kmean.table <- data.frame(kmean$size, kmean$centers)
kmean.dataset <- data.frame(cluster = kmean$cluster, cluster.dataset)
# head of df
head(kmean.dataset)

#find means of each cluster
aggregate(cluster.dataset, by=list(cluster=kmean$cluster), mean)

k.mean.cluster1 <- filter(kmean.dataset, cluster == 1)
summary(k.mean.cluster1)

k.mean.cluster2 <- filter(kmean.dataset, cluster == 2)
summary(k.mean.cluster2)

k.mean.cluster3 <- filter(kmean.dataset, cluster == 3)
summary(k.mean.cluster3)

# Boxplot
df.m <- melt(kmean.dataset, id.var = "cluster")
ggplot(data = df.m, aes(x=cluster, y=value)) + geom_boxplot(aes(fill=variable)) + scale_y_continuous(limits = c(-5,5))

# HIERARCHICAL CLUSTERING (AGGLOMERATIVE)
set.seed(1234)

# Distance matrix
res.dist <- get_dist(rescale.dataset.hit, method = "euclidean")
head(round(as.matrix(res.dist), 2))[, 1:15]
fviz_dist(res.dist, gradient = list(low = "#E0FFFF", mid = "white", high = "#FF4500"))

# Method to assess
m <- c("average", "single", "complete", "ward")
names(m) <- c("average", "single", "complete", "ward")

# Function to compute coefficient
ac_hit <- function(x) {
  agnes(rescale.dataset.hit, method = x)$ac
}

map.hit <- map_dbl(m, ac_hit)
map.hit

# Hierarchical clustering using Ward Linkage
res.hc.agg.hit <- agnes(rescale.dataset.hit, metric = "euclidean", method = "ward")
res.hc.agg.hit
res.hc.agg.hit$ac

# Plot the obtained dendrogram
dend.agg <- fviz_dend(res.hc.agg.hit, k = 6, rect = TRUE, color_labels_by_k = TRUE)
dend.agg

sub_grp <- cutree(as.hclust(res.hc.agg.hit), k = 6)
table(sub_grp)

fviz_cluster(list(data = rescale.dataset.hit, cluster = sub_grp), geom = "point") # scatter plot


agg.cluster.dataset <- cbind(cluster.dataset, cluster = sub_grp)

# Boxplot
ggplot(data = agg.cluster.dataset, aes(x=factor(cluster),y=pop_artist, fill=factor(cluster))) + 
  geom_violin(trim=FALSE) + 
  geom_boxplot(width = 0.07) + 
  scale_fill_brewer() + 
  theme_minimal()
ggplot(data = agg.cluster.dataset, aes(x=factor(cluster),y=pop_track, fill=factor(cluster))) + 
  geom_violin(trim=FALSE) + 
  geom_boxplot(width = 0.07) + 
  scale_fill_brewer() + 
  theme_minimal()
ggplot(data = agg.cluster.dataset, aes(x=factor(cluster),y=danceability, fill=factor(cluster))) + 
  geom_violin(trim=FALSE) + 
  geom_boxplot(width = 0.07) + 
  scale_fill_brewer() + 
  theme_minimal()
ggplot(data = agg.cluster.dataset, aes(x=factor(cluster),y=energy, fill=factor(cluster))) + 
  geom_violin(trim=FALSE) + 
  geom_boxplot(width = 0.07) + 
  scale_fill_brewer() + 
  theme_minimal()
ggplot(data = agg.cluster.dataset, aes(x=factor(cluster),y=loudness, fill=factor(cluster))) + 
  geom_violin(trim=FALSE) + 
  geom_boxplot(width = 0.07) + 
  scale_fill_brewer() + 
  theme_minimal()
ggplot(data = agg.cluster.dataset, aes(x=factor(cluster),y=speechiness, fill=factor(cluster))) + 
  geom_violin(trim=FALSE) + 
  geom_boxplot(width = 0.07) + 
  scale_fill_brewer() + 
  theme_minimal()
ggplot(data = agg.cluster.dataset, aes(x=factor(cluster),y=acousticness, fill=factor(cluster))) + 
  geom_violin(trim=FALSE) + 
  geom_boxplot(width = 0.07) + 
  scale_fill_brewer() + 
  theme_minimal()
ggplot(data = agg.cluster.dataset, aes(x=factor(cluster),y=instrumentalness, fill=factor(cluster))) + 
  geom_violin(trim=FALSE) + 
  geom_boxplot(width = 0.07) + 
  scale_fill_brewer() + 
  theme_minimal()
ggplot(data = agg.cluster.dataset, aes(x=factor(cluster),y=valence, fill=factor(cluster))) + 
  geom_violin(trim=FALSE) + 
  geom_boxplot(width = 0.07) + 
  scale_fill_brewer() + 
  theme_minimal()
ggplot(data = agg.cluster.dataset, aes(x=factor(cluster),y=tempo, fill=factor(cluster))) + 
  geom_violin(trim=FALSE) + 
  geom_boxplot(width = 0.07) + 
  scale_fill_brewer() + 
  theme_minimal()
ggplot(data = agg.cluster.dataset, aes(x=factor(cluster),y=liveness, fill=factor(cluster))) + 
  geom_violin(trim=FALSE) + 
  geom_boxplot(width = 0.07) + 
  scale_fill_brewer() + 
  theme_minimal()


# HIERARCHICAL CLUSTERING (DIVISIVE)
set.seed(1234)
# compute divisive hierarchical clustering
res.hc.div.hit <- diana(rescale.dataset.hit)
res.hc.div.hit

# Divise coefficient; amount of clustering structure found
res.hc.div.hit$dc

# plot dendrogram
dend.div <- fviz_dend(res.hc.div.hit, k = 5, rect = TRUE, color_labels_by_k = TRUE)
dend.div

sub_grp <- cutree(as.hclust(res.hc.div.hit), k = 5)
table(sub_grp)

fviz_cluster(list(data = rescale.dataset.hit, cluster = sub_grp), geom = "point") # scatter plot

# Measure difference between two methods
dend_list <- dendlist(as.dendrogram(as.hclust(res.hc.agg.hit)), as.dendrogram(as.hclust(res.hc.div.hit)))
tanglegram(as.dendrogram(as.hclust(res.hc.agg.hit)), as.dendrogram(as.hclust(res.hc.div.hit)), 
           highlight_distinct_edges = FALSE, common_subtrees_color_lines = TRUE,  common_subtrees_color_branches = FALSE,
           main = paste("entanglement =", round(entanglement(dend_list), 2)))


#SUPERVISED LEARNING
#Splitting dataset
set.seed(1234)
split <- sample.split(dataset, SplitRatio = 0.8)
split

train.reg <- subset(dataset, split == "TRUE")
test.reg <- subset(dataset, split == "FALSE")

#Pre-processing and Scaling
cols <- colnames(train.reg[,3:16])
pre.proc.val <- preProcess(train.reg[,cols], method = c("center", "scale"))

train.reg[,cols] = predict(pre.proc.val, train.reg[,cols])
test.reg[,cols] = predict(pre.proc.val, test.reg[,cols])

#MODEL 1: BINARY LOGISTIC REGRESSION
set.seed(1234)

logit.1 <- glm(hit~., family = binomial,data = train.reg)
summary(logit.1)

logit.2 <- stepAIC(logit.1)
summary(logit.2)

anova(logit.1, logit.2, test="Chisq")

#Evaluation of in-sample performances
summary(logit.2$fitted.values)
hist(logit.2$fitted.values,main = " Histogram ", col = 'light green')

results.train <- as.data.frame(train.reg)
results.train$logit2 <- if_else(logit.2$fitted.values>0.5, 1, 0)

train.eval.logit <- table(train.reg$hit,results.train$logit2)
rownames(train.eval.logit) <- c("Obs. non-hit","Obs. hit")
colnames(train.eval.logit) <- c("Pred. non-hit","Pred. hit")
train.eval.logit

accuracy.train.logit2 <- sum(diag(train.eval.logit))/sum(train.eval.logit)
accuracy.train.logit2

plot(ggpredict(logit.2))

'#
Precision : TP / (TP+FP) 
Recall : TP / (TP+FN) 
F1 Score : (2 * Precision * Recall) / (Precision+Recall)
#'

precision.train.logit2 <- diag(train.eval.logit) / colSums(train.eval.logit)
recall.train.logit2 <- diag(train.eval.logit) / rowSums(train.eval.logit) 
f1.train.logit2 <- (2 * precision.train.logit2 * recall.train.logit2) / (precision.train.logit2 + recall.train.logit2) 

data.frame(precision.train.logit2, recall.train.logit2, f1.train.logit2)

roc(hit~logit.2$fitted.values, data = train.reg, plot = TRUE, main = "ROC CURVE", col= "blue")
auc(hit~logit.2$fitted.values, data = train.reg)

'#Alternative way (names not updated)
train.reg$prediction <-  factor(train.reg$prediction)
train.reg$hit <-  factor(train.reg$hit)

confusion.res <- confusionMatrix(data=train.reg$prediction, reference = train.reg$hit)
confusion.res#'

#Evaluation of out-of-sample performances
pred.test.logit <- predict(logit.2, test.reg, type="response")

results.test <- as.data.frame(test.reg)
results.test$logit2 <- if_else(pred.test.logit>0.5, 1, 0)

test.eval.logit <- table(test.reg$hit,results.test$logit2)
rownames(test.eval.logit) <- c("Obs. non-hit","Obs. hit")
colnames(test.eval.logit) <- c("Pred. non-hit","Pred. hit")
test.eval.logit

accuracy.test.logit2 <- sum(diag(test.eval.logit))/sum(test.eval.logit)
accuracy.test.logit2

precision.test.logit2 <- diag(test.eval.logit) / colSums(test.eval.logit)
precision.test.logit2
recall.test.logit2 <- diag(test.eval.logit) / rowSums(test.eval.logit) 
recall.test.logit2
f1.test.logit2 <- (2 * precision.test.logit2 * recall.test.logit2) / (precision.test.logit2 + recall.test.logit2) 

data.frame(precision.test.logit2, recall.test.logit2, f1.test.logit2)

roc(hit~pred.test.logit, data = test.reg, plot = TRUE, main = "ROC CURVE", col= "blue")
auc(hit~pred.test.logit, data = test.reg)

test.eval.logit <- test.eval.logit / rowSums(test.eval.logit)
test.eval.logit <- as.data.frame(test.eval.logit, stringsAsFactors = TRUE)
test.eval.logit$Var2 <- factor(test.eval.logit$Var2, rev(levels(test.eval.logit$Var2)))

ggplot(test.eval.logit, aes(Var1, Var2, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = scales::percent(Freq))) +
  scale_fill_gradient(low = "white", high = "#3575b5") +
  labs(x = "True", y = "Guess", title = "Confusion matrix",
       fill = "Select")


#MODEL 2: DISCRIMINANT ANALYSIS
#LINEAR
set.seed(1234)
# Fit the model to train data
lda.model <- lda(hit~ ., data = train.reg)
lda.model
par("mar")
par(mar=c(1,1,1,1))
plot(lda.model)

# Make predictions of train data
pred.train.lda <- predict(lda.model, train.reg)

head(pred.train.lda$class)
head(pred.train.lda$posterior)
head(pred.train.lda$x)

ldahist(data = pred.train.lda$x[,1], g = train.reg$hit)

pred.train.lda.class <- predict(lda.model, train.reg)$class
train.eval.lda <- table(Predicted = pred.train.lda.class, Actual = train.reg$hit)
train.eval.lda

results.train$lda <- pred.train.lda.class

# Model accuracy of train data
accuracy.train.lda <- sum(diag(train.eval.lda))/sum(train.eval.lda)

# Make predictions of test data
pred.test.lda.class <- predict(lda.model, test.reg)$class
test.eval.lda <- table(Predicted = pred.test.lda.class, Actual = test.reg$hit)
test.eval.lda

results.test$lda <- pred.test.lda.class

# Model accuracy of test data
accuracy.test.lda <- sum(diag(test.eval.lda))/sum(test.eval.lda)

# QUADRATIC
# Fit the model
set.seed(1234)
qda.model <- qda(hit~ ., data = train.reg)
qda.model

# Make predictions of train data
pred.train.qda <- predict(qda.model, train.reg)

head(pred.train.qda$class)
head(pred.train.qda$posterior)
head(pred.train.qda$x)

pred.train.qda.class <- predict(qda.model, train.reg)$class
train.eval.qda <- table(Predicted = pred.train.qda.class, Actual = train.reg$hit)
train.eval.qda

results.train$qda <- pred.train.qda.class

# Model accuracy of train data
accuracy.train.qda <- sum(diag(train.eval.qda))/sum(train.eval.qda)

# Make predictions
pred.test.qda.class <- predict(qda.model, test.reg)$class
test.eval.qda <- table(Predicted = pred.test.qda.class, Actual = test.reg$hit)
test.eval.qda

results.test$qda <- pred.test.qda.class

# Model accuracy of test data
accuracy.test.qda <- sum(diag(test.eval.qda))/sum(test.eval.qda)


# MODEL 3: LASSO REGRESSION
set.seed(1234)
x = as.matrix(train.reg[2:16])
y.train = train.reg$hit
x.test = as.matrix(test.reg[2:16])
y.test = test.reg$hit
# Setting alpha = 1 implements lasso regression
cv.lasso.reg <- cv.glmnet(x, y.train, alpha = 1, family = "binomial",
                          type.measure = "class")
plot(cv.lasso.reg)
# Best
lambda.best <- cv.lasso.reg$lambda.min
lambda.best
lasso.model <- glmnet(x, y.train, alpha = 1, lambda = lambda.best,
                      family="binomial")
lasso.coef <- predict(lasso.model, type = "coefficients", s = lambda.best,
                      newx = x)
lasso.coef
#Lasso path
plot(cv.lasso.reg$glmnet.fit, "lambda", label=FALSE)
probabilities.train <- lasso.model %>% predict(newx = x, s = lambda.best,
                                               type = "response")
train.lasso.pred <- ifelse(probabilities.train > 0.5, 1, 0)
summary(probabilities.train)
results.train$lasso <- train.lasso.pred
train.eval.lasso <- table(train.reg$hit,results.train$lasso)
rownames(train.eval.lasso) <- c("Obs. non-hit","Obs. hit")
colnames(train.eval.lasso) <- c("Pred. non-hit","Pred. hit")
train.eval.lasso
accuracy.train.lasso <- sum(diag(train.eval.lasso))/sum(train.eval.lasso)
accuracy.train.lasso
probabilities.test <- lasso.model %>% predict(newx = x.test, s =
                                                lambda.best, type = "response")
test.lasso.pred <- ifelse(probabilities.test > 0.5, 1, 0)
summary(probabilities.test)
results.test$lasso <- test.lasso.pred
test.eval.lasso <- table(test.reg$hit,results.test$lasso)
rownames(test.eval.lasso) <- c("Obs. non-hit","Obs. hit")
colnames(test.eval.lasso) <- c("Pred. non-hit","Pred. hit")

test.eval.lasso
accuracy.test.lasso <- sum(diag(test.eval.lasso))/sum(test.eval.lasso)
accuracy.test.lasso

precision.test.lasso <- diag(test.eval.lasso) / colSums(test.eval.lasso)
precision.test.lasso
recall.test.lasso <- diag(test.eval.lasso) / rowSums(test.eval.lasso) 
recall.test.lasso

roc(y.test, c(test.lasso.pred), plot = TRUE, main = "ROC CURVE", col= "blue")

# MODEL 4: DECISION TREE (RANDOM FOREST)
set.seed(1234)
rf.model <- randomForest(as.factor(hit) ~ ., data = train.reg, proximity =
                           TRUE, importance = TRUE)
rf.model
importance(rf.model)
varImpPlot(rf.model)

# Predicting on train set
predTrain <- predict(rf.model, train.reg, type = "class")
# Checking classification accuracy
accuracy.train.rf <-mean(predTrain == train.reg$hit)
rf.train.class.accuracy <- table(predTrain, train.reg$hit)
head(rf.train.class.accuracy)
# Predicting on Test set
predTest <- predict(rf.model, test.reg, type = "class")
# Checking classification accuracy
accuracy.test.rf <- mean(predTest == test.reg$hit)
rf.test.class.accuracy <- table(predTest, test.reg$hit)
head(rf.test.class.accuracy)

#To calculate precision, recall and ROC
rf.predictions.test.prob <- predict(rf.model, test.reg, type = 'prob')[, 2]
rf.predictions.test <- ifelse(rf.predictions.test.prob > 0.5, 1, 0)

results.test$rf <- rf.predictions.test
test.eval.rf <- table(test.reg$hit,results.test$rf)
rownames(test.eval.rf) <- c("Obs. non-hit","Obs. hit")
colnames(test.eval.rf) <- c("Pred. non-hit","Pred. hit")
test.eval.rf

precision.test.rf <- diag(test.eval.rf) / colSums(test.eval.rf)
precision.test.rf
recall.test.rf <- diag(test.eval.rf) / rowSums(test.eval.rf) 
recall.test.rf

roc(y.test, c(rf.predictions.test), plot = TRUE, main = "ROC CURVE", col= "blue")

test.eval.rf <- test.eval.rf / rowSums(test.eval.rf)
test.eval.rf <- as.data.frame(test.eval.rf, stringsAsFactors = TRUE)
test.eval.rf$Var2 <- factor(test.eval.rf$Var2, rev(levels(test.eval.rf$Var2)))

ggplot(test.eval.rf, aes(Var1, Var2, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = scales::percent(Freq))) +
  scale_fill_gradient(low = "white", high = "#3575b5") +
  labs(x = "True", y = "Guess", title = "Confusion matrix for Random Forest",
       fill = "Select")

# MODEL 5: SUPPORT VECTOR MACHINE (SVM) ("linear")
# Fit Support Vector Machine model to data set
dat <- data.frame(x=x, y=as.factor(y.train))
svmfit <- svm(y ~ ., data=dat, kernel="linear", cost=10, scale=FALSE)
summary(svmfit)

set.seed (1234)
tune.out <- tune(method=svm, y ~ ., data=dat, kernel="linear", 
                 ranges=list(cost=c(0.001, 0.01, 0.1, 1, 5, 10, 100)))
summary(tune.out)

bestmod=tune.out$best.model
summary(bestmod)

# Predicting the train set results
svm.pred.train <- predict(bestmod, dat)
summary(svm.pred.train)

xtab.train <- table(predict = svm.pred.train, true = dat$y)
xtab.train

# Compute prediction accuracy
accuracy.train.svm1 <- sum(diag(xtab.train))/sum(xtab.train)
accuracy.train.svm1

# Predicting the Test set results
testdat=data.frame(x=x.test, y=as.factor(y.test))
svm.pred <- predict(bestmod, testdat)
summary(svm.pred)

xtab <- table(predict = svm.pred, true = y.test)
xtab

# Compute test prediction accuracy
accuracy.test.svm1 <- sum(diag(xtab))/sum(xtab)
accuracy.test.svm1



# MODEL 6: SUPPORT VECTOR MACHINE (SVM) ("radial")
# Fit Support Vector Machine model to data set
set.seed (1234)
tune.out2 <- tune(method=svm, y ~ ., data=dat, kernel="radial", 
                 ranges=list(gamma=c(0.0001, 0.001, 0.01, 0.1), cost=c(0.001, 0.01, 0.1, 1)))
summary(tune.out2)

bestmod2=tune.out2$best.model
summary(bestmod2)

plot(tune.out2, train.reg, energy ~ instrumentalness)

# Predicting the train set results
svm.pred2.train <- predict(bestmod2, dat)
summary(svm.pred2.train)

xtab2.train <- table(predict = svm.pred2.train, true = dat$y)
xtab2.train

# Compute prediction accuracy
accuracy.train.svm2 <- sum(diag(xtab2.train))/sum(xtab2.train)
accuracy.train.svm2

# Predicting the Test set results
svm.pred2 <- predict(bestmod2, testdat)
summary(svm.pred2)

xtab2 <- table(predict = svm.pred2, true = testdat$y)
xtab2

# Compute prediction accuracy
accuracy.test.svm2 <- sum(diag(xtab2))/sum(xtab2)
accuracy.test.svm2

test.eval.svm2 <- xtab2 / rowSums(xtab2)
test.eval.svm2 <- as.data.frame(test.eval.svm2, stringsAsFactors = TRUE)
test.eval.svm2$true <- factor(test.eval.svm2$true, rev(levels(test.eval.svm2$true)))

ggplot(test.eval.svm2, aes(predict, true, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = scales::percent(Freq))) +
  scale_fill_gradient(low = "white", high = "#3575b5") +
  labs(x = "True", y = "Guess", title = "Confusion matrix for SVM Rad",
       fill = "Select")


# TABLE OF ACCURACIES
train_accuracy = c(accuracy.train.logit2, accuracy.train.lda, accuracy.train.qda, accuracy.train.lasso, accuracy.train.rf, accuracy.train.svm1, accuracy.train.svm2)
test_accuracy = c(accuracy.test.logit2, accuracy.test.lda, accuracy.test.qda, accuracy.test.lasso, accuracy.test.rf, accuracy.test.svm1, accuracy.test.svm2)

table <- data.frame(Train_Accuracy= train_accuracy,
                 Test_Accuracy=test_accuracy)

table

# SUPERVISED LEARNING WITH PCA
# PRINCIPAL COMPONENT ANALYSIS
pca <- prcomp(x, scale=FALSE, center=FALSE)
summary(pca)

pca.var=pca$sdev ^2
pve=pca.var/sum(pca.var)
pve

screeplot(pca, type = "l", npcs = 15, main = "Screeplot of the first 10 PCs")
abline(h = 1, col="red", lty=5)
legend("topright", legend=c("Eigenvalue = 1"),
       col=c("red"), lty=5, cex=0.6)
cumpro <- cumsum(pca$sdev^2 / sum(pca$sdev^2))
plot(cumpro[0:15], xlab = "PC #", ylab = "Amount of explained variance", main = "Cumulative variance plot")
abline(v = 10, col="blue", lty=5)
abline(h = 0.88759, col="blue", lty=5)
legend("topleft", legend=c("Cut-off @ PC10"),
       col=c("blue"), lty=5, cex=0.6)

fviz_pca_ind(pca, label="none",
             addEllipses=TRUE, ellipse.level=0.95)

fviz_pca_var(pca,
             col.var = "contrib",
             gradient.cols = c("#FFF2CC", "#E7B800", "#CC0000"),
             repel = TRUE)

#New training and test set with PCA selected components
x.train.pca <- tbl_df(pca$x) %>%
  select(PC1:PC10)
x.test.pca <- tbl_df(predict(pca, newdata = x.test)) %>% 
  select(PC1:PC10)

train.reg.pca <- tibble(y.train, x.train.pca)
test.reg.pca <- tibble(y.test, x.test.pca)

x.train.pca <- as.matrix(x.train.pca)
x.test.pca <- as.matrix(x.test.pca)

#MODEL 1: BINARY LOGISTIC REGRESSION
logit.1.pca <- glm(y.train~., family = binomial,data = train.reg.pca)
summary(logit.1.pca)

logit.2.pca <- stepAIC(logit.1.pca)
summary(logit.2.pca)

anova(logit.1.pca, logit.2.pca, test="Chisq")

#Evaluation of in-sample performances
summary(logit.2.pca$fitted.values)
hist(logit.2.pca$fitted.values,main = " Histogram ", col = 'light green')

results.train.pca <- as.data.frame(train.reg.pca)
results.train.pca$logit2.pca <- if_else(logit.2.pca$fitted.values>0.5, 1, 0)

train.eval.logit.pca <- table(train.reg.pca$y.train,results.train.pca$logit2.pca)
rownames(train.eval.logit.pca) <- c("Obs. non-hit","Obs. hit")
colnames(train.eval.logit.pca) <- c("Pred. non-hit","Pred. hit")
train.eval.logit.pca

accuracy.train.logit2.pca <- sum(diag(train.eval.logit.pca))/sum(train.eval.logit.pca)
accuracy.train.logit2.pca

precision.train.logit2.pca <- diag(train.eval.logit.pca) / colSums(train.eval.logit.pca)
recall.train.logit2.pca <- diag(train.eval.logit.pca) / rowSums(train.eval.logit.pca) 
f1.train.logit2.pca <- (2 * precision.train.logit2.pca * recall.train.logit2.pca) / (precision.train.logit2.pca + recall.train.logit2.pca) 

data.frame(precision.train.logit2.pca, recall.train.logit2.pca, f1.train.logit2.pca)

roc(y.train~logit.2.pca$fitted.values, data = train.reg.pca, plot = TRUE, main = "ROC CURVE", col= "blue")
auc(y.train~logit.2.pca$fitted.values, data = train.reg.pca)

#Evaluation of out-of-sample performances
pred.test.logit.pca <- predict(logit.2.pca, test.reg.pca, type="response")

results.test.pca <- as.data.frame(test.reg.pca)
results.test.pca$logit2 <- if_else(pred.test.logit.pca>0.5, 1, 0)

test.eval.logit.pca <- table(test.reg.pca$y.test,results.test.pca$logit2)
rownames(test.eval.logit.pca) <- c("Obs. non-hit","Obs. hit")
colnames(test.eval.logit.pca) <- c("Pred. non-hit","Pred. hit")
test.eval.logit.pca

accuracy.test.logit2.pca <- sum(diag(test.eval.logit.pca))/sum(test.eval.logit.pca)
accuracy.test.logit2.pca

precision.test.logit2.pca <- diag(test.eval.logit.pca) / colSums(test.eval.logit.pca)
recall.test.logit2.pca <- diag(test.eval.logit.pca) / rowSums(test.eval.logit.pca) 
f1.test.logit2.pca <- (2 * precision.test.logit2.pca * recall.test.logit2.pca) / (precision.test.logit2.pca + recall.test.logit2.pca) 

data.frame(precision.test.logit2.pca, recall.test.logit2.pca, f1.test.logit2.pca)

roc(y.test~pred.test.logit.pca, data = test.reg.pca, plot = TRUE, main = "ROC CURVE", col= "blue")
auc(y.test~pred.test.logit.pca, data = test.reg.pca)

#MODEL 2: DISCRIMINANT ANALYSIS
#LINEAR
set.seed(1234)
# Fit the model to train data
lda.model.pca <- lda(y.train~ ., data = train.reg.pca)
lda.model.pca
par("mar")
par(mar=c(1,1,1,1))
plot(lda.model.pca)

# Make predictions of train data
pred.train.lda.pca <- predict(lda.model.pca, train.reg.pca)

head(pred.train.lda.pca$class)
head(pred.train.lda.pca$posterior)
head(pred.train.lda.pca$x)

ldahist(data = pred.train.lda.pca$x[,1], g = train.reg.pca$y.train)

pred.train.lda.class.pca <- predict(lda.model.pca, train.reg.pca)$class
train.eval.lda.pca <- table(Predicted = pred.train.lda.class.pca, Actual = train.reg.pca$y.train)
train.eval.lda.pca

results.train.pca$lda <- pred.train.lda.class.pca

# Model accuracy of train data
accuracy.train.lda.pca <- sum(diag(train.eval.lda.pca))/sum(train.eval.lda.pca)

# Make predictions of test data
pred.test.lda.class.pca <- predict(lda.model.pca, test.reg.pca)$class
test.eval.lda.pca <- table(Predicted = pred.test.lda.class.pca, Actual = test.reg.pca$y.test)
test.eval.lda.pca

results.test.pca$lda <- pred.test.lda.class.pca

# Model accuracy of test data
accuracy.test.lda.pca <- sum(diag(test.eval.lda.pca))/sum(test.eval.lda.pca)

# QUADRATIC
# Fit the model
set.seed(1234)
qda.model.pca <- qda(y.train~ ., data = train.reg.pca)
qda.model.pca

# Make predictions of train data
pred.train.qda.pca <- predict(qda.model.pca, train.reg.pca)

head(pred.train.qda.pca$class)
head(pred.train.qda.pca$posterior)
head(pred.train.qda.pca$x)

pred.train.qda.class.pca <- predict(qda.model.pca, train.reg.pca)$class
train.eval.qda.pca <- table(Predicted = pred.train.qda.class.pca, Actual = train.reg.pca$y.train)
train.eval.qda.pca

results.train.pca$qda <- pred.train.qda.class.pca

# Model accuracy of train data
accuracy.train.qda.pca <- sum(diag(train.eval.qda.pca))/sum(train.eval.qda.pca)

# Make predictions
pred.test.qda.class.pca <- predict(qda.model.pca, test.reg.pca)$class
test.eval.qda.pca <- table(Predicted = pred.test.qda.class.pca, Actual = test.reg.pca$y.test)
test.eval.qda.pca

results.test$qda.pca <- pred.test.qda.class.pca

# Model accuracy of test data
accuracy.test.qda.pca <- sum(diag(test.eval.qda.pca))/sum(test.eval.qda.pca)


# MODEL 3: LASSO REGRESSION
set.seed(1234)

# Setting alpha = 1 implements lasso regression
cv.lasso.reg.pca <- cv.glmnet(x.train.pca, y.train, alpha = 1, family = "binomial",
                          type.measure = "class")
plot(cv.lasso.reg.pca)
# Best
lambda.best.pca <- cv.lasso.reg.pca$lambda.min
lambda.best.pca
lasso.model.pca <- glmnet(x.train.pca, y.train, alpha = 1, lambda = lambda.best.pca,
                      family="binomial")
lasso.coef.pca <- predict(lasso.model.pca, type = "coefficients", s = lambda.best.pca,
                      newx = x.train.pca)
lasso.coef.pca
probabilities.train.pca <- lasso.model.pca %>% predict(newx = x.train.pca, s = lambda.best.pca,
                                               type = "response")
train.lasso.pred.pca <- ifelse(probabilities.train.pca > 0.5, 1, 0)
summary(probabilities.train.pca)
results.train.pca$lasso <- train.lasso.pred.pca
train.eval.lasso.pca <- table(train.reg.pca$y.train,results.train.pca$lasso)
rownames(train.eval.lasso.pca) <- c("Obs. non-hit","Obs. hit")
colnames(train.eval.lasso.pca) <- c("Pred. non-hit","Pred. hit")
train.eval.lasso.pca
accuracy.train.lasso.pca <- sum(diag(train.eval.lasso.pca))/sum(train.eval.lasso.pca)
accuracy.train.lasso.pca
probabilities.test.pca <- lasso.model.pca %>% predict(newx = x.test.pca, s =
                                                lambda.best.pca, type = "response")
test.lasso.pred.pca <- ifelse(probabilities.test.pca > 0.5, 1, 0)
summary(probabilities.test.pca)
results.test.pca$lasso <- test.lasso.pred.pca
test.eval.lasso.pca <- table(test.reg.pca$y.test,results.test.pca$lasso)
rownames(test.eval.lasso.pca) <- c("Obs. non-hit","Obs. hit")
colnames(test.eval.lasso.pca) <- c("Pred. non-hit","Pred. hit")

test.eval.lasso.pca
accuracy.test.lasso.pca <- sum(diag(test.eval.lasso.pca))/sum(test.eval.lasso.pca)
accuracy.test.lasso.pca

# MODEL 4: DECISION TREE (RANDOM FOREST)
set.seed(1234)
rf.model.pca <- randomForest(as.factor(y.train) ~ ., data = train.reg.pca, proximity =
                           TRUE, importance = TRUE)
rf.model.pca
importance(rf.model.pca)
varImpPlot(rf.model.pca)
# Predicting on train set
predTrain.pca <- predict(rf.model.pca, train.reg.pca, type = "class")
# Checking classification accuracy
accuracy.train.rf.pca <-mean(predTrain.pca == train.reg.pca$y.train)
rf.train.class.accuracy.pca <- table(predTrain.pca, train.reg.pca$y.train)
head(rf.train.class.accuracy.pca)
# Predicting on Test set
predTest.pca <- predict(rf.model.pca, test.reg.pca, type = "class")
# Checking classification accuracy
accuracy.test.rf.pca <- mean(predTest.pca == test.reg.pca$y.test)
rf.test.class.accuracy.pca <- table(predTest.pca, test.reg.pca$y.test)
head(rf.test.class.accuracy.pca)


# MODEL 5: SUPPORT VECTOR MACHINE (SVM) ("linear")
# Fit Support Vector Machine model to data set
dat.pca <- data.frame(x=x.train.pca, y=as.factor(y.train))
svmfit.pca <- svm(y ~ ., data=dat.pca, kernel="linear", cost=10, scale=FALSE)
summary(svmfit.pca)

set.seed (1234)
tune.out.pca <- tune(method=svm, y ~ ., data=dat.pca, kernel="linear", 
                 ranges=list(cost=c(0.001, 0.01, 0.1, 1, 5, 10, 100)))
summary(tune.out.pca)

bestmod.pca <- tune.out.pca$best.model
summary(bestmod.pca)

# Predicting the train set results
svm.pred.train.pca <- predict(bestmod.pca, dat.pca)
summary(svm.pred.train.pca)

xtab.train.pca <- table(predict = svm.pred.train.pca, true = dat.pca$y)
xtab.train.pca

# Compute prediction accuracy
accuracy.train.svm1.pca <- sum(diag(xtab.train.pca))/sum(xtab.train.pca)
accuracy.train.svm1.pca

# Predicting the Test set results
testdat.pca <- data.frame(x=x.test.pca, y=as.factor(y.test))
svm.pred.pca <- predict(bestmod.pca, testdat.pca)
summary(svm.pred.pca)

xtab.test.pca <- table(predict = svm.pred.pca, true = testdat.pca$y)
xtab.test.pca

# Compute test prediction accuracy
accuracy.test.svm1.pca <- sum(diag(xtab.test.pca))/sum(xtab.test.pca)
accuracy.test.svm1.pca

# MODEL 6: SUPPORT VECTOR MACHINE (SVM) ("radial")
# Fit Support Vector Machine model to data set
set.seed (1234)
tune.out2.pca <- tune(method=svm, y ~ ., data=dat.pca, kernel="radial", 
                  ranges=list(gamma=c(0.0001, 0.001, 0.01, 0.1), cost=c(0.001, 0.01, 0.1, 1)))
summary(tune.out2.pca)

bestmod2.pca <- tune.out2.pca$best.model
summary(bestmod2.pca)

# Predicting the train set results
svm.pred2.train.pca <- predict(bestmod2.pca, dat.pca)
summary(svm.pred2.train.pca)

xtab2.train.pca <- table(predict = svm.pred2.train.pca, true = dat.pca$y)
xtab2.train.pca

# Compute prediction accuracy
accuracy.train.svm2.pca <- sum(diag(xtab2.train.pca))/sum(xtab2.train.pca)
accuracy.train.svm2.pca

# Predicting the Test set results
svm.pred2.pca <- predict(bestmod2.pca, testdat.pca)
summary(svm.pred2.pca)

xtab2.test.pca <- table(predict = svm.pred2.pca, true = testdat.pca$y)
xtab2.test.pca

# Compute prediction accuracy
accuracy.test.svm2.pca <- sum(diag(xtab2.test.pca))/sum(xtab2.test.pca)
accuracy.test.svm2.pca


# TABLE OF ACCURACIES
train.accuracy.pca = c(accuracy.train.logit2.pca, accuracy.train.lda.pca, accuracy.train.qda.pca, accuracy.train.lasso.pca, accuracy.train.rf.pca, accuracy.train.svm1.pca, accuracy.train.svm2.pca)
test.accuracy.pca = c(accuracy.test.logit2.pca, accuracy.test.lda.pca, accuracy.test.qda.pca, accuracy.test.lasso.pca, accuracy.test.rf.pca, accuracy.test.svm1.pca, accuracy.test.svm2.pca)

table.pca <- data.frame(Train_Accuracy= train.accuracy.pca,
                    Test_Accuracy=test.accuracy.pca)

table.pca
