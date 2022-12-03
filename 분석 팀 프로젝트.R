library(dplyr)
library(car)
library(caret)
library(lubridate)
library(lmtest)


daily <- read.csv("USD_KRW_day.csv", header=T, encoding = "UTF-8")
head(daily)
#daily<-daily[,c(1,2,7)]
#colnames(daily)<-c('date','won','var')
daily<-daily[,-6]
colnames(daily)<-c('date','won','start','high','low','var')

daily$won <- gsub(",", "", daily$won)   
daily$start <- gsub(",", "", daily$start)   
daily$high <- gsub(",", "", daily$high)   
daily$low <- gsub(",", "", daily$low)   
daily$var <- gsub("%", "", daily$var)

daily$date<-as.Date(daily$date)
daily<-daily%>%arrange(date)%>%mutate(predate=date-max(date))


daily$won<-as.numeric(daily$won)
daily$start<-as.numeric(daily$start)
daily$high<-as.numeric(daily$high)
daily$low<-as.numeric(daily$low)

daily$var<-as.numeric(daily$var)


summary(daily)
head(daily)

#회귀 분석 실시
y <- daily$won
x <- as.numeric(daily$predate)
x1 <- daily$start
x2 <- daily$high
x3 <- daily$low
df <- data.frame(x,x1, x2,x3, y)
model <- lm(formula = y ~x+x1+x2+x3, data = df)
summary(model)


#다중 공선성(Multicollinearity)문제 확인
vif(model)
sqrt(vif(model))>3
cor(df)
#세 변수간 상관관계가 강하므로 상관관계가 높은 x1,x2,x3변수 제거 x(날짜)변수 한개로 단순회귀분석을 실시

model2 <- lm(formula = y~x, data = df)
summary(model2)
str(model2)
plot((model2$model$y~model2$model$x))
#회귀식 : won=1406+0.7244*days

start_date<-ymd(221115)
ydate <- (ymd(221231)-start_date)
ydate<-data.frame(x=as.numeric(c(1:ydate)))
pred<-predict(model2,newdata=ydate)
str(pred)
plot(pred)
#######################################기본 가정 충족 확인######################
#잔차 독립성 검정(더빈왓슨)
dwtest(model2)
#DW = 2.1918, p-value = 0.9197
#alternative hypothesis: true autocorrelation is greater than 0
#p-value가 0.05 이상이 되어 독립성이 있다고 볼 수 있다.

#등분산성 검정
plot(model2, which = 1)
#점점 분산이 커지는 분포이다

#잔차 정규성 검정
attributes(model)
res <- residuals(model)
shapiro.test(res)
par(mfrow = c(1, 2))
hist(res, freq = F)
qqnorm(res)
#W = 0.97307, p-value = 7.911e-05 <0.05이므로 정규성 만족
res <- residuals(model2)
shapiro.test(res)
par(mfrow = c(1, 2))
hist(res, freq = F)
qqnorm(res)

##############################################################################

library(TTR)
#install.packages("forecast")
library(forecast)
daily$date<-as.Date(daily$date)
daily2<-daily%>%arrange(date)%>%mutate(predate=as.numeric(date-min(date)))
tsdaily <- ts(daily2$won, start=c(1), frequency = 1)

pacf(na.omit(tsdaily), main = "자기 상환함수", col = "red")
plot(diff(tsdaily, differences = 3))
plot(tsdaily, type = "l", col = "red")

par(mfrow = c(1, 2))

arima <- auto.arima(tsdaily)
arima
model <- arima(tsdaily, order = c(0, 1, 0))
model
Box.test(model$residuals, lag = 1, type = "Ljung")#pvalue가 0.69이므로 통계적으로유의하다
model2 <- forecast(model, h = 45) #n= 45(45일=12월31일)까지 예측
model2
plot(model2)


# 2. 위스콘신 유방암 데이터셋을 대상으로 분류기법 2개를 적용하여 기법별 결과를
# 비교하고 시각화하시오. (R과 python 버전으로 모두 실행)
# -종속변수는diagnosis: Benign(양성), Malignancy(악성)

library(mlbench)
library(randomForest)
library(rpart)
library(dplyr)
library(caret)
library(ModelMetrics)
library(e1071)
library(car)
library(nnet)
library(ROCR)
#데이터 호출
df <- read.csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data', header=F)
head(df, 10)
summary(df)
str(df)
df<-df[,-1]
df$V2 <- as.factor(df$V2)

#결측치 확인 및 제거
colSums(is.na(df))
df <- na.omit(df)

# train데이터와 test데이터 2개의 집단으로 분리
# 랜덤으로 훈련 7: 테스트 3의 비율로 분리
set.seed(1)
samples <- sample(nrow(df), 0.7 * nrow(df))
train <- df[samples, ]
test <- df[-samples, ]

table(train$V2)
table(test$V2)


#모델생성

#model_glm <- glm(V2~., data=train2, family="binomial") #로지스틱 회귀분석
model_rf <- randomForest(V2~., data=train, ntree=100) #랜덤포레스트
model_svm <- svm(V2~., data=train)
#model_mnet <- nnet(V2 ~ ., data=train, size = 2)

#pd_glm <- predict(model_glm,newdata=test2, type = "response")
pd_rf<-predict(model_rf,newdata=test, type = "response")
pd_svm<-predict(model_svm,newdata=test)
#pd_mnet<-predict(model_mnet,newdata=test, type="class")

#table(pd_glm,test2$V2)
table(pd_rf,test$V2)
table(pd_svm,test$V2)
#table(pd_mnet,test$V2)
#pd_mnet <- as.factor(pd_mnet)

#F1-Score
caret::confusionMatrix(test$V2, pd_rf)$byClass[7]
caret::confusionMatrix(test$V2, pd_svm)$byClass[7]
#caret::confusionMatrix(test$V2, pd_mnet)$byClass[7]

#Accuracy
caret::confusionMatrix(test$V2, pd_rf)$overall[1]
caret::confusionMatrix(test$V2, pd_svm)$overall[1]
#caret::confusionMatrix(test$V2, pd_mnet)$overall[1]

#Roc auc
auc(test$V2, pd_rf)
auc(test$V2, pd_svm)
#auc(test$V2, pd_mnet)



#############################################################################
# 3. mlbench패키지 내 BostonHousing 데이터셋을 대상으로 예측기법 2개를 적용하여
# 기법별 결과를 비교하고 시각화하시오. (R과 python 버전으로 모두 실행)
# -종속변수는MEDV 또는CMEDV를사용
library(caret)
library(randomForest)
library(mlbench)
library(ModelMetrics)

data("BostonHousing")
df <- BostonHousing
str(df)
summary(df)


#scale
pre_df<-df[,-c(4,14)]
head(pre_df)
summary(pre_df)
pre <- preProcess(pre_df,'range')
pre_df<-predict(pre,pre_df)
pre_df<-cbind(pre_df,df[,c(4,14)])
head(pre_df)
#train, test 셋 나누기
set.seed(1)
idx <- sample(1:nrow(pre_df),nrow(pre_df)*0.7)
train <- pre_df[idx,]
test <- pre_df[-idx,]

########svm도 써보기
m_lm=lm(medv~.,data=train)
m_rf=randomForest(medv~.,data=train,ntree=100,proximity=T)
p_lm=predict(m_lm,test)
p_rf=predict(m_rf,test,type="response")
summary(m_lm)
m_rf


R2(p_lm,test$medv)
rmse(p_lm,test$medv)

R2(p_rf,test$medv)
rmse(p_rf,test$medv)



#################
#4. 아래의 조건을 고려하여 군집분석을 실행하시오.
# (1) 데이터: ggplot2 패키지 내 diamonds 데이터
# (2) philentropy::distance() 함수 내 다양한 거리 계산 방법 중 Euclidian거리를 제외한
# 3개를 이용하여 거리 계산 및 사용된 거리에 대한 설명
# (3) 탐색적 목적의 계층적 군집분석 실행
# (4) 군집수 결정 및 결정 사유 설명
# (5) k-means clustering 실행
# (6) 시각화
# (7) 거리 계산 방법에 따른 결과 차이 비교

library(cluster)
data("diamonds")

diamonds <- na.omit(diamonds)
set.seed(123)
t <- sample(1:nrow(diamonds),20)
df <- diamonds[t,]
dist <- dist(df, method = "manhattan")
dist2 <- dist(df, method = "canberra")
dist3 <- dist(df, method = "minkowski")
hc <- hclust(dist)
hc2 <- hclust(dist2)
hc3 <- hclust(dist3)
par(mfrow=c(1,2))
plot(hc,hang = -1)
rect.hclust(hc, k = 3, border ="red")
plot(hc2,hang = -1)
rect.hclust(hc2, k = 3, border ="red")
plot(hc3,hang = -1)
rect.hclust(hc3, k = 5, border ="red")

agn1 <- agnes(df, metric="manhattan", stand=TRUE)
plot(agn1)
rect.hclust(hc, k = 3, border ="red")


agn2 <- agnes(df, metric="canberra", stand=TRUE)
plot(agn2)
rect.hclust(hc2, k = 10, border ="red")


agn3 <- agnes(df, metric="minkowski", stand=TRUE)
plot(agn1)
rect.hclust(hc3, k = 5, border ="red")


t <- sample(1:nrow(diamonds), 1000)
test <- diamonds[t, ]
mydia <- test[c("price", "carat", "depth", "table")]
head(mydia)

result <- hclust(dist(mydia), method = "average")
result
plot(result, hang = -1)

result2 <- kmeans(mydia, 5)
names(result2)
result2$cluster
mydia$cluster <- result2$cluster
head(mydia)
cor(mydia[ , -5], method = "pearson")
plot(mydia[ , -5])

plot(test[c("price", "carat", "depth", "table")], col=mydia$cluster)

plot(mydia$carat, mydia$price, col = mydia$cluster)
points(result2$centers[ , c("carat", "price")],col = c( 1, 2,3), pch = 8, cex = 5)


#군집수 결정 및 결정 사유 설명
#군집 수에 따른 집단 내 제곱합(within-groups sum of squares)의 그래프
# 
# 군집간의 개체간 거리의 제곱합 : 데이터가 얼마나 뭉쳐져있는지
# 뭉쳐져있는 값이 커서도 안되고 너무 작아서도 안됨, 각 객체마다 적절한 withiness를 가져야하며 tot.withiness의 산
# 점도를 그려 거기서 적절한 중간값을 찾는다.
test_tot <- as.numeric()
for (i in 1:10){
  result <- kmeans(t,i)
  test_tot[i] <- result$tot.withinss
}
plot(c(1:10),test_tot,type='b')
