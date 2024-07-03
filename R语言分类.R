
# 安装和加载必要的包

install.packages("caret")  
#"caret"是一个流行的R软件包，提供了许多工具和函数，用于分类、回归和特征选择等机器学习任务
install.packages("mlbench")
#执行此命令后，R 将会尝试从 CRAN（The Comprehensive R Archive Network）下载并安装 "mlbench" 软件包。这个软件包提供了一些用于机器学习的数据集和函数，可用于练习和演示各种机器学习算法。
library(lattice) #lattice 软件包是一个用于数据可视化和图形呈现的强大工具。它提供了一系列高层次的图形函数，能够方便地创建各种统计图表，如散点图、箱线图、直方图等
library(caret) #caret 软件包提供了一整套用于分类、回归和特征选择等机器学习任务的函数和工具。
library(mlbench)#mlbench 软件包则提供了一些常用的机器学习数据集，这些数据集可以用于演示、测试和评估机器学习算法。这些数据集覆盖了不同类型的机器学习问题，包括分类、回归和聚类。
library(ggplot2)

# 加载数据集
data("PimaIndiansDiabetes2",package='mlbench')  
#从 mlbench 软件包中加载名为 PimaIndiansDiabetes2 的数据集。
install.packages("ipred")  #R会尝试从CRAN（The Comprehensive R Archive Network）下载并安装 "ipred" 软件包。"ipred" 是一个流行的R软件包，提供了一些用于预测建模和评估的工具和函数。
library(ipred)

# 数据预处理
preproc<-preProcess(PimaIndiansDiabetes2[-9],method="bagImpute")
data <- predict(preproc,PimaIndiansDiabetes2[-9])
data$Class <- PimaIndiansDiabetes2[,9]


# 设置训练控制参数
# 使用朴素贝叶斯建模，这里使用了三次10折交叉检验得到30个结果 
fitControl <- trainControl(method = "repeatedcv", number = 10, repeats = 3,returnResamp = "all")
install.packages("klaR")

library(klaR)
library(MASS)

# 训练模型
model1 <- train(Class ~., data=data,method='nb',trControl = fitControl)
resampleHist(model1)
pre <- predict(model1)
confusionMatrix(pre,data$Class)


# 加载数据集
data("iris")
# 设置随机种子以确保结果可复现
set.seed(123)
# 生成随机索引向量
ind <- sample(2, nrow(iris), replace = TRUE, prob = c(0.7, 0.3))
# 根据索引向量分割数据集
trainset <- iris[ind == 1, ]
testset <- iris[ind == 2, ]


# 安装 neuralnet 包
install.packages("neuralnet")
# 加载 neuralnet 包
library(neuralnet)
#neuralnet：这是neuralnet包中的一个函数，用于创建和训练神经网络模型。在这个项目中，创建了一个包含3个隐藏层的神经网络。


#（3）根据数据集在 Species 列取值不同，为训练集新增 versicolor，virginica，setosa 数据列
trainset$setosa <- trainset$Species == "setosa"
trainset$virginica <- trainset$Species == "virginica"
trainset$versicolor <- trainset$Species == "versicolor"

#（4）调用 neuralnet 函数创建一个包含 3 个隐藏层的神经网络，训练结果有可能随机发生变化，所以得到的结果可能不同，可以开始指定 seed 值使得每次训练返回相同的值。
attach(iris)
network <- neuralnet(versicolor + virginica + setosa ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width, trainset, hidden = 3)
network

#（5）输出构建好的神经网络模型的结果矩阵 result.matrix
network$result.matrix
#（6）调用 head 函数，返回 network 模型的犬种第一项。
head(network$generalized.weights[[1]])

#（7）可视化
plot(network)>plot(network)


#3、支持向量机
#（1）安装加载 e1071 库
install.packages("e1071")
library(e1071)
#（2）iris 数据准备
data(iris)
summary(iris)


#（3）建立 SVM 模型。vm()函数在建立支持向量分类机模型的时候有两种建立方式：
#A. 根据既定公式建立模型
x=iris[,-5]       #提取数据中除第 5 列以外的数据作为特征向量
y=iris[,5]        #提取数据中的第 5 列数据作为结果变量

#model<-svm(x,y,kernel="radial",gamma=if(is.vector(x))lelsel/ncol(x)) #建立 svm 模型
# 确保 y 是因子类型
y <- as.factor(y)

# 建立 SVM 模型
model <- svm(Species ~ ., data = iris, kernel = "radial", gamma = 0.25)
summary(model) #结果分析

#B:
gamma=if(is.vector(x))lelsel/ncol(x))


#（4）预测判别
x=iris[,1:4] #确认需要进行预测的样本特征矩阵
pred=predict(model,x) #根据模型 model 对 x 数据进行预测
pred[sample(1:150,8)]
table(pred,y)


#（5）综合建模
attach(iris) #数据集按列单独确认为向量
x=subset(iris,select=-Species) #确认特征变量为数据集 iris 中除去 Species 的其他项
y=Species #确认结果变量为数据集 iris 中的 Species 项
type = c("C-classification", "nuclassification", "one classification") # 确认将要使用的分类方式
#确认将要使用的分类方式
kernel <- c("linear", "polynomial", "radial", "sigmoid") # 确定将要使用的核函数
pred=array(0,dim=c(150,3,4)) #初始化预测结果矩阵的三维长度分别为 150,3,4
accuracy=matrix(0,3,4) #初始化模型精度矩阵的两维分别为 3，4
yy=as.integer(y) #为方便模型精度计算，将结果变量数量化为 1,2,3
for (i in 1:3) {
  for (j in 1:4) {
    model <- svm(x, y, type = type[i], kernel = kernel[j])
    pred[, i, j] <- predict(model, x)
    if (i > 2) {
      accuracy[i, j] <- sum(pred[, i, j] != 1)
    } else {
      accuracy[i, j] <- sum(pred[, i, j] != yy)
    }
  }
}
dimnames(accuracy)=list(type,kernel) #确定模型精度变量的列名和行名

table(pred[,1,3],y)


#（6）可视化分析
plot(cmdscale(dist(iris[, -5])),
     col = c("red", "black", "gray")[as.integer(iris[, 5])],
     pch = c("o", "+")[1:150 %in% model$index + 1])

legend(2, 0.8, c("setosa", "versicolor", "virginica"),
       col = c("red", "black", "gray"), lty = 1)

# 加载数据集
data(iris)
# 利用公式格式建立 SVM 模型
model <- svm(Species ~ ., data = iris)

# 绘制模型类别关于花宽度和长度的分类情况
plot(model, iris, Petal.Width ~ Petal.Length, fill = FALSE, 
     symbolPalette = c("red", "black", "grey"), svSymbol = "+")

# 标记图例
legend(1, 2.5, c("setosa", "versicolor", "virginica"), 
       col = c("red", "black", "gray"), lty = 1)
