setwd("Z:/联合建模")
library(smbinning)
library(dplyr)
library(foreach)
library(doParallel)
library(ROCR)
library(scorecard)
library(openxlsx)
library(ggplot2)
library(tidyr)
library(caret)

options(scipen = 200)

# 读取模型数据
model_df_all <- read.csv('model_df_0816.csv',stringsAsFactors = FALSE)
sum(is.na(model_df_all$cpc_call_1_cnt),na.rm = TRUE)

df_59_sd <- subset(model_df_all,product_type == 59 & is_patron == 0 & rpd>15)   # 提取首贷秒啦样本
df_35_sd <- subset(model_df_all,product_type == 35 & is_patron == 0 & rpd>15)   # 提取首贷嘉卡样本

# 定义Y变量dpd15
df_59_sd$yvar0 <- ifelse(df_59_sd$dpd15>12,1,0)
df_35_sd$yvar0 <- ifelse(df_35_sd$dpd15>12,1,0)

prop.table(table(df_59_sd$yvar0))
prop.table(table(df_35_sd$yvar0))

# 提取riks变量中的连续型变量名
var_list <- read.xlsx('var_list.xlsx')

model_var <- as.vector(na.omit(var_list[var_list$type=='continuous' & var_list$source=='risk','var']))
# 所要提取的riak中的分类型变量
multiclass_var <- c('cpa_auth_match_1','cpa_auth_match_all',
                    'cpa_last_auth_match','si_is_valid','eb_auth_match','eb_has_app_live_addr',
                    'eb_binding_jingdong','eb_binding_taobao')
model_var <- c(model_var,multiclass_var)

model_var[!(model_var %in% names(model_df_all))]  # 检验是否均在宽表中


# 并行计算
iv_result <- function(data,var_select){
  cl <- makeCluster(8)
  registerDoParallel(cl)
  # 定义计算IV的函数  
  woe_bin <- function(x){
    bins_2var <- woebin(data[data$source_channel %in% c(3,4,5),c('yvar0',x)],y="yvar0",x=x)
    iv <- eval(parse(text=paste0('bins_2var$',x)))$total_iv[1]
    iv_df <- data.frame(var_namne=x,total_iv=iv)
  }
  # 并行计算IV
  data_iv <- foreach(i=names(data[,var_select]), .combine=rbind,
                      .packages=c('scorecard')) %dopar% woe_bin(i)
  
  data_iv <- data_iv[order(data_iv$total_iv,decreasing = TRUE),]
  stopCluster(cl)
  return(data_iv)
}


# 提缺失值值占比小于0.7的变量
summary_var <- function(data,model_var){
  var_name <- c()   # 定义选择的变量
  var_na_pct <- c()   # 缺失值的占比
  var_mean <- c()     # 字段的平均值
  var_std <- c()      # 字段的标准差
  var_pct_0 <- c()    # 字段的0的个数占比
  var_unique_num <- c()   # 不同值个数
  for (x in model_var){
    if ((sum(is.na(data[,x])) / nrow(data)) < 0.7){
      var_name <- c(var_name,x)
      var_na_pct <- c(var_na_pct,(sum(is.na(data[,x])) / nrow(data)))
      var_mean <- c(var_mean,mean(data[,x],na.rm = T))
      var_std <- c(var_std,sd(data[,x],na.rm = T))
      var_pct_0 <- c(var_pct_0,sum(data[,x]==0,na.rm = T) / nrow(data))
      var_unique_num <- c(var_unique_num,unique(data[,x]))
    }
  }
  var_summary <- data.frame(var_name=var_name,var_na_pct=var_na_pct,var_mean=var_mean,var_std=var_std,var_pct_0=var_pct_0)
  return(var_summary)
}

# 输出嘉卡的结果
var_summary_35 <- summary_var(df_35_sd,model_var)
var_select_35 <- as.vector(var_summary_35[,1])
data_iv_35 <- iv_result(df_35_sd,var_select_35)
data_iv_35

# 与表var_summary_35进行连接
iv_select_35 <- data_iv_35 %>% left_join(var_summary_35,by=c('var_namne'='var_name'))
iv_select_35 <- iv_select_35 %>% left_join(var_list,by=c('var_namne'='var'))
write.csv(iv_select_35,'iv_select_35.csv')

woe_bin_result_35 <- data.frame()

for (x in var_select_35){
  bins_2var <- woebin(df_35_sd[df_35_sd$source_channel %in% c(3,4,5),c('yvar0',x)],y="yvar0",x=x)
  woe_bin <- eval(parse(text=paste0('bins_2var$',x)))
  woe_bin_result_35 <- rbind(woe_bin_result_35,woe_bin)
}

woe_bin_result_35 <- woe_bin_result_35 %>% left_join(var_list,by=c('variable'='var'))

write.csv(woe_bin_result_35,'risk_woe_bin_result_35.csv',row.names = F)



# 输出秒啦的结果
var_summary_59 <- summary_var(df_59_sd[df_59_sd$source_channel %in% c(5),],model_var)
var_select_59 <- as.vector(var_summary_59[,1])
data_iv_59 <- iv_result(df_59_sd,var_select_59)
data_iv_59

# 与表var_summary_59进行连接
iv_select_59 <- data_iv_59 %>% left_join(var_summary_59,by=c('var_namne'='var_name'))
iv_select_59 <- iv_select_59 %>% left_join(var_list,by=c('var_namne'='var'))
write.csv(iv_select_59,'risk_iv_select_59_5.csv')



woe_bin_result_59 <- data.frame()

for (x in var_select_59){
  bins_2var <- woebin(df_59_sd[df_59_sd$source_channel %in% c(3,4,5),c('yvar0',x)],y="yvar0",x=x)
  woe_bin <- eval(parse(text=paste0('bins_2var$',x)))
  woe_bin_result_59 <- rbind(woe_bin_result_59,woe_bin)
}

woe_bin_result_59 <- woe_bin_result_59 %>% left_join(var_list,by=c('variable'='var'))

write.csv(woe_bin_result_59,'risk_woe_bin_result_59.csv',row.names = F)












# by_day观察IV 5.15-6.15
# 5.15-5.21
result_by_day_1 <- iv_result(df_59_sd[df_59_sd$add_date>=20180515 & df_59_sd$add_date<=20180521,],var_select_59)


# 5.22-5.28
result_by_day_2 <- iv_result(df_59_sd[df_59_sd$add_date>=20180522 & df_59_sd$add_date<=20180528,],var_select_59)

# 5.29-6.4
result_by_day_3 <- iv_result(df_59_sd[df_59_sd$add_date>=20180529 & df_59_sd$add_date<=20180604,],var_select_59)

# 6.5-6.10
result_by_day_4 <- iv_result(df_59_sd[df_59_sd$add_date>=20180605 & df_59_sd$add_date<=20180610,],var_select_59)

# 6.11-6.15
result_by_day_5 <- iv_result(df_59_sd[df_59_sd$add_date>=20180611 & df_59_sd$add_date<=20180615,],var_select_59)


result_by_day <- result_by_day_1 %>% left_join(result_by_day_2,by=c('var_namne'='var_namne'))
result_by_day <- result_by_day %>% left_join(result_by_day_3,by=c('var_namne'='var_namne'))
result_by_day <- result_by_day %>% left_join(result_by_day_4,by=c('var_namne'='var_namne'))
result_by_day <- result_by_day %>% left_join(result_by_day_5,by=c('var_namne'='var_namne'))



write.csv(result_by_day,'result_by_day.csv',row.names = F)


# 35 by_day观察IV 5.15-6.15
# 5.15-5.21
result_by_day_1 <- iv_result(df_35_sd[df_35_sd$add_date>=20180515 & df_35_sd$add_date<=20180521,],var_select_35)

# 5.22-5.28
result_by_day_2 <- iv_result(df_35_sd[df_35_sd$add_date>=20180522 & df_35_sd$add_date<=20180528,],var_select_35)

# 5.29-6.4
result_by_day_3 <- iv_result(df_35_sd[df_35_sd$add_date>=20180529 & df_35_sd$add_date<=20180604,],var_select_35)

# 6.5-6.10
result_by_day_4 <- iv_result(df_35_sd[df_35_sd$add_date>=20180605 & df_35_sd$add_date<=20180610,],var_select_35)

# 6.11-6.15
result_by_day_5 <- iv_result(df_35_sd[df_35_sd$add_date>=20180611 & df_35_sd$add_date<=20180615,],var_select_35)


result_by_day <- result_by_day_1 %>% left_join(result_by_day_2,by=c('var_namne'='var_namne'))
result_by_day <- result_by_day %>% left_join(result_by_day_3,by=c('var_namne'='var_namne'))
result_by_day <- result_by_day %>% left_join(result_by_day_4,by=c('var_namne'='var_namne'))
result_by_day <- result_by_day %>% left_join(result_by_day_5,by=c('var_namne'='var_namne'))

write.csv(result_by_day,'result_by_day_35.csv',row.names = F)

# 计算秒啦所选变量详细分箱数据
setwd("Z:/联合建模/risk&miguan")
select_var_59 <- read.xlsx('iv_select_59.xlsx')
select_var_59_final <- select_var_59[select_var_59$select_type=='keep','var_namne']  # 选择初步筛选的变量

woe_bin_result <- data.frame()

for (x in select_var_59_final){
  bins_2var <- woebin(df_59_sd[df_59_sd$source_channel %in% c(3,4,5),c('yvar0',x)],y="yvar0",x=x)
  woe_bin <- eval(parse(text=paste0('bins_2var$',x)))
  woe_bin_result <- rbind(woe_bin_result,woe_bin)
}

woe_bin_result <- woe_bin_result %>% left_join(var_list,by=c('variable'='var'))

write.csv(woe_bin_result,'woe_bin_result_59.csv')


# 画图输出
bins <- woebin(df_59_sd[,c('yvar0',select_var_59_final)], y="yvar0",method='tree')
plotlist <- woebin_plot(bins)

for (i in 1:length(plotlist)){
  ggsave(paste0(names(plotlist[i]), ".png"), plotlist[[i]],width = 15, height = 9, units="cm" )
}

# 时间范围5.15-6.15
table(df_59_sd$add_date)

# train
df_59_sd_train <- df_59_sd[df_59_sd$add_date<=20180606,]
nrow(df_59_sd_train) # 19659
prop.table(table(df_59_sd_train$yvar0))  # 0.07955644
# ott
df_59_sd_ott <- df_59_sd[df_59_sd$add_date>20180606,]
nrow(df_59_sd_ott) # 7224
prop.table(table(df_59_sd_ott$yvar0))  # 0.08817829


# 输出训练集的WOE表格
woe_bin_result_train <- data.frame()

for (x in select_var_59_final){
  bins_2var <- woebin(df_59_sd_train[df_59_sd_train$source_channel %in% c(3,4,5),c('yvar0',x)],y="yvar0",x=x)
  woe_bin <- eval(parse(text=paste0('bins_2var$',x)))
  woe_bin_result_train <- rbind(woe_bin_result_train,woe_bin)
}

woe_bin_result_train <- woe_bin_result_train %>% left_join(var_list,by=c('variable'='var'))

write.csv(woe_bin_result_train,'woe_bin_result_59_train.csv')







# 自定义切割点进行分箱
var_name <- 'user_gray.contacts_number_statistic.pct_cnt_to_black'
eval(parse(text=paste0("woebin(df_59_sd_train[,c('yvar0','",var_name,"')],y='yvar0',x='",var_name,"',method='chimerge',
                       breaks_list=list(",var_name,"=c(0.935,0.97,Inf)))")))



# ???ذ?
library(caret)
library(car)
library(smbinning)
library(dplyr)
library(foreach)
library(doParallel)
library(ROCR)

# ??ѵ��???????Լ??ֿ?
set.seed(123)
trainIndex <- createDataPartition(dfTrain$yvar0, p=0.7,list=FALSE, times=1)
df_train <- dfTrain[trainIndex,]     # ѵ��??
(nrow(df_train))  # 23527
df_test  <- dfTrain[-trainIndex,]    # ???Լ?
(nrow(df_test))  # 7842

# ʹ?ò??м???IVֵѵ��??
cl <- makeCluster(4)
registerDoParallel(cl)
woe_data <- function(x){
  data_iv <- smbinning.sumiv(df_train[,c(x,'yvar0')],y="yvar0")
}
data_iv <- foreach(i=names(df_train[,-143]), .combine=rbind,
                   .packages=c('smbinning')) %dopar% woe_data(i)
data_iv <- data_iv[order(data_iv$IV,decreasing = TRUE),]
stopCluster(cl)
data_iv

# ???????Լ?IV
cl <- makeCluster(4)
registerDoParallel(cl)
woe_data <- function(x){
  test_iv <- smbinning.sumiv(df_test[,c(x,'yvar0')],y="yvar0")
}
test_iv <- foreach(i=names(df_test[,-143]), .combine=rbind,
                   .packages=c('smbinning')) %dopar% woe_data(i)
test_iv <- test_iv[order(test_iv$IV,decreasing = TRUE),]
stopCluster(cl)
test_iv

# ????ʱ????????IV
cl <- makeCluster(4)
registerDoParallel(cl)
woe_data <- function(x){
  out_iv <- smbinning.sumiv(dfOut[,c(x,'yvar0')],y="yvar0")
}
out_iv <- foreach(i=names(dfOut[,-143]), .combine=rbind,
                   .packages=c('smbinning')) %dopar% woe_data(i)
out_iv <- out_iv[order(out_iv$IV,decreasing = TRUE),]
stopCluster(cl)
out_iv

train_iv <- data_iv[,1:2]
colnames(train_iv)[2] <- c('train')
colnames(test_iv)[2] <- c('test')
colnames(out_iv)[2] <- c('out')

result_iv <- train_iv %>% left_join(test_iv[,1:2]) %>% left_join(out_iv[,1:2])
write.csv(result_iv,'IV????.csv')


# ????��????????????
iv_sub <- as.vector(data_iv[data_iv$IV>=0.02 & !is.na(data_iv$IV),1]) # ??ȡIV????0.02?ı?��
bin_iv <- data.frame()    # ?????????ݿ?
var_name <- names(df_train[,c('yvar0',iv_var)])
for(i in var_name) {
  if(is.numeric(df_train[,i]) & i != 'yvar0'){
    bin_tbl <- smbinning(df_train, y='yvar0', x= i)
    bin_iv <- rbind(bin_iv, data.frame(bin_tbl$ivtable, variable=i))
  }

  if(is.factor(df_train[,i])){
    bin_tbl <- smbinning.factor(df_train, y='yvar0', x= i)
    bin_iv <- rbind(bin_iv, data.frame(bin_tbl$ivtable, variable=i))
  }
}
write.csv(bin_iv,'bin_iv.csv')     # ?????б?��??WOE IV????????csv





### 5.1 训练集训练模型
fit <- glm(yvar0~.,data=train_data,family=binomial)
summary(fit)
### 5.2 逐步回归模型结果
ml_model <- step(fit)
summary(ml_model)

### 5.3 共线性诊断
library(car)
var_vif <- as.data.frame(vif(ml_model,digits=3))
var_vif$name <- row.names(var_vif)
var_vif
var <- var_vif[var_vif$`vif(ml_model, digits = 3)`<=5,'name']       # 将VIF大于10的变量删除
train_data <- train_data[,c(var,'yvar0')]

### 5.4 再次训练模型
glmodel <- glm(yvar0~.,data=train_data,family = binomial)
summary(glmodel)

### 相关性系数矩阵
cor_data <- as.data.frame(cor(train_data))
write.csv(cor_data,'相关系数矩阵.csv')

#################### 去掉相关系数比较高的变量 ##########
glmodel_fit <- glm(yvar0~WOE_MB_MULTI_PLATS+WOE_DEGREE+WOE_SEARCHED_ORG_CNT
                   +WOE_sum_reject_num_60d+WOE_call_in_dev
                   +WOE_DEVICE_MULTI_CNT+WOE_r_max_calling_cnt
                   +WOE_r_max_call_time+WOE_re_apply_3d
                   +WOE_call_out_cv++WOE_call_in_cv+WOE_sex_man+WOE_r_min_called_time_min
                   +WOE_min_total_amount+WOE_IDCARD_WITH_OTHER_PHONES_CNT,
                   data=train_data,family = binomial)
summary(glmodel_fit)

# 6、模型评估
myKS <- function(pre,label){
  true <- sum(label)  
  false <- length(label)-true  
  tpr <- NULL  
  fpr <- NULL  
  o_pre <- pre[order(pre)] # let the threshold in an order from small to large  
  for (i in o_pre){
    tp <- sum((pre >= i) & label)  
    tpr <- c(tpr,tp/true)  
    fp <- sum((pre >= i) & (1-label))  
    fpr <- c(fpr,fp/false)  
  }  
  plot(o_pre,tpr,type = "l",col= "green",xlab="threshold",ylab="tpr,fpr")  
  lines(o_pre,fpr,type="l", col = "red") 
  KSvalue <- max(tpr-fpr)  
  sub = paste("KS value =",KSvalue)  
  title(sub=sub)  
  cutpoint <- which(tpr-fpr==KSvalue)  
  thre <- o_pre[cutpoint]  
  lines(c(thre,thre),c(fpr[cutpoint],tpr[cutpoint]),col = "blue")  
  cat("KS-value:",KSvalue)
  return(KSvalue)
}
# 训练集
library(pROC)
pred_train <- predict(ml_model, newdata = train_data,type = "response")
modelroc <- roc(train_data$yvar0,pred_train)
plot(modelroc, print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1, 0.2),
     grid.col=c("green", "red"), max.auc.polygon=TRUE,
     auc.polygon.col="skyblue", print.thres=TRUE)
train_ks <- myKS(pred_train, train_data$yvar0) # 0.264787 0.2589663

# 测试集
pred_test <- predict(ml_model, newdata = dfTest,type = "response")
modelroc <- roc(dfTest$yvar0,pred_test)
plot(modelroc, print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1, 0.2),
     grid.col=c("green", "red"), max.auc.polygon=TRUE,
     auc.polygon.col="skyblue", print.thres=TRUE)
test_ks <- myKS(pred_test, dfTest$yvar0) # 0.2565465 0.2435035

# 时间外样本
pred_out <- predict(ml_model, newdata = out_dt,type = "response")
modelroc <- roc(out_dt$yvar0,pred_out)
plot(modelroc, print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1, 0.2),
     grid.col=c("green", "red"), max.auc.polygon=TRUE,
     auc.polygon.col="skyblue", print.thres=TRUE)
out_ks <- myKS(pred_out, out_dt$yvar0)# 0.2223082  0.217747




# 使用ROCR包进行计算
pred <- predict(ml_model, newdata = out_dt,type = "response")
library(ROCR)
t <- prediction(pred, out_dt[, 'yvar0'])
t_roc <- performance(t, 'tpr', 'fpr')
plot(t_roc)
t_auc <- performance(t, 'auc')
t_auc@y.values
title(main = 'ROC Curve')
# 6.3.2 KS value
ks <- max(attr(t_roc, "y.values")[[1]] - (attr(t_roc, "x.values")[[1]])); print(ks)


pred_train <- predict(ml_model, newdata = dfTest,type = "response")
quantile(pred_train,probs=seq(0,1,0.05))
y <- dfTest$yvar0
b <- data.frame(x=pred_train,y=y)
b$p <- cut(pred_train, breaks =as.vector(quantile(pred_train,probs=seq(0,1,0.05))))
write.csv(b,'a.csv')









