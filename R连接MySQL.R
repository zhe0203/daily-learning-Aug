library(RMySQL)  # 加载包
library(dplyr)
# 建立连接
con <- dbConnect(MySQL(),host='localhost',port=3306,dbname="mysql",user="root",password="123456")  
summary(con)  # 获取连接的信息
dbGetInfo(con) # 获取数据库信息
dbListTables(con) # 获取数据库中的所有表名
dbRemoveTable(con,'fruits') # 删除数据库中test表

# 将新建的表导入到数据库中
fruits <- data.frame(id=1:5,
                     name=c("apple","banana","lizi","yumi","xigua"),
                     price=c(8.8,4.98,7.8,6,2.1),
                     status=c("0","1","0","2","3"))
dbListTables(con) # 获取当前数据库中所有表name
dbWriteTable(con,"fruits",fruits)  # 将表导入到数据库中
# 再次查看数据库中是否有fruits表，判断导入是否成功
"fruits" %in% dbListTables(con) # TRUE

# 读数据库
dbReadTable(con,'fruits') # 相当于select * from fruits
# 如果出现乱码的问题，这是因为字符编码格式不统一问题
dbSendQuery(con,'SET NAMES utf8') # 

# 写数据，覆盖追加
testA <-data.frame(id=1:6,e=c("a","b","c","d","e","f"),c=c("w","d","s","j","b","d"))
testB <-data.frame(id=7:13,e=c("g","h","i","j","k","l","m"),c=c("q","m","g","n","y","y","y"))
dbRemoveTable(con,'test')

# 创建test表
test <- data.frame(id=c('1'),e=c('1'),c=c('1'),row.names = NULL)  # 首先创建一个新的空的表
test = test[-1,]
dbWriteTable(con,'test',test)           # 然后将其导入，则创建新的数据表

# 直接写testA写入test表中
dbWriteTable(con,'test',testA,append=T,row.names=F)
dbReadTable(con,'test')
dbWriteTable(con,'test',testB,append=T,row.names=T)
dbReadTable(con,'test')

#用SQL语句查询dbGetQuery()和dbSendQuery()两种方法  
# dbGetQuery(con, "SELECT * FROM a")

res <- dbSendQuery(con, "SELECT * FROM a")  
data <- dbFetch(res,n=-1) #取前2条数据，n=-1时是获取所有数据  
for (i in 1:nrow(data)){
  print(data[i,])
}

dbClearResult(res)  
dbDisconnect(con) #断开连接 

# 建立连接
con1 <- dbConnect(MySQL(),host='localhost',port=3306,dbname="mysql",user="root",password="123456") 

res1 <- dbSendQuery(con1,'select a.*,(select count(*) from sheet1$ where num<=a.num and d=a.d) as rownum from sheet1$ as a order by d,rownum asc')
# 获取数据
data1 <- dbFetch(res1,n=-1)
data1

dbClearResult(res1)  # 清除结果
dbDisconnect(con1)  # 断开连接

################### 批量查询 #############################
# 设置这样支持批量查询
con2 <- dbConnect(MySQL(),host="localhost",dbname="mysql",user="root",password="123456",client.flag= CLIENT_MULTI_STATEMENTS)
dbSendQuery(con2,'SET NAMES utf8')
sql = "SELECT * FROM fruits;SELECT * FROM test;SELECT * FROM fruits"
res1 = dbSendQuery(con2,sql)
dbFetch(res1,n=-1) # 获取全部数据

# 对于存在多条查询语句的情况下
num = length(strsplit(sql,';')[[1]])

for (i in 1:(num-1)){
  # print(i)
  if (dbMoreResults(con2)){
    res2 = dbNextResult(con2)
    dbFetch(res2,n=-1)
  }
}
# if (dbMoreResults(con2)){
#   res2 = dbNextResult(con2)
#   dbFetch(res2,n=-1)
# }

dbListResults(con2)
dbClearResult(res1)
dbClearResult(res2)
dbDisconnect(con2)
