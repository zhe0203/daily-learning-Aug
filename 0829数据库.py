# -*- coding: utf-8 -*-
import MySQLdb

# 链接数据库
db = MySQLdb.connect('localhost','root','123456','mysql')

# 使用cursor()方法执行SQL语句
cursor = db.cursor()

# 执行sql语句
for i,j,k in zip(range(10),list('ABCDEFGHIJ'),range(10)):
    j = "'" + j + "'"  # 由于NAME为字符串的形式，因此需要将其转换为字符串的格式，带有双引号
    sql = 'INSERT INTO EMPLOYEE(ID,NAME,INCOME) VALUES({0},{1},{2})'.format(i,j,k)
    print sql
    try:
        cursor.execute(sql)   # 执行sql语句
        db.commit()           # 提交到数据库执行
    except:
        db.rollback()

# cursor.execute("select * from proc")
#
# # 使用fetchall()方法来获取全部的数据
# data = cursor.fetchall()
# # 使用循环的方法来返回全部的数据
# for i in data:
#     print i[0],i[1],i[2],i[3]

# 关闭数据库
db.close()

#############################  批量的建立表格 ############################
db = MySQLdb.connect('localhost','root','123456','mysql')
# 使用cursor()方法执行SQL语句
cursor = db.cursor()
# 执行sql语句
cursor.execute('DROP TABLE IF EXISTS PERSON2')
sql = """CREATE TABLE IF NOT EXISTS PERSON2(number INT(11) NOT NULL,name VARCHAR(255),birthday DATE)"""
print sql
cursor.execute(sql)
db.close()
