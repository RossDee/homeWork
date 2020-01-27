#%%
###编写一个程序，读入一个三位整数num，求出num的百位数字、十位数字和个位数字
## //取整数，返回除法的整数部分（商）
## %取余数，返回除法的余数
def readNum(a):
    ##x,百位数，y，十位数，z，个位数
    x = a//100%10
    y = a//10%10
    z = a//1%10
    return(x,y,z)
#%%
readNum(3678)

#%%

##输入鸡和兔的总数量n和腿的总数m，程序判断是否能运算，如能就输出鸡和兔的数量
## 设鸡数量x,兔数量y,n= x+y,m=2x+4y, 消元法后，y=(m-2n)/2
def calCount(n,m):
    ## m为偶数，则返回空
    if m%2!=0:
        chickenCount = 0
        rabbitCount = 0
    else:
        rabbitCount = int((m - 2 * n) / 2)
        chickenCount = int(n - rabbitCount)

    return chickenCount,rabbitCount

#%%
def cal_inputCount():
    totalCount = int(input('请输入鸡和兔的总数量：'))
    totallegCount = int(input('请输入腿的总数量：'))
    calChickenCount,calRabbitCount = calCount(totalCount,totallegCount)
    print("经过计算，鸡的数量:",calChickenCount,"兔的数量:",calRabbitCount)


#%%

##输入一个字符串，去掉重复的字符后，按照字符的ASCII码值从大到小输出
##逐个比对字符串中的字符，然后将不重复的字符返回
def deDuplicationAndQueuebyASCII(x):
    x1 = ''
    for char in x:
        if not char in x1:
            x1 += char
    for charx2 in x1:
        x2 = ord(charx2)
    x3 = sorted(str(x2))
    
    return x3

#%%
x = input('Please input string：')
deDuplicationAndQueuebyASCII(x)





