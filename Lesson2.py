#%%
##两个非负整数p,q，求最大公约数
# 定义一个函数
def hcf(p, q):
    """该函数返回两个数的最大公约数"""

    # 获取最小值
    if p > q:
        smaller = q
    else:
        smaller = p

    for i in range(1, smaller + 1):
        if ((p % i == 0) and (q % i == 0)):
            hcf = i

    return hcf


# 用户输入两个数字
num1 = int(input("输入第一个数字: "))
num2 = int(input("输入第二个数字: "))

print(num1, "和", num2, "的最大公约数为", hcf(num1, num2))

#%%
import sys
def gcd(x,y):
    if y == 0:
        return x
    else:
         return gcd(y,x%y)
x=int(input("输入第一个数字: "))
y=int(input("输入第二个数字: "))
print(gcd(x,y))
#%%
"""
"""
def gcd2(x,y):
    while y:
        x,y = y,x%y
    return x
x=int(input("输入第一个数字: "))
y=int(input("输入第二个数字: "))
print(gcd2(x,y))
#%%
'''
辗转相减

'''
def gcd3(x,y):
    while x!= y:
        if x > y:
            x = x - y
            return x
        else:
            y = y -x
            return y
x=int(input("输入第一个数字: "))
y=int(input("输入第二个数字: "))
print(gcd3(x,y))


#%%
import matplotlib.pyplot as plt
input_values = [1,2,3,4,5]
squares = [1,4,9,16,25]
plt.plot(input_values,squares,linewidth =5)
plt.title("Square Numbers",fontsize = 24)
plt.xlabel("Value",fontsize = 16)
plt.ylabel("Square of Value",fontsize = 16)
plt.tick_params(axis = 'both',labelsize = 14)
plt.show()



#%%
import numpy as np
import matplotlib.pyplot as plt

x_values = list(range(1,101))
y_values = [x**2 for x in x_values]
plt.scatter(x_values,y_values,c = y_values,cmap = plt.cm.Blues,edgecolors = 'None',s=9 )
plt.title("Square Numbers",fontsize = 24)
plt.xlabel("Value",fontsize = 16)
plt.ylabel("Square of Value",fontsize = 16)
plt.axis([0,110,0,11000])
plt.savefig('square_scatter.png',bbox_inches='tight')
plt.show()

