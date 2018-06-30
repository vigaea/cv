# 定义可变参数和定义一个list或tuple参数相比，仅仅在参数前面加了一个 * 号。在函数内部，
# 参数numbers接收到的是一个tuple，因此，函数代码完全不变。但是，调用该函数时，可以传入任意个参数，包括0个参数

def calc(*nums):
    sum = 0
    for n in nums:
        sum = sum + n * n
    return sum

print(calc(1,2,3,4,5,6,7))

