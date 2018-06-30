def trim(s):
    if s=='':
        return ''
    while s[0]==' ':
        if len(s)==1:
            return ''
        s=s[1:]
    while s[-1]==' ':
         if len(s) == 1:
             return ''
         s=s[:-1]
    return s

if trim('hello  ') != 'hello':
    print('测试失败!')
elif trim('  hello') != 'hello':
    print('测试失败!')
elif trim('  hello  ') != 'hello':
    print('测试失败!')
elif trim('  hello  world  ') != 'hello  world':
    print('测试失败!')
elif trim('') != '':
    print('测试失败!')
elif trim('    ') != '':
    print('测试失败!')
else:
    print('测试成功!')

# L = ['Michael', 'Sarah', 'Tracy', 'Bob', 'Jack']
# print(L[0:-1])
# print(L[:-1])
# print(L[-1])