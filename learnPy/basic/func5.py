# 参数定义的顺序必须是：必选参数、默认参数、可变参数、命名关键字参数和关键字参数


def f1(a, b, c=0, *args, **kw):
    print('a =', a, 'b =', b, 'c =', c, 'args =', args, 'kw =', kw)

def f2(a, b, c=0, *, d, **kw):
    print('a =', a, 'b =', b, 'c =', c, 'd =', d, 'kw =', kw)

f1(1,2)
f1(1,2,c=3)
f1(1,2,3,'a','b')
f1(1,2,'a','b')
f1(1,2,3,'a','b',abc=123)
f2(1,2,3,d=4,abc=None)