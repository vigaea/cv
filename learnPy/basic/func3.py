import math

def quardratic(a, b, c):
    if not (isinstance(a,(int, float)) & isinstance(b,(int,float)) & isinstance(c,(int, float))):
        raise TypeError('bad operand type')
    delta = b**2 - 4 * a * c
    if delta < 0:
        return 'no answer'
    else:
        x1 = (-b - math.sqrt(delta)) / 2 / a
        x2 = -(-b - math.sqrt(delta)) / 2 / a
        return (x1, x2)

print(quardratic(2,6,2))