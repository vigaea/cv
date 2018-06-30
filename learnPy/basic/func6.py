def product(*nums):
    sum = 1
    if nums==():
        raise TypeError('none')
    else:
        for n in nums:
            sum = sum * n
    return sum

print('product(5) =', product(5))
print('product(5, 6) =', product(5, 6))
print('product(5, 6, 7) =', product(5, 6, 7))
print('product(5, 6, 7, 9) =', product(5, 6, 7, 9))
