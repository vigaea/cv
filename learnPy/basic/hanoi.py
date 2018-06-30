def move(n, A, B, C):
    if n == 1:
        print(A,'-->',C)
    else:
        # 将A的前n-1的盘子从A移至B
        move(n-1,A, C, B)
        move(1,A, B, C)
        move(n-1, B, A, C)

# n = int(input('please enter the level of hanoi:'))
move(3, 'A', 'B', 'C')