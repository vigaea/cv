# age = 20
# if age > 6:
#     print('kid')
# elif age > 13:
#     print('teenager')

height = 1.75
weight = 80.5
BMI = weight/(height * height)
if BMI <= 18.5:
    print('you\'re too light')
elif 18.5< BMI <= 25:
    print('you\'re normal')
elif 25 < BMI <= 28:
    print('you\'re too heavy')
elif 28 < BMI <= 32:
    print('you\'re fat')
elif 32 < BMI:
    print('you\'re too fat')