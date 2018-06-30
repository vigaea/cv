# name = input('please enter your name:')
# oldS = input('please enter your last score:')
# newS = input('please enter your new score:')
# rate =(float(newS)-float(oldS))/float(oldS)*100
# if oldS < newS:
#     print('Congratulation %s,your score raise %.1f%%.'%(name,rate))
# elif oldS > newS:
#     print('UP up %s,your score drop %.1f%%.'%(name,rate))

name = input('please enter your name:')
oldS = input('please enter your last score:')
newS = input('please enter your new score:')
rate =(float(newS)-float(oldS))/float(oldS)*100
if oldS < newS:
    print('Congratulation {0},your score raise {1:.1f}%'.format(name,rate))
elif oldS > newS:
    print('UP UP {0},your score drop {1:.1f}%'.format(name,rate))