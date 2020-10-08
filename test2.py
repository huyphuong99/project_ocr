import re


def input_raw():
    print('Mat khau:')
    pas = input()
    temp = [x for x in pas.split(',')]
    value = []
    print(temp)
    for i in range(len(temp)):
        if len(temp[i]) < 6 or len(temp[i]) > 12:
            continue

        else:
            pass

        if not re.search("[a-z]", temp[i]):
            continue

        elif not re.search("[0-10]", temp[i]):
            continue

        elif not re.search("[A-Z]", temp[i]):
            continue

        elif not re.search("[#@$]", temp[i]):
            continue

        else:
            pass

        value.append(temp[i])

    print(value)


# input_raw()

def dictionaries():
    alient_0 = {'color': 'green', 'point': 5}
    print(alient_0['color'], " ", alient_0['point'])
    alient_color = alient_0.get('color')
    alient_point = alient_0.get('point', 0) #giong tr
    print(alient_color, " ", alient_point)
    alient_0['x'] = 0
    alient_0['y'] = 25
    alient_0['speed'] = 1.5
    print(alient_0)


dictionaries()
