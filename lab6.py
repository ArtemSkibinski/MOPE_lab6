from random import uniform as rand
import numpy as np
from prettytable import PrettyTable

def lab6(m, N, kk):
    Yi = find_Yi()
    Y = np.sum(Yi, axis=1) / m
    b = []
    for i in range(kk):
        S = 0
        for j in range(N):
            S += (x[j][i] * Y[j]) / N
        b.append(round(S, 3))
    print("Рівняння регресії:")
    print("y = {} + {}*x1 + {}*x2 + {}*x3 + {}*x1x2 + {}*x1x3 + {}*x2x3 + {}*x1x2x3 + {}*x1^2 + {}*x2^2 + {}*x3^3 \n"
          .format(b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8], b[9], b[10]))
    solve_y(b)

    print("Перевірка однорідності дисперсії за критерієм Кохрена:")
    D = []
    for i in range(N):
        Di = sum([(j - Y[i]) ** 2 for j in Yi[i]]) / m
        D.append(round(Di, 3))

    Dmax = max(D)
    Dsum = sum(D)
    Gp = Dmax / Dsum
    print("Коефіцієнт Gp = ", round(Gp, 5))

    f1 = m - 1
    f2 = N
    print("f1 = ", f1)
    print("f2 = ", f2)

    Gtable = {3: 0.4377, 4: 0.3910, 5: 0.3595, 6: 0.3362, 7: 0.3185, 8: 0.3043, 9: 0.2926,
              10: 0.2829, range(11, 17): 0.2462, range(17, 37): 0.2022, range(37, 145): 0.1616}
    Gt = Gtable.get(m)
    print("За таблицею Gt = ", Gt)

    if(Gp < Gt):
        print("Gp < Gt, отже дисперсія однорідна. Критерій Кохрена виконується")

        print("Оцінимо значимість коефіцієнтів регресії згідно критерію Стьюдента:")
        mD = Dsum / N
        Db = mD / (m * N)
        sD = Db ** 0.5
        print("Дисперсія відносності Db = ", round(Db, 3))
        print("sD = ", round(sD, 3))
        print("Оцінка за t-критерієм Стьюдента:")
        t = []
        for i in range(11):
            T = abs(b[i]) / sD
            t.append(round(T, 3))
        f3 = f1 * f2
        print("f3 = ", f3)
        Ttabl = 2.042
        print("За таблицею в 28 рядку Ttabl = ", Ttabl)
        d = 0
        for i in range(kk):
            if (t[i] < Ttabl):
                print("Коефіцієнт b{} є статистично незначущим, виключаємо його з рівняння регресії".format(i))
                b[i] = 0
            else:
                d += 1
                print("Гіпотеза не підтверджується, тобто b{} – значимий коефіцієнт і він залишається в рівнянні "
                      "регресії.".format(i))
        y = solve_y(b)

        print("Перевірка адекватності за критерієм Фішера:")
        print("Кількість значущих коефіцієнтів d = ", d)
        Dad = 0
        for i in range(kk):
            Dad += (m / (N - d)) * ((y[i] - Y[i]) ** 2)
        print("Дисперсія адекватності Dad = ", round(Dad, 3))
        Fp = Dad / Db
        print("Перевірка адекватності Fp = ", round(Fp, 3))
        f4 = N - d
        print("f4 = ", f4)
        Ftable = {1: 4.2, 2: 3.3, 3: 2.9, 4: 2.7, 5: 2.5, 6: 2.4, 12: 2.1, 24: 1.9, "нескінченність": 1.6}
        if(f4 < 7):
            Ft = Ftable.get(f4)
        elif(f4 > 6 and f4 < 13):
            f4 = 12
            Ft = Ftable.get(f4)
        elif(f4 > 12 and f4 < 25):
            f4 = 24
            Ft = Ftable.get(f4)
        else:
            f4 = "нескінченність"
            Ft = Ftable.get(f4)
        print("За таблицею Ft = ", Ft)
        if (Fp < Ft):
            print("Fp < Ft, отримана математична модель адекватна експериментальним даним.")
        else:
            print("Fp > Ft, отже, рівняння регресії неадекватно оригіналу")
            m = 3
            lab6(m, N, kk)
            return

    else:
        print("Gp > Gt, отже дисперсія неоднорідна. Збільшуємо кількість дослідів на 1 ")
        m = m + 1
        lab6(m, N, kk)
        return

def find_Yi():
    print("Згенерована матриця значень Y: ")
    Yi = []
    for j in range(m):
        Yi.append([])
        for i in range(N):
            Yi[j].append(9.1 + 3.9 * X[i][0] + 5.3 * X[i][1] + 4.6 * X[i][2] + 7 * X[i][0] * X[i][1] +
                      1 * X[i][0] * X[i][2] + 5.7 * X[i][1] * X[i][2] + 2.5 * X[i][0] * X[i][1] * X[i][2] +
                      4.8 * (X[i][0] ** 2) + 0.7 * (X[i][1] ** 2) + 3.6 * (X[i][2] ** 2) + rand(0, 10) - 5)
    Yi = np.array(Yi)
    Yi = Yi.swapaxes(0, 1)
    columns = ['Y1', 'Y2', 'Y3', ]
    tableview(Yi, columns)
    return Yi

def solve_y(b):
    print("Значення y з урахуванням значимих коефіцієнтів:")
    y = []
    for i in range(kk):
        y.append(b[0] + b[1] * X[i][0] + b[2] * X[i][1] + b[3] * X[i][2] + b[4] * X[i][0] * X[i][1]
                 + b[5] * X[i][0] * X[i][2] + b[6] * X[i][1] * X[i][2] + b[7] * X[i][0] * X[i][1] * X[i][2]
                 + b[8] *(X[i][0] ** 2) + b[9] * (X[i][1] ** 2) + b[10] * (X[i][2] ** 2))

    print("y1 = {}, y2 = {}, y3 = {}, y4 = {}, y5 = {}, y6 = {}, y7 = {}, y8 = {}, y9 = {}, y10 = {}, y11 = {}\n"
          .format(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10]))
    return y

def tableview(a, col):
    t_data = []
    for row in a:
        for i in row:
            t_data.append(round(i, 3))
    table = PrettyTable(col)
    while t_data:
        table.add_row(t_data[:len(col)])
        t_data = t_data[len(col):]
    print(table)

X1min = 15
X1max = 45
X01 = (X1max + X1min) / 2
dX1 = X1max - X01

X2min = -70
X2max = -10
X02 = (X2max + X2min) / 2
dX2 = X2max - X02

X3min = 15
X3max = 30
X03 = (X3max + X3min) / 2
dX3 = X3max - X03

Ymax = 200 + (X1max + X2max + X3max)/3
Ymin = 200 + (X1min + X2min + X3min)/3
l = 1.73

X = [[X1min, X2min, X3min],
     [X1min, X2min, X3max],
     [X1min, X2max, X3min],
     [X1min, X2max, X3max],
     [X1max, X2min, X3min],
     [X1max, X2min, X3max],
     [X1max, X2max, X3min],
     [X1max, X2max, X3max],
     [(-l) * dX1 + X01, X02, X03],
     [l * dX1 + X01, X02, X03],
     [X01, (-l) * dX2 + X02, X03],
     [X01, l * dX2 + X02, X03],
     [X01, X02, (-l) * dX3 + X03],
     [X01, X02, l * dX3 + X03]]

X1m = X2m = X3m = -1
X1M = X2M = X3M = 1

x = [[1, X1m, X2m, X3m, X1m * X2m, X1m * X3m, X2m * X3m, X1m * X2m * X3m, X1m * X1m, X2m * X2m, X3m * X3m],
     [1, X1m, X2m, X3M, X1m * X2m, X1m * X3M, X2m * X3M, X1m * X2m * X3M, X1m * X1m, X2m * X2m, X3M * X3M],
     [1, X1m, X2M, X3m, X1m * X2M, X1m * X3m, X2M * X3m, X1m * X2M * X3m, X1m * X1m, X2M * X2M, X3m * X3m],
     [1, X1m, X2M, X3M, X1m * X2M, X1m * X3M, X2M * X3M, X1m * X2M * X3M, X1m * X1m, X2M * X2M, X3M * X3M],
     [1, X1M, X2m, X3m, X1M * X2m, X1M * X3m, X2m * X3m, X1M * X2m * X3m, X1M * X1M, X2m * X2m, X3m * X3m],
     [1, X1M, X2m, X3M, X1M * X2m, X1M * X3M, X2m * X3M, X1M * X2m * X3M, X1M * X1M, X2m * X2m, X3M * X3M],
     [1, X1M, X2M, X3m, X1M * X2M, X1M * X3m, X2M * X3m, X1M * X2M * X3m, X1M * X1M, X2M * X2M, X3m * X3m],
     [1, X1M, X2M, X3M, X1M * X2M, X1M * X3M, X2M * X3M, X1M * X2M * X3M, X1M * X1M, X2M * X2M, X3M * X3M],
     [1, -l, 0, 0, 0, 0, 0, 0, (-l)**2, 0, 0],
     [1, l, 0, 0, 0, 0, 0, 0, l**2, 0, 0],
     [1, 0, -l, 0, 0, 0, 0, 0, 0, (-l)**2, 0],
     [1, 0, l, 0, 0, 0, 0, 0, 0, l**2, 0],
     [1, 0, 0, -l, 0, 0, 0, 0, 0, 0, (-l)**2],
     [1, 0, 0, l, 0, 0, 0, 0, 0, 0, l**2]]

m = 3
N = 14
kk = 11

print("Кодовані значення факторів: ")
columns = ['x0', 'x1', 'x2', 'x3', 'x1*x2', 'x1*x3', 'x2*x3', 'x1*x2*x3', 'x1^2', 'x2^2', 'x3^2']
tableview(x, columns)
print("Натуральні значення факторів: ")
columns = ['X1', 'X2', 'X3']
tableview(X, columns)

lab6(m, N, kk)