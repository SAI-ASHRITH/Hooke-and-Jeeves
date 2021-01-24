import sys
import matplotlib.pylab as pl
import matplotlib as plt
import numpy as np

from PIL import Image
from PyQt5.QtWidgets import QApplication, QMainWindow

from Menu import Ui_MainWindow


def fun(x,n):
    if n==1:
        return (x[0] - 2) ** 2 + (x[1] - 2) ** 2
    elif n==2:
        return 0.26*(x[0]**2+x[1]**2)-0.48*x[0]*x[1]
    elif n==3:
        return ((1.0 - x[0])**2) + (100.0 * (x[1] - (x[0]**2))**2)
    elif n==4:
        return ((1.5 - x[0] + x[0]*x[1])**2) + ((2.25 - x[0] + x[0]*x[1]**2)**2) + ((2.625 - x[0] + x[0]*x[1]**3)**2)
    elif n==5:
        return (x[0] + 2 * x[1] - 7)**2 + (2 * x[0] + x[1] - 5)**2


def search(x0, deltax, n, rot):

    x1 = [x0[0] + deltax[0], x0[1]]
    if fun(x0, n) > fun(x1, n):
        x0 = x1[:]
    x2 = [x0[0] - deltax[0], x0[1]]
    if fun(x0, n) > fun(x2, n):
        x0 = x2[:]
    x3 = [x0[0], x0[1] + deltax[1]]
    if fun(x0, n) > fun(x3, n):
        x0 = x3[:]
    x4 = [x0[0], x0[1] - deltax[1]]
    if fun(x0, n) > fun(x4, n):
        x0 = x4[:]
    if rot == 4:
        return x0
    x1 = [x0[0] + deltax[0], x0[1] + deltax[1]]
    if fun(x0, n) > fun(x1, n):
        x0 = x1[:]
    x2 = [x0[0] - deltax[0], x0[1] + deltax[1]]
    if fun(x0, n) > fun(x2, n):
        x0 = x2[:]
    x3 = [x0[0] + deltax[0], x0[1] - deltax[1]]
    if fun(x0, n) > fun(x3, n):
        x0 = x3[:]
    x4 = [x0[0] - deltax[0], x0[1] - deltax[1]]
    if fun(x0, n) > fun(x4, n):
        x0 = x4[:]
    return x0


def plot(points, nu):

    n = 256
    x = np.linspace(-12, 12, n)
    y = np.linspace(-12, 12, n)
    X, Y = np.meshgrid(x, y)

    xs = []
    ys = []

    C = pl.contour(X, Y, fun([X, Y], nu), 8, colors='black')

    for i in range(len(points)):
        xs.append(points[i][0])
        ys.append(points[i][1])

    pl.plot(xs, ys, 'ro', marker='.', color='black')


    plt.pyplot.show()



class mw:
    def __init__(self):
        self.main_win = QMainWindow()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self.main_win)


        self.ui.subBut.clicked.connect(self.subClicked)



    def show(self):
        self.main_win.show()

    def subClicked(self):

        print("test")

        funum = int(self.ui.funNum.text())
        print(funum)
        x0 = [0, 0]
        x0[0] = int(self.ui.xVal.text())
        x0[1] = int(self.ui.yVal.text())
        print(x0)
        deltax = [0, 0]
        deltax[0] = deltax[1] = float(self.ui.delVal.currentText())
        print(deltax)
        epsilon = float(self.ui.epiVal.currentText())
        print(epsilon)
        nc = int(self.ui.nVal.text())
        print(nc)
        rot= int(self.ui.rot.currentText())

        print("Starting Hook-Jeeves...")
        self.ui.op.setText("Starting Hook-Jeeves...")
        x1 = [0, 0]
        points = []

        n = 1

        if funum == 2:
            img=Image.open('C:/Users/ashri/Desktop/test/images/matya.jpg')
        elif funum == 3:
            img=Image.open('C:/Users/ashri/Desktop/test/images/rosen.jpg')
        elif funum == 4:
            img=Image.open('C:/Users/ashri/Desktop/test/images/beale.jpg')
        elif funum == 5:
            img=Image.open('C:/Users/ashri/Desktop/test/images/booth.jpg')

        img.show()

        while deltax[0] > epsilon:
            x1 = search(x0, deltax, funum, rot)

            while fun(x1,funum) < fun(x0,funum):
                print("x%d: fun[%8.5f, %8.5f] = %8.5f" % (n, x1[0], x1[1], fun(x1,funum)))
                self.ui.op.append("x%d: fun[%8.5f, %8.5f] = %8.5f" % (n, x1[0], x1[1], fun(x1,funum)))
                points.append(x1)
                plot(points, funum)
                x2 = [2 * x1[0] - x0[0], 2 * x1[1] - x0[1]]
                print("x2" + str(x2) + "fun(x2)" + str(fun(x2,funum)))
                self.ui.op.append("x2[%8.5f, %8.5f]fun(x2)%8.5f" %(x2[0],x2[1],fun(x2,funum)))


                if fun(x1,funum) < fun(x0,funum):
                    x0 = x1[:]
                    x1 = search(x2, deltax, funum, rot)

                n += 1

                if n>nc:
                    break
            if n>nc:
                break

            deltax[0] /= 2
            deltax[1] /= 2


        self.ui.op.append("END")
        print("END")

        print(points)
        self.ui.op.append(str(points))

        plot(points,funum)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_win = mw()
    main_win.show()
    sys.exit(app.exec_())

