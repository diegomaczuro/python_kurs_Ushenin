__author__ = 'Anastasia Bazhutina'
import numpy as np
import matplotlib.pylab as plt
from scipy.integrate import odeint
from matplotlib.widgets import Slider, Button, RadioButtons
from matplotlib.lines import Line2D

class param:
    pass

pr = param()


def Lotka_Volterra_equations(arg, time, params):
    x, y = arg
    alpha, betta, gamma, delta = params
    dxdt = alpha*x - betta*x*y
    dydt = delta*x*y - gamma*y
    return [dxdt, dydt]

fig = plt.figure(1, figsize=(1.8, 8))

pr.alpha = 3.0
pr.betta = 2.5
pr.gamma = 1.6
pr.delta = 2.0
pr.step_time = 300
pr.stop_time = 20
pr.start_time = 0
pr.x0 = 1.0
pr.y0 = 1.0

time = np.linspace(pr.start_time, pr.stop_time, pr.step_time)
yinit = np.array([pr.x0, pr.y0])
params = [pr.alpha, pr.betta, pr.gamma, pr.delta]
solve = odeint(Lotka_Volterra_equations, yinit, time, args=(params,))


sol_x = fig.add_subplot(3, 1, 1)
line1 = Line2D([], [], color='green')
line1.set_data(time, solve[:, 0])
sol_x.add_line(line1)
sol_x.set_xlabel('time')
sol_x.set_ylabel('x(t)')
sol_x.set_xlim(0, 20)
sol_x.set_ylim(0, 1.6)

sol_y = fig.add_subplot(3, 1, 2)
line2 = Line2D([], [], color='blue')
line2.set_data(time, solve[:, 1])
sol_y.add_line(line2)
sol_y.set_xlabel('time')
sol_y.set_ylabel('y(t)')
sol_y.set_xlim(0, 20)
sol_y.set_ylim(0, 1.6)

sol_x_y = fig.add_subplot(3, 1, 3)
line3 = Line2D([], [], color='red')
line3.set_data(solve[:, 0], solve[:, 1])
sol_x_y.add_line(line3)
sol_x_y.set_xlabel('time')
sol_x_y.set_ylabel('y(x)')
sol_x_y.set_xlim(0, 20)
sol_x_y.set_ylim(0, 1.6)
sol_x_y.axis('equal')

plt.tight_layout()


axcolor = 'lightgoldenrodyellow'
alpha_sl = plt.axes([0.057, 0.3, 0.27, 0.02], axisbg=axcolor)
betta_sl = plt.axes([0.057, 0.35, 0.27, 0.02], axisbg=axcolor)
gamma_sl = plt.axes([0.057, 0.40, 0.27, 0.02], axisbg=axcolor)
delta_sl = plt.axes([0.057, 0.45, 0.27, 0.02], axisbg=axcolor)
step_time_sl = plt.axes([0.057, 0.50, 0.27, 0.02], axisbg=axcolor)
start_time_sl = plt.axes([0.057, 0.55, 0.27, 0.02], axisbg=axcolor)
stop_time_sl = plt.axes([0.057, 0.60, 0.27, 0.02], axisbg=axcolor)
x0_sl = plt.axes([0.057, 0.65, 0.27, 0.02], axisbg=axcolor)
y0_sl = plt.axes([0.057, 0.7, 0.27, 0.02], axisbg=axcolor)


pr.alpha_s = Slider(alpha_sl, 'alpha', -10.0, 20.0, valinit=pr.alpha)
pr.betta_s = Slider(betta_sl, 'betta', -10.0, 20.0, valinit=pr.betta)
pr.gamma_s = Slider(gamma_sl, 'gamma', -10.0, 20.0, valinit=pr.gamma)
pr.delta_s = Slider(delta_sl, 'delta', -10.0, 20.0, valinit=pr.delta)
pr.step_time_s = Slider(step_time_sl, 'step_time', pr.stop_time-pr.start_time, 1000.0, valinit=pr.step_time)
pr.start_time_s = Slider(start_time_sl, 'start_time', -10, 100.0, valinit=pr.start_time)
pr.stop_time_s = Slider(stop_time_sl, 'stop_time', -10, 100.0, valinit=pr.stop_time)
pr.x0_s = Slider(x0_sl, 'x0', 0.0, 100.0, valinit=pr.x0)
pr.y0_s = Slider(y0_sl, 'y0', 0.0, 100.0, valinit=pr.y0)

def update_func():
    pr.stop_time = pr.start_time_s.val
    pr.start_time =pr.stop_time_s.val
    time = np.linspace(pr.start_time, pr.stop_time, pr.step_time)
    yinit = np.array([pr.x0, pr.y0])
    params = [pr.alpha, pr.betta, pr.gamma, pr.delta]
    solve = odeint(Lotka_Volterra_equations, yinit, time, args=(params,))

    line1.set_data(time, solve[:, 0])
    sol_x.add_line(line1)
    sol_x.set_xlim(pr.start_time, pr.stop_time)
    sol_x.set_ylim(min(solve[:, 0]), max(solve[:, 0]))

    line2.set_data(time, solve[:, 1])
    sol_y.add_line(line2)
    sol_y.set_xlim(pr.start_time, pr.stop_time)
    sol_y.set_ylim(min(solve[:, 1]), max(solve[:, 1]))

    line3.set_data(solve[:, 0], solve[:, 1])
    sol_x_y.add_line(line3)
    sol_x_y.set_xlim(min(solve[:, 0]), max(solve[:, 0]))
    sol_x_y.set_ylim(min(solve[:, 1])-1, max(solve[:, 1]+1))
    #sol_x_y.axis('equal')
    #sol_x_y.axis('off')
    sol_x_y.margins(0)


def update(val):
    pr.alpha = pr.alpha_s.val
    pr.betta = pr.betta_s.val
    pr.gamma = pr.gamma_s.val
    pr.delta = pr.delta_s.val
    pr.step_time = pr.step_time_s.val

    pr.x0 = pr.x0_s.val
    pr.y0 = pr.y0_s.val
    update_func()


pr.alpha_s.on_changed(update)
pr.betta_s.on_changed(update)
pr.gamma_s.on_changed(update)
pr.delta_s.on_changed(update)
pr.step_time_s.on_changed(update)
pr.x0_s.on_changed(update)
pr.y0_s.on_changed(update)

plt.show()
