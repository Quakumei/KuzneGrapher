import dearpygui.dearpygui as dpg
import numpy as np
from scipy import integrate
from math import cos, pi, sin, sqrt, inf
import matplotlib.pyplot as plt

GRAPH_FROM = 0
GRAPH_TO = 50

from scipy.integrate import odeint, solve_bvp, solve_ivp
import scipy


def get_fvdp1(k):
    return lambda t, y: [y[1], - 2 * k * pow((y[0]), (k - 1))]


def solve_second_order_ode(k, E, alpha):
    '''
         Решить ОДУ второго порядка
    '''
    t2 = np.linspace(0, 50, 5000)
    y0 = [E / alpha * (1 / k), 0]  # Условия начального значения
    # Начальное значение [2,0] означает y (0) = 2, y '(0) = 0
    # Возвращаем y, где y [:, 0] - это значение y [0], которое является окончательным решением, а y [:, 1] - это значение y '(x)
    y = scipy.integrate.odeint(get_fvdp1(k), y0, t2, tfirst=True)
    print(y[:, 0][:1000 ])
    return t2, y[:, 0]


def getFuncTE(k=2.0):
    # k = 0 - не существует (E1/0)
    # k = (-0.5;0) - существует
    # k = -1 - существует, но оч странно
    i = lambda k: lambda z: 1 / sqrt(1 - pow(abs(z), k))
    I = abs(integrate.quad(i(k), -1, 1)[0]) if k > 0 else 2 * abs(integrate.quad(i(k), 1, inf)[0])
    T = lambda E: I * pow(E, (1 / k - 0.5))
    return T, I


def genLinearData(func, a, b, margin_a=0, margin_b=0, step=10):
    x = []
    y = []
    for i in range(a * 1000 + margin_a, (b + 1) * 1000 + margin_b, step):
        x.append(i / 1000)
        y.append(func(i / 1000))
    return x, y


E, alpha = 1, 1
T_func, I = getFuncTE(k=2.00)
TE_x, TE_y = genLinearData(T_func, GRAPH_FROM, GRAPH_TO, margin_a=1)
Xt_x, Xt_y = solve_second_order_ode(k=2.00, E=E, alpha=alpha)
Xt_y = Xt_y.copy(order='C')

def update_series_te(sender):
    T, I = getFuncTE(k=dpg.get_value(sender))
    TE_x, TE_y = genLinearData(T, GRAPH_FROM, GRAPH_TO, margin_a=1, step=250)
    Xt_x, Xt_y = solve_second_order_ode(dpg.get_value(sender), E=E, alpha=alpha)
    Xt_y = Xt_y.copy(order='C')
    dpg.set_value('te_series', [TE_x, TE_y])
    dpg.set_value('Xt_series', [Xt_x, Xt_y])
    dpg.set_value('i_value', "I: " + str(I))

def update_fslider(sender):
    value = dpg.get_item_user_data(sender)
    dpg.set_value('float_slider', dpg.get_value('float_slider') + value)
    update_series_te('float_slider')


dpg.create_context()
with dpg.window(label="KuzneGrapher", tag="mainwindow", no_title_bar=True, no_scrollbar=True, menubar=False,
                no_move=True, no_close=True, no_background=True):
    # create plot
    with dpg.group(horizontal=True) as group_buttons:
        with dpg.plot(label="Zavisimost' perioda ot nachal'noj jenergii", height=400, width=400):
            # optionally create legend
            dpg.add_plot_legend()
            # REQUIRED: create x and y axes
            dpg.add_plot_axis(dpg.mvXAxis, label="E")
            dpg.add_plot_axis(dpg.mvYAxis, label="T", tag="y_axis")

            # series belong to a y axis
            dpg.add_line_series(TE_x, TE_y, label="T(E)", parent="y_axis", tag='te_series')

        with dpg.plot(label="Movement graph", height=400, width=400):
            # optionally create legend
            dpg.add_plot_legend()
            # REQUIRED: create x and y axes
            dpg.add_plot_axis(dpg.mvXAxis, label="t")
            dpg.add_plot_axis(dpg.mvYAxis, label="X", tag="y_axis_X")

            # series belong to a y axis
            dpg.add_line_series(Xt_x, Xt_y, label="X(t)", parent="y_axis_X", tag='Xt_series')

        # dpg.add_image("3d_graph")

    dpg.add_slider_float(label="k", default_value=2.00, max_value=10, min_value=-10, callback=update_series_te,
                         tag="float_slider")
    with dpg.group(horizontal=True) as group_buttons:
        dpg.add_button(label="-1", callback=update_fslider, user_data=-1)
        dpg.add_button(label="-0.5", callback=update_fslider, user_data=-0.5)
        dpg.add_button(label="-0.1", callback=update_fslider, user_data=-0.1)
        dpg.add_button(label="+0.1", callback=update_fslider, user_data=0.1)
        dpg.add_button(label="+0.5", callback=update_fslider, user_data=0.5)
        dpg.add_button(label="+1", callback=update_fslider, user_data=1)

    dpg.add_text("I: " + str(I), tag='i_value')

dpg.create_viewport(title='Custom Title', width=800, height=600)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.set_primary_window("mainwindow", True)
dpg.start_dearpygui()
dpg.destroy_context()
