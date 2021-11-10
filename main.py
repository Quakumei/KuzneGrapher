import dearpygui.dearpygui as dpg
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy import integrate
from math import cos, pi, sin, sqrt, inf

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

import matplotlib.pyplot as plt

GRAPH_FROM = 0
GRAPH_TO = 30
def getFuncXt(k=1, alpha=1):
    X = lambda t, E: pow(E / alpha, 1 / k) * cos(2 * t * pi / getFuncTE(alpha=alpha, k=k)(E))
    return X


# def getFuncTE(k=1, m=1, alpha=1):
#     f = lambda alpha, k, E, m: lambda x: 1 / sqrt((2 / m) * (E - alpha * pow(abs(x), k)))
#     T = lambda E: abs(integrate.quad(f(alpha, k, E, m), -pow(E / alpha, 1 / k), pow(E / alpha, 1 / k))[0]) * 2
#     return T

def getFuncTE(k=1):
    # k = 0 - не существует (E1/0)
    # k = (-0.5;0) - существует
    # k = -1 - существует, но оч странно
    i = lambda k: lambda z: 1/sqrt(1 - pow(abs(z), k))
    I =abs(integrate.quad(i(k), -1, 1 )[0]) if k > 0 else 2*abs(integrate.quad(i(k), 1, inf )[0])
    T = lambda E: I * pow(E, (1/k - 0.5))
    return T, I



def genLinearData(func, a, b, margin_a=0, margin_b=0, step=10):
    x = []
    y = []
    for i in range(a * 1000 + margin_a, (b + 1) * 1000 + margin_b, step):
        x.append(i / 1000)
        y.append(func(i / 1000))
    return x, y


# def genDoubleLinearData(a, b, func=getFuncXt(), margin_a=0, margin_b=0, step=10):
#     XEt_E = []
#     XEt_t = []
#     for E in range(a * 1000 + margin_a, (b + 1) * 1000 + margin_b, step):
#         for t in range(a * 1000 + margin_a, (b + 1) * 1000 + margin_b, step):
#             Earg = E / 1000
#             targ = t / 1000
#             XEt_E.append(Earg)
#             XEt_t.append(targ)
#     XEt_X = [[func(XEt_t[t],XEt_E[E]) for t in range(len(XEt_t))] for E in range(len(XEt_E))]
#     return XEt_X, XEt_E, XEt_t


# Graph T(E)
T_func, I = getFuncTE(k=1)
TE_x, TE_y = genLinearData(T_func, GRAPH_FROM, GRAPH_TO, margin_a=1)

# # Graph X(t,E)
# X_func = getFuncXt()
# XEt_X, XEt_E, XEt_t = genDoubleLinearData(0, 13, func=X_func, margin_a=1, step=1000)


def update_series_te(sender):
    T, I = getFuncTE(k=dpg.get_value(sender))
    TE_x, TE_y = genLinearData(T, GRAPH_FROM, GRAPH_TO, margin_a=1, step=250)
    dpg.set_value('te_series', [TE_x, TE_y])
    dpg.set_value('i_value', "I: " + str(I))

def update_fslider(sender):
    value = dpg.get_item_user_data(sender)
    print(sender)
    dpg.set_value('float_slider', dpg.get_value('float_slider') + value)
    update_series_te('float_slider')



dpg.create_context()

width, height, channels, data = dpg.load_image("output.png")
with dpg.texture_registry(show=True):
    dpg.add_static_texture(width, height, data, tag="3d_graph")

with dpg.window(label="KuzneGrapher", tag="mainwindow", no_title_bar=True, no_scrollbar=True, menubar=False, no_move=True, no_close=True, no_background=True):
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

        dpg.add_image("3d_graph")





    dpg.add_slider_float(label="k", default_value=1, max_value=10, min_value=-10, callback=update_series_te, tag="float_slider")
    with dpg.group(horizontal=True) as group_buttons:
        dpg.add_button(label="-1", callback=update_fslider, user_data=-1)
        dpg.add_button(label="-0.5", callback=update_fslider, user_data=-0.5)
        dpg.add_button(label="-0.1", callback=update_fslider, user_data=-0.1)
        dpg.add_button(label="+0.1", callback=update_fslider, user_data=0.1)
        dpg.add_button(label="+0.5", callback=update_fslider, user_data=0.5)
        dpg.add_button(label="+1", callback=update_fslider, user_data=1)

    dpg.add_text("I: " + str(I), tag='i_value')


    # # x and y axis
    # x = np.array(XEt_t)
    # y = np.array(XEt_E)
    #
    # z = np.array(XEt_X)
    #
    #
    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # ax.plot_surface(x, y, z)
    # ax.set_title('Surface X(t,E)')
    # # fig = plt.figure()
    # # ax = Axes
    # ax.legend()
    #
    # ax.set_xlabel('$t$', fontsize=20)
    # ax.set_ylabel('$E$', fontsize=20)
    # ax.set_zlabel('$X$', fontsize=20)
    # canvas = FigureCanvas(fig)
    # #
    # canvas.print_figure("output.png")

dpg.create_viewport(title='Custom Title', width=800, height=600)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.set_primary_window("mainwindow", True)
dpg.start_dearpygui()
dpg.destroy_context()
