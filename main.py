import dearpygui.dearpygui as dpg
import numpy as np
from scipy import integrate
from math import cos, pi, sin, sqrt, inf

GRAPH_FROM = 0
GRAPH_TO = 30

def getFuncXt(k=1, alpha=1):
    X = lambda t, E: pow(E / alpha, 1 / k) * cos(2 * t * pi / getFuncTE(alpha=alpha, k=k)(E))
    return X

def getFuncTE(k=1):
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


T_func, I = getFuncTE(k=1)
TE_x, TE_y = genLinearData(T_func, GRAPH_FROM, GRAPH_TO, margin_a=1)


def update_series_te(sender):
    T, I = getFuncTE(k=dpg.get_value(sender))
    TE_x, TE_y = genLinearData(T, GRAPH_FROM, GRAPH_TO, margin_a=1, step=250)
    dpg.set_value('te_series', [TE_x, TE_y])
    dpg.set_value('i_value', "I: " + str(I))


def update_fslider(sender):
    value = dpg.get_item_user_data(sender)
    dpg.set_value('float_slider', dpg.get_value('float_slider') + value)
    update_series_te('float_slider')


dpg.create_context()

width, height, channels, data = dpg.load_image("output.png")
with dpg.texture_registry(show=False):
    dpg.add_static_texture(width, height, data, tag="3d_graph")

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

        #dpg.add_image("3d_graph")

    dpg.add_slider_float(label="k", default_value=1, max_value=10, min_value=-10, callback=update_series_te,
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
