import warnings

import dearpygui.dearpygui as dpg
import numpy as np
from scipy import integrate
import scipy

TITLE = "KUZNEGRAPH"
WINDOW_SIZE_WIDTH = 800
WINDOW_SIZE_HEIGHT = 600
GRAPH_MARGIN = 50 / 2

LEFT_PLOT_LABEL = "T(E)"
LEFT_PLOT_FROM = 0
LEFT_PLOT_TO = 100
LEFT_PLOT_STEP = 10  # N of points between integers
LEFT_PLOT_POINTS_BETWEEN_INTEGERS = 1000

RIGHT_PLOT_LABEL = "X(t)"
RIGHT_PLOT_FROM = 0
RIGHT_PLOT_TO = 50
RIGHT_PLOT_STEP = 10  # N of points between integers
RIGHT_PLOT_POINTS_BETWEEN_INTEGERS = 1000

MAX_K = 10.0
MIN_K = -MAX_K
DEFAULT_K = 2.5

PARAM_LIST = ["k", "alpha", "E0", "m"]
PARAM_LIMITS = [(0, 20), (0, 20), (0, 20), (0, 20)]
PARAM_DEFAULTS = [2, 1, 1, 1]
BUTTON_VALUES = [-1, -0.5, -0.1, 0.1, 0.5, 1]

PRECISION = 1


def update_slider(sender):
    """
    Обновляет значение параметра через кнопку
    :param sender: button
    :return: None
    """
    val_change, slider_tag = dpg.get_item_user_data(sender)
    new_value = dpg.get_value(slider_tag) + val_change
    new_value = new_value if new_value >= 0 else 0
    dpg.set_value(slider_tag, new_value)
    update_graphs(slider_tag)


def test_callback_value(sender, userdata=False):
    """
    Выводит в логи значения, полученного в callback функции
    :param sender:
    :return: None
    """
    if userdata:
        print(dpg.get_item_user_data(sender))
    else:
        print(dpg.get_value(sender))


def continue_sine(sine_start: np.ndarray):
    '''
    Красиво оформленный костыль
    :param sine_start:
    :return: sine_y - график с заменёнными nan'ами
    '''

    if not np.isnan(sine_start).any():
        return sine_start
    sine_y = []
    val = 0
    i = 0
    while not np.isnan(val):
        val = sine_start[i]
        i += 1

    #np.array((0,))
    sine = np.concatenate(
        (sine_start[0:i - 1],
         np.negative(sine_start[i - 1:0:-1]),
         np.negative(sine_start[0 : i - 1]),
         sine_start[i - 1:0:-1]))
    print(sine)
    leng = len(sine_start)
    leng_s = len(sine)
    for j in range(0, leng):
        sine_y.append(sine[j % leng_s])

    return sine_y


def update_graphs(sender):
    """
    updates graph after initial values' change
    :param sender: one of the sliders
    :return: None
    """
    k = round(dpg.get_value("k_slider"), PRECISION)
    alpha = round(dpg.get_value("alpha_slider"), PRECISION)
    E0 = round(dpg.get_value("E0_slider"), PRECISION)
    m = round(dpg.get_value("m_slider"), PRECISION)

    # Debug: print values
    print(f"\n==================\nk = {k}\nalpha = {alpha}\nE0 = {E0}\nm = {m}")

    # T(E) graph: E != E0, E is a variable.
    I = lambda k: lambda z: 1 / np.square(1 - np.power(abs(z), k))
    I_val, I_abserr = integrate.quad(I(k), 1e-5, 1)
    dpg.set_value('I_value_text', f"I = {I_val}\t I_abserr = {2 * I_abserr}")
    T = lambda E: np.square(2 * m) / (k * np.square(alpha)) * np.power(E, (1 / k - 1 / 2)) * abs(2 * I_val)

    T_x = []
    T_y = []
    for i in range(LEFT_PLOT_FROM * LEFT_PLOT_POINTS_BETWEEN_INTEGERS,
                   LEFT_PLOT_TO * LEFT_PLOT_POINTS_BETWEEN_INTEGERS,
                   LEFT_PLOT_STEP):
        arg = i / LEFT_PLOT_POINTS_BETWEEN_INTEGERS
        T_x.append(arg)
        T_y.append(T(arg))
    dpg.set_value('TE_series', (T_x, T_y))
    dpg.fit_axis_data('TE_y_axis')
    dpg.fit_axis_data('TE_x_axis')

    # X(t) graph:
    # solve X(t)'' + 2k * (X(t))^(k-1) = 0, x(0) = (E/a)^1/k, x(0)' = 0
    # X>0, build solution
    #
    t2 = np.linspace(RIGHT_PLOT_FROM,
                     RIGHT_PLOT_TO,
                     int(RIGHT_PLOT_POINTS_BETWEEN_INTEGERS * (RIGHT_PLOT_TO - RIGHT_PLOT_FROM)))
    equation = lambda t, y: [y[1], - 2 * k * np.power((y[0]), (k - 1))]
    A = E0 / (alpha * k)
    y0 = [A, 0]  # Начальное значение [2,0] означает y (0) = 2, y '(0) = 0
    t_span = (RIGHT_PLOT_FROM, RIGHT_PLOT_TO)
    t = np.arange(RIGHT_PLOT_FROM, RIGHT_PLOT_TO, 0.01)
    # X = scipy.integrate.solve_ivp(equation, t_span, y0)
    X, infodict = scipy.integrate.odeint(equation, y0, t, tfirst=True, full_output=True)

    print(infodict["message"])
    print(X)
    # print(f"Len xs: {len(X_x)}")
    # print(f"Len ys: {len(X_y)}")
    X_x = t
    X_y = X[:, 0]
    X_y = X_y.copy(order="C")
    X_y = continue_sine(X_y)
    # X_x = X.t.copy(order="C")
    # X_y = X.y.copy(order="C")

    dpg.set_value('Xt_series', (X_x, X_y))
    dpg.fit_axis_data('Xt_y_axis')
    dpg.fit_axis_data('Xt_x_axis')


def window_initialize():
    """
    Создание окна DearPyGui, генерация его наполнения
    :return: None
    """
    with dpg.window(
            tag="window_main",
            no_title_bar=True,
            no_scrollbar=True,
            menubar=False,
            no_move=True,
            no_close=True,
            no_background=True
    ):
        with dpg.group(horizontal=True, ) as graphs_group:
            # График зависимости периода от начальной энергии...
            with dpg.plot(
                    label=LEFT_PLOT_LABEL,
                    width=int(WINDOW_SIZE_WIDTH / 2 - GRAPH_MARGIN),
            ):
                # dpg.add_plot_legend()
                dpg.add_plot_axis(dpg.mvXAxis, label="E", tag="TE_x_axis")
                dpg.add_plot_axis(dpg.mvYAxis, label="T", tag="TE_y_axis")
                dpg.add_line_series(
                    [],
                    [],
                    label=LEFT_PLOT_LABEL,
                    parent="TE_y_axis",
                    tag='TE_series'
                )

            # График зависимости координаты от времени...
            with dpg.plot(
                    label=RIGHT_PLOT_LABEL,
                    width=int(WINDOW_SIZE_WIDTH / 2 - GRAPH_MARGIN),
            ):
                # dpg.add_plot_legend()
                dpg.add_plot_axis(dpg.mvXAxis, label="t", tag="Xt_x_axis")
                dpg.add_plot_axis(dpg.mvYAxis, label="X", tag="Xt_y_axis")
                dpg.add_line_series(
                    [],
                    [],
                    label="T(E)",
                    parent="Xt_y_axis",
                    tag='Xt_series'
                )

        # Рассчётная штука
        dpg.add_text("I = ", tag='I_value_text')

        # Слайдеры - k, E_0, alpha,
        # k [-10;10]
        # alpha [-a; a] \ 0
        # m (0; +]
        dpg.add_text("Parameters:")
        with dpg.group() as sliders_group:
            for i in range(len(PARAM_LIST)):
                slider_tag = f"{PARAM_LIST[i]}_slider"
                dpg.add_slider_float(
                    label=PARAM_LIST[i],
                    default_value=PARAM_DEFAULTS[i],
                    min_value=PARAM_LIMITS[i][0],
                    max_value=PARAM_LIMITS[i][1],
                    tag=slider_tag,
                    clamped=False,
                    format=f"%.{PRECISION}f",
                    callback=update_graphs
                )
                with dpg.group(horizontal=True) as group_buttons:
                    for value in BUTTON_VALUES:
                        dpg.add_button(
                            label=(lambda v: '+' + str(value) if (v > 0) else str(value))(value),
                            callback=update_slider,
                            user_data=(value, slider_tag)
                        )


def dpg_initialize():
    """
    Создаёт окно DearPyGui с интерфейсом.
    :return: None
    """
    dpg.create_context()
    window_initialize()
    dpg.create_viewport(
        title=TITLE,
        width=WINDOW_SIZE_WIDTH,
        height=WINDOW_SIZE_HEIGHT)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("window_main", True)
    dpg.start_dearpygui()
    dpg.destroy_context()


if __name__ == "__main__":
    dpg_initialize()
