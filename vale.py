import os
import csv
from datetime import datetime
import math
from pyproj import Transformer
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.signal as sc_si
import time


# v.10 -> works with pandas dataframes for easier comprehension
#################
# _da means that the values are for the distance above the point
# _p means that the value is for the point (weighted with time spent on the path before and after point)

# import of the gps data
def import_csv(file_name, path):
    file_path = path + "\\" + file_name
    file = []

    with open(file_path, "r") as f:
        read_file = csv.reader(f)

        for row in read_file:
            file.append(row)

    for row in file[1:]:
        row[1] = float(row[1])
        row[2] = float(row[2])
        row[3] = float(row[3])
        row[4] = datetime.strptime(row[4], "%Y-%m-%dT%H:%M:%S")

    return file

# creating boxplot for comparison of data
def box_plot(df, *args):
    for arg in args:
        df.boxplot(column=[arg], labels=[arg])
        plt.show()

# trimming GPS data
def data_trimming(df, *args):
    for arg in args:
        quantiles = df[arg].quantile([0.25, 0.75]).tolist()
        IQR = quantiles[1] - quantiles[0]
        l_range = quantiles[0] - 1.5 * IQR
        u_range = quantiles[1] + 1.5 * IQR

        for index, row in df.iterrows():
            if row[arg] < l_range or row[arg] > u_range:
                df.drop(index, inplace=True)

    df.reset_index(drop=True, inplace=True)

# smoothing velocity
def smoothing_outliers(df, window, *args):
    for arg in args:
        c = 0
        for i in range(1, len(df) - window - 1):
            mean = sum(df[arg][i:i + window]) / window
            quantiles = df[arg][i:i + window].quantile([0.25, 0.75]).tolist()
            IQR = quantiles[1] - quantiles[0]
            l_range = quantiles[0] - 1.5 * IQR
            u_range = quantiles[1] + 1.5 * IQR
            a = 0
            for row in df[arg][i:i + window]:
                if row < l_range or row > u_range:
                    df[arg][i + a] = mean
                    c += 1
                    print(c)

                a += 1
    df.reset_index(drop=True, inplace=True)

# calculating timesteps between GPS points
def calc_timestep_da(df):
    timestep = ["NaN" for x in range(len(df))]

    for i in range(len(df)):

        try:
            timestep[i] = (df["TIMESTAMP"][i] - df["TIMESTAMP"][i - 1]).seconds

        except KeyError:
            pass

    df["TIMESTEP_s_da"] = timestep

# calculating distance between GPS points
def calc_distance_da(df):
    distance = ["NaN" for x in range(len(df))]

    for i in range(len(df)):
        try:
            d_north = df["NORTH"][i] - df["NORTH"][i - 1]
            d_east = df["EAST"][i] - df["EAST"][i - 1]
            d_elevation = df["ELEVATION"][i] - df["ELEVATION"][i - 1]

            distance[i] = math.sqrt(d_north ** 2 + d_east ** 2 + d_elevation ** 2)

        except KeyError:
            pass

    df["DISTANCE_m_da"] = distance


# function to smooth timestep and distance
def smoothing(df, value):
    pd.set_option('mode.chained_assignment', None)
    timestep = "TIMESTEP_s_da"
    distance = "DISTANCE_m_da"

    for i in range(1, len(df)):
        a = 1
        t_step = df[timestep][i]
        dist = df[distance][i]

        while t_step < value:
            try:
                t_step += df[timestep][i + a]
                dist += df[distance][i + a]
                a += 1
            except KeyError:
                break

        for x in range(a):
            df[timestep][i + x] = t_step / a
            df[distance][i + x] = dist / a

    pd.set_option('mode.chained_assignment', "warn")

# savitzky golay filter to smoothe velocity
def sg_smoothing_velocity(df, m, p):
    velocity = df["VELOCITY_kmh_da"].tolist()
    vel_smoothed = sc_si.savgol_filter(velocity[1:len(velocity) - 1], m, p).tolist()
    vel_smoothed.insert(0, "NaN")
    vel_smoothed.append("NaN")
    df["VELOCITY_[km/h]_da_smoothed"] = vel_smoothed

# count outliers outside of IQR, not used
def count_outliers(df, *args):
    for arg in args:
        count = 0
        print(df[arg][1:4])
        quantiles = df[arg][1:len(df) - 1].quantile([0.25, 0.75]).tolist()
        IQR = quantiles[1] - quantiles[0]
        l_range = quantiles[0] - 1.5 * IQR
        u_range = quantiles[1] + 1.5 * IQR

        for index, row in df[1:len(df) - 1].iterrows():

            if row[arg] < l_range or row[arg] > u_range:
                count += 1

        print("OUTLIERS-" + str(arg) + ": " + str(count))

# calculation of velocity
def calc_velocity_da(df):
    velocity = ["NaN" for x in range(len(df))]

    for i in range(len(df)):

        try:
            velocity[i] = df["DISTANCE_m_da"][i] / df["TIMESTEP_s_da"][i] * 3.6

        except (KeyError, TypeError):
            pass

    df["VELOCITY_kmh_da"] = velocity

# calculation of velocity in points, based on distance and time before and after point
def calc_velocity_p(df, timestep):
    velocity_p = ["NaN" for i in range(len(df))]

    for i in range(len(df)):

        try:
            velocity_p[i] = \
                (df["DISTANCE_m_da"][i] + df["DISTANCE_m_da"][i + 1]) / (df[timestep][i] + df[timestep][i + 1])

        except (TypeError, KeyError):
            pass

    velocity_p = [x if x == "NaN" else x * 3.6 for x in velocity_p]
    df["VELOCITY_kmh_p"] = velocity_p

# calculate acceleration based on velocity difference
def calc_acceleration_p(df, velocity, timestep, name):
    acceleration = ["NaN" for x in range(len(df))]

    for i in range(len(df)):

        try:
            acceleration[i] = (df[velocity][i + 1] - df[velocity][i]) / \
                              (3.6 * (df[timestep][i + 1] / 2 + df[timestep][i] / 2))

        except (TypeError, KeyError):
            pass

    df[name] = acceleration

# calculate new timesteps based on smoothed velocity
def calc_timestep_new(df):
    ts_new = ["NaN" for x in range(len(df))]

    for i in range(len(df)):

        try:
            ts_new[i] = df["DISTANCE_m_da"][i] / df["VELOCITY_[km/h]_da_smoothed"][i] * 3.6

        except (KeyError, TypeError):
            pass

    df["TIMESTEP_s_da_smoothed"] = ts_new

# project coordination data so it is compatible with plotly
def project_to_wgs84(df):
    points = [["NaN", "NaN"] for x in range(len(df))]

    for i in range(len(df)):
        points[i] = [df["EAST"][i], df["NORTH"][i]]

    points_tf = []
    tf = Transformer.from_crs(31370, 4326)  # other transformation: 4326 3857

    for pt in tf.itransform(points):
        '{:.3f} {:.3f}'.format(*pt)
        points_tf.append(pt)

    points_tf_df = pd.DataFrame(points_tf, columns=["LATITUDE", "LONGITUDE"])
    df["LATITUDE"], df["LONGITUDE"] = points_tf_df["LATITUDE"], points_tf_df["LONGITUDE"]

# used for graphical representation
def add_color(df, arg):
    color = []
    l_bound = []
    u_bound = []

    # create color coding for EMISSION_da Values
    value = df[arg]
    value = list(set(value))
    value = [x for x in value if not isinstance(x, str)]

    if "velocity" in arg.lower():
        maximum = 150
        minimum = 0

        r_col = (maximum - minimum) / 5
        r_col1 = minimum + r_col
        r_col2 = r_col1 + r_col
        r_col3 = r_col2 + r_col
        r_col4 = r_col3 + r_col
        r_col5 = r_col4 + r_col

    elif "acc" in arg.lower():
        minimum = -3
        maximum = 3

        r_col = (maximum - minimum) / 10
        r_col_l = [minimum + r_col * x for x in range(11)]
        col = ["1green", "2yellowGreen", "3yellow", "4orange", "5red", "6grey"]

        a = 0
        for x in df[arg]:
            try:  # numbers are for plotting order
                if r_col_l[4] < x < r_col_l[6]:
                    color.append(col[0])
                    l_bound.append("between " + str(round(r_col_l[4], 2)))
                    u_bound.append(" and " + str(round(r_col_l[6], 2)))

                elif r_col_l[3] < x < r_col_l[7]:
                    color.append(col[1])
                    l_bound.append("between " + str(round(r_col_l[3], 2)))
                    u_bound.append(" and " + str(round(r_col_l[7], 2)))

                elif r_col_l[2] < x < r_col_l[8]:
                    color.append(col[2])
                    l_bound.append("between " + str(round(r_col_l[2], 2)))
                    u_bound.append(" and " + str(round(r_col_l[8], 2)))

                elif r_col_l[1] < x < r_col_l[9]:
                    color.append(col[3])
                    l_bound.append("between " + str(round(r_col_l[1], 2)))
                    u_bound.append(" and " + str(round(r_col_l[9], 2)))

                elif r_col_l[0] < x < r_col_l[10]:
                    color.append(col[4])
                    l_bound.append("between " + str(round(r_col_l[0], 2)))
                    u_bound.append(" and " + str(round(r_col_l[10], 2)))

                else:
                    color.append("6skyBlue")
                    l_bound.append("outside" + str(round(r_col_l[0], 2)))
                    u_bound.append("and" + str(round(r_col_l[0], 2)))

            except TypeError:
                color.append("7grey")
                l_bound.append("START")
                u_bound.append("/END")

        df[str(arg) + "_COLOR"] = color
        df[str(arg) + "_lbound"] = l_bound
        df[str(arg) + "_ubound"] = u_bound
        return

    else:
        maximum = max(value)
        minimum = min(value)

        r_col = (maximum - minimum) / 5
        r_col1 = minimum + r_col
        r_col2 = r_col1 + r_col
        r_col3 = r_col2 + r_col
        r_col4 = r_col3 + r_col
        r_col5 = r_col4 + r_col

    if arg == "VELOCITY_[km/h]_da_smoothed":

        for x in df[arg]:

            try:  # numbers are for plotting order

                if x <= r_col1:
                    color.append("1red")
                    l_bound.append(str(round(min(value))))
                    u_bound.append(" - " + str(round(r_col1)))

                elif x <= r_col2:
                    color.append("2orange")
                    l_bound.append(str(round(r_col1)))
                    u_bound.append(" - " + str(round(r_col2)))

                elif x <= r_col3:
                    color.append("3yellow")
                    l_bound.append(str(round(r_col2)))
                    u_bound.append(" - " + str(round(r_col3)))

                elif x <= r_col4:
                    color.append("4yellowGreen")
                    l_bound.append(str(round(r_col3)))
                    u_bound.append(" - " + str(round(r_col4)))

                elif x <= r_col5:
                    color.append("5green")
                    l_bound.append(str(round(r_col4)))
                    u_bound.append(" - " + str(round(r_col5)))

                else:
                    color.append("6grey")
                    l_bound.append("START")
                    u_bound.append("/END")

            except TypeError:
                color.append("6grey")
                l_bound.append("START")
                u_bound.append("/END")

    else:

        for x in df[arg]:

            try:  # numbers are for plotting order

                if x <= r_col1:
                    color.append("5green")
                    l_bound.append(str(round(min(value))))
                    u_bound.append(" - " + str(round(r_col1)))

                elif x <= r_col2:
                    color.append("4yellowGreen")
                    l_bound.append(str(round(r_col1)))
                    u_bound.append(" - " + str(round(r_col2)))

                elif x <= r_col3:
                    color.append("3yellow")
                    l_bound.append(str(round(r_col2)))
                    u_bound.append(" - " + str(round(r_col3)))

                elif x <= r_col4:
                    color.append("2orange")
                    l_bound.append(str(round(r_col3)))
                    u_bound.append(" - " + str(round(r_col4)))

                elif x <= r_col5:
                    color.append("1red")
                    l_bound.append(str(round(r_col4)))
                    u_bound.append(" - " + str(round(r_col5)))

                else:
                    color.append("6grey")
                    l_bound.append("START")
                    u_bound.append("/END")

            except TypeError:
                color.append("6grey")
                l_bound.append("START")
                u_bound.append("/END")

    df[str(arg) + "_COLOR"] = color
    df[str(arg) + "_lbound"] = l_bound
    df[str(arg) + "_ubound"] = u_bound

# plot the maps
def plot_df(df, *argv):
    center_lon = df["LONGITUDE"][len(df) // 2]
    center_lat = df["LATITUDE"][len(df) // 2]

    if len(argv) == 0:
        data = [go.Scattermapbox(
            lon=df["LONGITUDE"],
            lat=df["LATITUDE"],
            mode="lines",
            name="Trajectory",
            line=dict(width=2, color="black"))
        ]

        layout = go.Layout(mapbox={"style": "open-street-map", "zoom": 9,
                                   "center": {"lon": center_lon, "lat": center_lat}},
                           title={'text': "ROUTE", 'x': 0.5, 'xanchor': 'center',
                                  "yanchor": 'top', 'font': dict(size=36)},
                           showlegend=True,
                           autosize=True)

        fig = go.Figure({"data": data, "layout": layout})
        fig.show()

    for arg in argv:
        add_color(df, arg)
        colors = sorted(list(set(list(df[str(arg) + "_COLOR"]))))

        data = [go.Scattermapbox(
            lon=df["LONGITUDE"],
            lat=df["LATITUDE"],
            mode="lines",
            name="Trajectory",
            line=dict(width=2, color="black"))
        ]

        for color in colors:
            df_col = df.loc[df[str(arg) + "_COLOR"] == color]
            df_col.reset_index(drop=True, inplace=True)
            l_bound = str(df_col[str(arg) + "_lbound"][0])
            u_bound = str(df_col[str(arg) + "_ubound"][0])

            data.append(go.Scattermapbox(
                lon=df_col["LONGITUDE"],
                lat=df_col["LATITUDE"],
                mode="markers",
                marker=dict(size=10, color=color[1:]),
                name=l_bound + u_bound,
                hoverinfo="text",
                text=df_col[arg],
                textfont=dict(color="black")))

        layout = go.Layout(mapbox={"style": "open-street-map", "zoom": 9,
                                   "center": {"lon": center_lon, "lat": center_lat}},
                           title={'text': arg, 'x': 0.5, 'xanchor': 'center',
                                  "yanchor": 'top', 'font': dict(size=36)},
                           showlegend=True,
                           autosize=True)

        fig = go.Figure({"data": data, "layout": layout})
        fig.show()

# emission model
def emission_model(df):
    emission = ["NaN" for x in range(len(df))]

    e_0 = 0
    # values for f are for diesel cars and C02
    f = [3.24 * (10 ** (-1)), 8.59 * (10 ** (-2)), 4.96 * (10 ** (-3)), (-5.86) * (10 ** (-2)), 4.48 * (10 ** (-1)),
         2.3 * (10 ** (-1))]

    for i in range(len(df)):
        try:
            emission[i] = max(e_0, f[0] + f[1] * (df["VELOCITY_kmh_p"][i] / 3.6) + f[2] * (
                        df["VELOCITY_kmh_p"][i] / 3.6) ** 2 +

                              f[3] * df["ACCELERATION_ms2_p"][i] + f[4] * df["ACCELERATION_ms2_p"][i] ** 2 +
                              f[5] * df["ACCELERATION_ms2_p"][i] * (df["VELOCITY_kmh_p"][i] / 3.6))

        except TypeError:
            i += 1

    df["EMISSION_[g/s]_p"] = emission

# save new csv file with all the calculated values for evaluation
def save_csv(file_name, path, df):
    file_path = path + "\\" + file_name
    data = [df.columns.tolist()] + df.values.tolist()

    with open(file_path, "w") as f:
        wr = csv.writer(f)
        wr.writerows(data)

# output of values of df for evaluation
def print_df(df, lower, upper):
    print(df[lower:upper].to_string())

# 3d plot for data comparisson (emission model values)
def _3d_plot(df, x, y, z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim3d(0, 70)
    ax.set_ylim3d(-4, 4)
    ax.set_zlim3d(0, 12)
    x = list(df[x])
    y = list(df[y])
    z = list(df[z])

    for i in range(len(x)):
        try:
            x[i] = float(x[i])
            y[i] = float(y[i])
            z[i] = float(z[i])
        except ValueError:
            x.pop(i)
            y.pop(i)
            z.pop(i)

    ax.scatter(x, y, z)
    plt.show()


def main():
    in_path = os.getcwd() + "\\data"
    out_path = os.getcwd() + "\\test"
    file_name = "testdata_1.csv"    # change this value in case of different test csv
    data = import_csv(file_name, in_path)
    df = pd.DataFrame(data[1:], columns=data[0])

    project_to_wgs84(df)
    # box_plot(df, "LATITUDE", "LONGITUDE")
    data_trimming(df, "LATITUDE", "LONGITUDE")  # removing outliers
    calc_timestep_da(df)
    calc_distance_da(df)
    smoothing(df, 10)  # general smoothing timesteps and distances
    calc_velocity_da(df)
    # smoothing timesteps and distance
    # smoothing_outliers(df, 3, "VELOCITY_kmh_da")

    sg_smoothing_velocity(df, 11, 3)  # smoothing velocity (savitzky-golay filter) m=7(window), 3degree
    calc_timestep_new(df)
    calc_acceleration_p(df, "VELOCITY_kmh_da", "TIMESTEP_s_da", "ACCELERATION_ms2_p")
    calc_acceleration_p(df, "VELOCITY_[km/h]_da_smoothed", "TIMESTEP_s_da_smoothed", "ACCELERATION_[m/s^2]_p_smoothed")
    calc_velocity_p(df, "TIMESTEP_s_da_smoothed")
    # count_outliers(df, "ACCELERATION_ms2_p")
    emission_model(df)
    #  _3d_plot(df, "VELOCITY_[km/h]_da_smoothed", "ACCELERATION_[m/s^2]_p_smoothed", "EMISSION_[g/s]_p")
    save_csv("test.csv", out_path, df)
    # print_df(df, 0, 4)
    plot_df(df, "VELOCITY_[km/h]_da_smoothed", "ACCELERATION_[m/s^2]_p_smoothed", "EMISSION_[g/s]_p")


main()
