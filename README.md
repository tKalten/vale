# Vale

Vale is an app developed to calculate emission values, accounting for acceleration and velocity. In this demonstration these values are derived from GPS data.

Currently the output is CO<sub>2</sub> emission measured in [g/s]


## Prerequisites

The following libraries need to be installed:

[os](https://github.com/python/cpython/blob/3.8/Lib/os.py)  
[csv](https://github.com/python/cpython/blob/3.8/Lib/csv.py)  
[datetime](https://github.com/python/cpython/blob/3.8/Lib/datetime.py)  
[pyproj](http://pyproj4.github.io/pyproj/stable/)  
[pandas](https://github.com/pandas-dev/pandas/releases)  
[plotly](https://github.com/plotly/plotly.py)  
[matplotlib](https://github.com/matplotlib/matplotlib)  
[mpl_toolkits.mplot3d](https://matplotlib.org/mpl_toolkits/mplot3d/tutorial.html)  
[scipy](https://github.com/scipy/scipy)  
[time](https://docs.python.org/3/library/time.html)  
[math](https://docs.python.org/3/library/math.html)  


## Execution

To execute the App, simply download the folder and run it in a python interpreter.


### Data

Currently there are two sample GPS files available which were provided to us by Chris Tampère.

In case you want to analyze your own GPS data, the corresponding csv file must be added into the data folder and follow the following format:
```
NAME,NORTH,EAST,ELEVATION,TIMESTAMP
"9446","206769.7218","153150.678","49.060765",2019-09-30T05:16:00
"9447","206738.9178","153147.0632","48.734342",2019-09-30T05:16:01
"9448","206707.87","153143.1845","48.99065",2019-09-30T05:16:02
```
Then simply change the `file_name` variable in the main section to the new one.

In case the calculated values need to be analyzed, the output can be found in `\test\test.csv`.

## Development

The emission model in use is based on the paper [Modelling instantaneous traffic emission and the influence of traffic speed limits](https://www.sciencedirect.com/science/article/pii/S004896970600636X)
by Luc Int Panis, Steven Broekx and Ronghui Liu.

This project was developed by [Matteo Casamenti](@casamenti.matteo) and [Thomas Kaltenleitner](@koidn) as part of the course Intelligent Transport Systems (ITS) at KU Leuven.
Special thanks to Chris Tampère, Sajid Raza and Mohammad Ali Arman who supported us in this process.
