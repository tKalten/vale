data <- read.csv2("out_data.csv", sep=",", dec=".")

data$velocity_difference <- (data$VELOCITY_kmh_da - data$VELOCITY_.km.h._da_smoothed)
data$acceleration_difference <- data$ACCELERATION_ms2_p - data$ACCELERATION_.m.s.2._p_smoothed

#boxplot(data$ACCELERATION_ms2_p, main="ACCELERATION")

#plot(data$ACCELERATION_ms2_p)

#plot(data$VELOCITY_kmh_da, col="green")
#lines(data$VELOCITY_kmh_da_smoothed, col="red")

boxplot(data$ACCELERATION_ms2_p, data$ACCELERATION_.m.s.2._p_smoothed, main="ACCELERATION")

plot(data$ACCELERATION_ms2_p, col="green")
lines(data$ACCELERATION_.m.s.2._p_smoothed, col="red")

#plot(data$velocity_difference)
#plot(data$acceleration_difference)
