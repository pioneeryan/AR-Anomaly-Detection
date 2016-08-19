# The definition of Data, AR model.
# The anomaly detection functions.
# Some demos considering the engine data of aeroplanes.
import math
import numpy
import matplotlib.pyplot as plt


class Data:
    def __init__(self, lst):  # Tested
        self.value = lst
        self.len = len(lst)
        data_sum = 0
        for i in self.value:
            data_sum += i
        self.average = data_sum / self.len

    def __len__(self):  # Tested
        return self.len

    def mean_zero(self):  # Tested
        lst = []
        average = self.average
        for i in self.value:
            lst.append(i-average)
        return Data(lst)

    def differencing(self, times):  # Tested
        lst = self.value
        last = []
        for i in range(times):
            last.append(lst[-1])
            lst = [lst[j+1] - lst[j] for j in range(len(lst) - 1)]
        return lst, last

    def anti_differencing(self, times, last_value):  # Tested
        lst = self.value
        for i in range(times):
            new = []
            start = last_value.pop()
            new.append(start)
            sum = 0
            lst.reverse()
            for j in lst:
                sum += j
                new.append(start - sum)
            new.reverse()
            lst = new
        return lst

# Here the autocovariance divides (n-h) instead of n.
    def autocovariance(self, h):  # Tested
        new = self.mean_zero()
        gamma = 0
        for k in range(new.len - h):
            gamma += new.value[k] * new.value[k + h]
        gamma /= (new.len - h)
        return gamma

# Here h refers to the lag.
    def ACF(self, h):  # Tested
        gamma_0 = self.autocovariance(0)
        gamma_h = self.autocovariance(h)
        return gamma_h / gamma_0

# THe function returns a matrix, with diagonal from \phi_{11} to \phi_{hh}
# Be careful that h should be positive.
    def PACF(self, n):  # Tested
        if n == 1:
            return [[self.ACF(1)]]
        else:
            return self.PACF_iterative(n, self.PACF(n-1))

    def PACF_iterative(self, n, last_one):  # Tested
        old = last_one.copy()
        for i in range(n - 1):
            old[i].append(0)
        newline = []
        for i in range(n):
            newline.append(0)
        old.append(newline)
        rho = [self.ACF(i) for i in range(n + 1)]
        numerator = rho[n]
        for k in range(1, n):
            numerator -= old[n - 2][k - 1] * rho[n - k]
        denominator = 1
        for k in range(1, n):
            denominator -= old[n - 2][k - 1] * rho[k]
        phi_nn = numerator / denominator
        old[n - 1][n - 1] = phi_nn
        for k in range(1, n):
            element = old[n - 2][k - 1] - old[n - 1][n - 1] * old[n - 2][n - k - 1]
            old[n - 1][k - 1] = element
        return old


class AR:
    def __init__(self, coefficient, sigma):  # Tested
        self.order = len(coefficient)
        self.phi = coefficient
        self.noise_variance = sigma

    def __eq__(self, other):  # Tested
        return self.phi == other.phi and self.noise_variance == other.noise_variance

    def psi(self, last):
        if last == 0:
            return [1]
        else:
            lst = [1]
            for i in range(1, last + 1):
                new = 0
                j = 1
                while i - j >= 0:
                    if j > self.order:
                        break
                    new += lst[i-j] * self.phi[j-1]
                    j += 1
                lst.append(new)
            return lst

    def prediction(self, original, time):  # Tested
        assert len(original) >= self.order
        data = Data(original)
        mean = data.average
        data = data.mean_zero()
        now = data.value
        for i in range(time):
            forecast = 0
            for j in range(0, self.order):
                forecast += self.phi[j] * now[-j-1]
            now.append(forecast)
        now = list(map(lambda x: x + mean, now))
        return now[len(original):]

    def prediction_with_differencing(self, original, times, time):
        o = Data(original)
        data = o.differencing(times)
        lst = self.prediction(data[0], time)
        last = data[1]
        for i in range(times):
            new = []
            s = last[i]
            for j in range(time):
                s += lst[j]
                new.append(s)
            lst = new
        return lst

    def prediction_error(self, time):
        lst = []
        s = 0
        psi = self.psi(time)
        for i in range(time):
            s += self.noise_variance * (psi[i] ** 2)
            lst.append(s)
        return lst

    def modifier(self):  # Tested
        lst = self.phi
        while lst:
            if math.fabs(lst[-1]) < 0.01:
                lst.pop()
            else:
                break
        return AR(lst, self.noise_variance)

# This function returns the position of the anomalies in the list of test data.
    def anomaly_rate(self, original, test):  # Tested
        time = len(test)
        forecast = self.prediction(original, time)
        forecast_error = self.prediction_error(time)
        forecast_error_rate = [math.fabs(forecast[i] - test[i]) / math.sqrt(math.fabs(forecast_error[i])) for i in range(time)]
        return forecast_error_rate

    # Only normal attributes are used for future prediction. Anomalies are replaced by prediction of previous ones.
    # Predictions and error rates are given.
    def prediction_advanced(self, training, test, used_threshold=10):
        data = Data(training)
        mean = data.average
        data = data.mean_zero()
        now_lst = data.value  # Modified training data.
        time = len(test)
        forecast_lst = []
        forecast_error_rate = []
        forecast_error = list(map(lambda x: math.sqrt(math.fabs(x)), self.prediction_error(len(test))))
        forecast_error_lst = []
        last_right = 0
        for i in range(time):
            next_value = 0
            for j in range(0, self.order):
                next_value += self.phi[j] * now_lst[-j-1]
            a = (next_value + mean - test[i]) / forecast_error[last_right]
            forecast_error_rate.append(math.fabs(a))
            forecast_lst.append(next_value + mean)
            forecast_error_lst.append(math.fabs(forecast_error[last_right]))
            if math.fabs(a) > used_threshold:
                now_lst.append(next_value)
                last_right += 1
            else:
                now_lst.append(test[i] - mean)
                last_right = 0
        return forecast_lst, forecast_error_rate, list(map(lambda x: x + mean, now_lst[len(data):])), forecast_error_lst

    def examine(self, training):
        output = []
        for i in range(self.order):
            output.append(0)
        data = Data(training)
        mean = data.average
        data = data.mean_zero()
        now = data.value
        for i in range(self.order, len(now)):
            forecast = 0
            for j in range(0, self.order):
                forecast += self.phi[j] * now[i-j-1]
            output.append(math.fabs(now[i] - forecast))
        return output

    def anomaly(self, original, test):
        time = len(test)
        forecast_error_rate = self.anomaly_rate(original, test)
        anomaly = []
        for i in range(time):
            if forecast_error_rate[i] > 2:
                anomaly.append(i)
        return anomaly


# The length of training data should be at least 10.
# There is a upper bound of the order.
def order_generator(training, start=1, length=7):  # Tested
    data = Data(training)
    n = len(data)
    if start >= n / 20:
        print('AR model does not fit this problem. ')
        return
    max_j = int(n / 10)
    if max_j > 30:
        max_j = 50
    j = 1
    current_PACF = data.PACF(1)
    collector = 0
    while j < max_j:
        pacf = current_PACF[-1][-1]
        if pacf < start / n:
            collector += 1
        else:
            collector = 0
        if collector >= 2:
            break
        j += 1
        current_PACF = data.PACF_iterative(j, current_PACF)
    if j == max_j:
        print('Failed.')
        new_start = start + 1
        return order_generator(training, start=new_start)
    else:
        lst = []
        for i in range(j - 2, j - 2 + length):
            if i > 30:
                break
            lst.append(i)
        return tuple(lst)


# Generate models automatically for the data, or generate models of given orders.
def model_generator(training, order=(), length=7):  # Tested
    if not order:
        order = order_generator(training, length=length)
    max_order = order[-1]
    data = Data(training)
    matrix = data.PACF(max_order)
    rho = [data.ACF(i) for i in range(1, max_order+1)]
    acv = data.autocovariance(0)
    model_lst = []
    for i in order:
        parameter = matrix[i-1][:i]
        a = 0
        for j in range(i):
            a += parameter[j] * rho[j]
        a = 1 - a
        sigma = acv * a
        model = AR(parameter, sigma)
        model_lst.append(model)
    return model_lst


# Training data should be longer than 5 + order_estimator.
# Using non-real-time prediction.
def model_estimator(training, input_num=7, output_num=3):  # Tested
    model_lst = model_generator(training, length=input_num)
    error_lst = []
    if len(model_lst) <= output_num:
        return model_lst
    for m in model_lst:
        part_1 = training[0:len(training) - 5]
        part_2 = training[len(training) - 5:len(training)]
        part_2_prediction = m.prediction(part_1, 5)
        error = 0
        # model_error_lst = m.anomaly_rate(part_1, part_2)
        for k in range(len(part_2)):
            error += math.fabs(part_2[k] - part_2_prediction[k])
        error_lst.append(error)
    final_model_lst = []
    copy = sorted(error_lst)
    for i in range(output_num):
        for k in range(len(error_lst)):
            if error_lst[k] <= copy[i]:
                final_model_lst.append(model_lst.pop(k))
                error_lst.pop(k)
                break
    return final_model_lst


# Using real time prediction.
def anomaly_detection_score(original, test, candidate_num=7, used_num=3, threshold=2, used_threshold=5, models=[]):
    if models == []:
        models = model_estimator(original, input_num=candidate_num, output_num=used_num)
    else:
        pass
    time = len(test)
    anomaly_score = []
    for i in range(time):
        anomaly_score.append(0)
    for m in models:
        now = m.prediction_advanced(original, test, used_threshold=used_threshold)[1]
        for i in range(time):
            anomaly_score[i] += math.fabs(now[i])
    standard = threshold * len(models)
    anomaly = []
    for i in range(time):
        if anomaly_score[i] > standard:
            anomaly.append(i)
    return anomaly


# Using real time prediction.
def anomaly_detection_more_than_two(original, test, threshold=2, used_threshold=5):
    models = model_estimator(original)
    time = len(test)
    anomaly_times = []
    anomaly_1 = models[0].prediction_advanced(original, test, used_threshold=used_threshold)[1]
    anomaly_2 = models[1].prediction_advanced(original, test, used_threshold=used_threshold)[1]
    anomaly_3 = models[2].prediction_advanced(original, test, used_threshold=used_threshold)[1]
    for i in range(time):
        score = 0
        if anomaly_1[i] > threshold:
            score += 1
        if anomaly_2[i] > threshold:
            score += 1
        if anomaly_3[i] > threshold:
            score += 1
        anomaly_times.append(score)
    anomaly_lst = []
    for i in range(time):
        if anomaly_times[i] >= 2:
            anomaly_lst.append(i)
    return anomaly_lst


# Using real time prediction.
def anomaly_detection_least(original, test, threshold=2, used_threshold=5, also_largest=False, model_lst=[]):
    if model_lst == []:
        models = model_estimator(original)
    else:
        models = model_lst
    time = len(test)
    anomaly_score = []
    largest_anomaly_score = []
    for m in range(len(models)):
        now = models[m].prediction_advanced(original, test, used_threshold=used_threshold)[1]
        if m == 0:
            for i in range(time):
                anomaly_score.append(now[i])
                largest_anomaly_score.append(now[i])
        else:
            for i in range(time):
                if now[i] < anomaly_score[i]:
                    anomaly_score[i] = now[i]
                if now[i] > largest_anomaly_score[i]:
                    largest_anomaly_score[i] = now[i]
    anomaly_lst = []
    for i in range(time):
        if anomaly_score[i] >= threshold:
            anomaly_lst.append(i)
    if also_largest:
        return anomaly_lst, largest_anomaly_score
    return anomaly_lst


def read_data(filename):  # Tested
    f = open(filename, 'r')
    value = []
    time = []
    for line in f:
        lst = line.split(sep=',')
        value.append(float(lst[0]))
        tpl = (int(lst[1]), int(lst[2]))
        time.append(tpl)
    return value, time


def resample(data, period):
    length = len(data)
    result = []
    for i in range(period):
        result.append([])
    for i in range(length):
        r = i%period
        result[r].append(data[i])
    return tuple(result)


def synthetic(model, start_data, length):  # Tested
    l = length - len(start_data)
    omega = list(numpy.random.normal(0, model.noise_variance, l))
    lst = start_data.copy()
    for i in range(l):
        s = omega[i]
        for j in range(model.order):
            if len(lst) - 1 - j < 0:
                break
            s += model.phi[j] * lst[-1-j]
        lst.append(s)
    return lst


def synthetic_demo():
    print('Here is the demo of synthetic data.')
    n = 1000
    start = [2.5, -2]
    parameter = [1.5, -0.75]
    attempt = synthetic(AR(parameter, 1), start, n)
    training_t = n - 20
    set_anomaly = [4, 5, 11]
    for x in set_anomaly:
        if training_t + x < n:
            attempt[training_t + x] += 0
    c = model_estimator(attempt[:training_t])
    plt.plot(attempt[training_t - 20:])
    for x in c:
        print('Model', x.phi, x.noise_variance)
        print('Result',x.prediction_advanced(attempt[:training_t], attempt[training_t:])[1])
        plt.plot(attempt[training_t - 20:training_t] + x.prediction_advanced(attempt[:training_t], attempt[training_t:])[0])
    r_1 = anomaly_detection_score(attempt[:training_t], attempt[training_t:])
    r_2 = anomaly_detection_more_than_two(attempt[:training_t], attempt[training_t:])
    print('Result 1', r_1)
    print('Result 2', r_2)
    plt.show()
    return


def flight_data_stair_demo_1():
    print('Here is the first demo of real data in a certain stair after resampling.')
    para6 = read_data('para6_108D.csv')[0]
    para6_resample = resample(para6, 16)[3][:200]
    demo_training = para6_resample[:35]
    demo_test = para6_resample[35:45]
    print('Model orders', order_generator(demo_training))
    for demo_model in model_estimator(demo_training):
        print('Model order', demo_model.order)
        print('Model coefficient', demo_model.phi)
        print('Model sigma', demo_model.noise_variance)
        print('Model anomaly rate', demo_model.prediction_advanced(demo_training, demo_test)[1])
    b_0 = anomaly_detection_score(demo_training, demo_test)
    print('Result', b_0)
    plt.plot(demo_test)
    for demo_model in model_estimator(demo_training):
        plt.plot(demo_model.prediction_advanced(demo_training, demo_test)[0])
    plt.show()
    return


def flight_data_stair_demo_2():
    print('Here is the second demo of real data in a certain stair after resampling.')
    para6 = read_data('para6_108D.csv')[0]
    para6_resample = resample(para6, 16)[0][:100]
    para6_resample = [para6_resample[i] + 2 * i for i in range(100)]
    demo_training = para6_resample[:35]
    demo_test = para6_resample[35:45]
    plt.plot(demo_test)
    print('Model orders', order_generator(demo_training))
    for demo_model in model_estimator(demo_training):
        print('Model order', demo_model.order)
        print('Model coefficient', demo_model.phi)
        print('Model sigma', demo_model.noise_variance)
        print('Model anomaly rate', demo_model.prediction_advanced(demo_training, demo_test)[1])
        plt.plot(demo_model.prediction_advanced(demo_training, demo_test)[0])
    b_0 = anomaly_detection_score(demo_training, demo_test)
    print('Result', b_0)
    plt.show()
    return


def flight_data_stair_demo_2_differenced():
    print('Here is the second demo of differenced real data in a certain stair after resampling.')
    para6 = read_data('para6_108D.csv')[0]
    para6_resample = resample(para6, 16)[0][:52]
    para6_resample_diff = Data(para6_resample).differencing(2)[0]
    demo_training = para6_resample_diff[:35]
    demo_test = para6_resample_diff[35:52]
    plt.plot(demo_test)
    for demo_model in model_estimator(demo_training):
        print('Model order', demo_model.order)
        print('Model coefficient', demo_model.phi)
        print('Model sigma', demo_model.noise_variance)
        print('Model anomaly rate', demo_model.prediction_advanced(demo_training, demo_test)[1])
        plt.plot(demo_model.prediction_advanced(demo_training, demo_test)[0])
    b_0 = anomaly_detection_least(demo_training, demo_test)
    print('Result', b_0)
    plt.show()
    return


# flight_data_stair_demo_1()
# flight_data_stair_demo_2()
# flight_data_stair_demo_2_differenced()
# synthetic_demo()

