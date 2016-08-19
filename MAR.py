# The multivariate AR model.
import math
import numpy
import matplotlib.pyplot as plt
import AR


# lst is the coefficient, where the first term of lst is a list representing the coefficients of the nearest example.
class MAR:
    def __init__(self, lst):
        self.order = len(lst)
        self.dimension = len(lst[0])
        self.parameter_mat = lst

    def __eq__(self, other):
        self.parameter_mat = other.parameter_mat

    # Using the values of all variables in a time period to predict the value of one variable in the next second.
    def prediction_of_one(self, training):
        if len(training[0]) < self.order:
            n = self.order - len(training[0])
            lst = []
            for i in range(self.dimension):
                lst.append([])
            for i in range(self.dimension):
                for j in range(n):
                    lst[i].append(0)
            for i in range(self.dimension):
                lst[i] += training[i]
        else:
            lst = training

        prediction_num = 0
        for j in range(self.order):
            coefficient = self.parameter_mat[j]
            for k in range(self.dimension):
                prediction_num += coefficient[k] * lst[k][-j-1]
        return prediction_num

    def prediction_error_of_one(self, training, test):
        prediction_num = self.prediction_of_one(training)
        return prediction_num - test

    # The sum of square of the prediction errors.
    # Can be used to quantify the performance of a certain AR model.(objective function)
    def error_sum(self, data, target_variable):
        s = 0
        for i in range(self.order, len(data[0])):
            lst = []
            for k in range(self.dimension):
                lst.append(data[k][i - self.order:i])
            s += self.prediction_error_of_one(lst, data[target_variable][i]) ** 2
        return s

    def error_list(self, data_lst, target_variable):
        error_lst = []
        for i in range(self.order, len(data_lst[0])):
            lst = []
            for k in range(self.dimension):
                lst.append(data_lst[k][i - self.order:i])
            error_lst.append(math.fabs(self.prediction_error_of_one(lst, data_lst[target_variable][i])))
        return error_lst

    def anomaly(self, data_lst, target_variable, threshold=0.1):
        error_lst = self.error_list(data_lst, target_variable)
        anomaly_lst = []
        for i in range(len(error_lst)):
            if math.fabs(error_lst[i]) > threshold:
                anomaly_lst.append(i + self.order)
        return anomaly_lst


# From a previous model to gain a new model.
def SGD_iteration(model, data, target_variable, learning_rate, examples):
    gradient_sum = []
    for i in range(model.order):
        gradient_sum.append([])
    for i in range(model.order):
        for j in range(model.dimension):
            new = 0
            for t in examples:
                lst = []
                for k in range(model.dimension):
                    if t >= model.order:
                        lst.append(data[k][t - model.order:t])
                    else:
                        lst.append(data[k][:t])
                if t >= model.order:
                    new += 2 * data[j][t-1-i] * model.prediction_error_of_one(lst, data[target_variable][t])
                else:
                    new += 0
            gradient_sum[i].append(new)
    new = []
    for i in range(model.order):
        lst = [model.parameter_mat[i][j] - learning_rate * gradient_sum[i][j] for j in range(model.dimension)]
        new.append(lst)
    return MAR(new)


# From an initial model to gain the best model.
def SGD_estimation(initial_model, data, target_variable, learning_rate=0.003, least_time=50, most_try=100):
    current_score = initial_model.error_sum(data, target_variable)
    current_model = initial_model
    n = len(data[0])
    a = [numpy.random.randint(current_model.order, n - 1)]
    new_model = SGD_iteration(current_model, data, target_variable, learning_rate, a)
    new_score = new_model.error_sum(data, target_variable)
    times = 0
    if current_score < new_score:
        best_model = current_model
        best_score = current_score
    else:
        best_model = new_model
        best_score = new_score
    while True:
        if new_score < best_score:
            best_model = new_model
            best_score = new_score
        difference = new_score - current_score
        # The condition of break.
        if difference >= - 1 and difference < 0 and times >= least_time:
            break
        times += 1
        print('Iteration', times)
        print('Progress', difference)
        print('Now score', current_score)
        print('Which term', a)
        current_model = new_model
        current_score = new_score
        j = 0
        while True:
            a = [numpy.random.randint(current_model.order, n - 1)]
            new_model = SGD_iteration(current_model, data, target_variable, learning_rate, a)
            new_score = new_model.error_sum(data, target_variable)
            if new_score < current_score:
                break
            j += 1
            if j >= most_try:
                break
        # a = [numpy.random.randint(current_model.order, n - 1)]
        # new_model = SGD_iteration(current_model, data, target_variable, learning_rate, a)
        # new_score = new_model.error_sum(data, target_variable)
    return best_model


def initial_model_generator(order, dimension):
    original = []
    for o in range(order):
        lst = []
        for d in range(dimension):
            lst.append(0)
        original.append(lst)
    return MAR(original)


# The data are pretreated so that they lie in the interval [-1, 1].
def read_file(filename):
    read_data_file = open(filename, 'r')
    data = []
    for line in read_data_file:
        string = line.strip()
        lst = string.split(sep=',')
        number_lst = []
        for i in range(len(lst)):
            lst[i] = lst[i].strip()
        for i in range(1300, len(lst)):
        # for i in range(3000, 4003):
            if isint(lst[i]):
                new = int(lst[i])
                number_lst.append(new)
            elif isfloat(lst[i]):
                new = float(lst[i])
                number_lst.append(new)
            else:
                print('Error')
            if i == 1300:
                max_num = new
                min_num = new
            else:
                if max_num < new:
                    max_num = new
                if min_num > new:
                    min_num = new
        max_abs = 0
        if math.fabs(max_num) > math.fabs(min_num):
            max_abs = math.fabs(max_num)
        else:
            max_abs = math.fabs(min_num)
        data.append(list(map(lambda x: x / max_abs, number_lst)))
    return data


def isfloat(string):  # Tested
    value = re.compile(r'^[-+]?[0-9]+\.[0-9]+$')
    return bool(value.match(string))


def isint(string):
    value = re.compile(r'^[-+]?[0-9]+$')
    return bool(value.match(string))


def differencing(lst):
    result = []
    for i in range(len(lst) - 1):
        result.append(lst[i + 1] - lst[i])
    return result


# In order for the number not to be too large, we gain smaller data from read_file.
def flight_data_multivariate_AR_model_demo_step_1():
    data_lst = read_file('flight_data_final.txt')
    print(data_lst[0])
    print('Input process finished.')
    dim = len(data_lst)
    original = []
    for o in range(5):
        lst = []
        for d in range(dim):
            lst.append(0)
        original.append(lst)
    original_model = MAR(original)
    print('Original model setting process finished.')
    result = SGD_estimation(original_model, data_lst, 141, learning_rate=0.002, least_time=100, most_try=5)
    mat = result.parameter_mat
    for i in range(len(mat)):
        for j in range(len(mat[i])):
            if math.fabs(mat[i][j]) >= 0.005:
                print('Time', i, 'Variable', j, 'parameter', mat[i][j])
    f = open('flight_data_AR_altitude(141)_order_5.txt', 'w')
    for i in mat:
        lst = i
        string = ''
        for j in range(len(lst)):
            if j < len(lst) - 1:
                string += str(lst[j]) + ', '
            else:
                string += str(lst[j]) + '\n'
        f.write(string)
    f.flush()
    f.close()
    print('End of model estimation.')
    anomaly_lst = result.anomaly(data_lst, 141, threshold=0.02)
    print(anomaly_lst)
    return


def flight_data_multivariate_AR_model_demo_step_2():
    file = open('flight_data_AR_altitude(141)_order_5.txt', 'r')
    parameter = []
    for line in file:
        string = line.strip()
        lst = string.split(sep=', ')
        for j in range(len(lst)):
            lst[j] = lst[j].strip()
        for j in range(len(lst)):
            lst[j] = float(lst[j])
        for j in range(len(lst)):
            if math.fabs(lst[j]) > 0.009:
                print(j)
        parameter.append(lst)
    data_lst = read_file('flight_data_final.txt')
    result_demo_1 = MAR(parameter).error_list(data_lst, 141)
    plt.hist(list(map(lambda x: x*100, result_demo_1)), bins='auto')
    plt.show()
    anomaly = MAR(parameter).anomaly(data_lst, 141, threshold=0.1)
    print('Anomalies', anomaly)
    print('Number of anomaly', len(anomaly))


# flight_data_multivariate_AR_model_demo_step_2()

# file = open('flight_data_final.txt', 'r')
# line_num = 0
# for line in file:
#     if line_num in [42, 99, 118, 140, 141, 142, 148, 149, 153, 154, 177]:
#         print(line)
#     line_num += 1

# file = open('flight_data_L_Press_Alt.txt', 'r')
# for line in file:
#     string = line.strip()
#     lst = string.split(sep=',')
#     for i in range(len(lst)):
#         lst[i] = lst[i].strip()
#         lst[i] = float(lst[i])
# print(len(lst))
# # From 1224, the data becomes normal. 3200-9000 is the time of flying in a horizontal direction.
# a = differencing(lst)[3200:9000]
# training = a[:4300]
# test = a[4300:5000]
# for m in AR.model_estimator(training):
#     print('Order', m.order)
#     print('Parameter', m.phi)
#     plt.plot(m.prediction(training, 100))
# plt.plot(test[:100])
# print(AR.anomaly_detection_score(training, test))
# plt.show()

