import Data_treatment
import AR
import matplotlib.pyplot as plt
import math
import Old_AR
# There are kinds of thresholds:
# (1) The threshold of anomaly. Now it is 2, we can set it lower.
# (2) The threshold of data for prediction, used in prediction function. Now it is 10.
# (3) The threshold of big anomaly, used in prediction update. Now it is 50.
# If we want to change (2)(3), we can observe the histogram of the anomaly rates.


def real_time_anomaly_detection(data, anomaly_threshold=2, used_threshold=10, change_threshold=50, force_to_change=1, batch=1000):
    training = data[:batch]
    test = data[batch:2*batch]
    model_lst = AR.model_estimator(training)
    left_data = data[batch:]
    anomaly_lst = []
    anomaly_rate = []
    for i in range(len(model_lst)):
        new = []
        for j in range(len(training)):
            new.append(0)
        anomaly_rate.append(new)
    prediction = []
    for i in range(len(model_lst)):
        prediction.append(training.copy())
    start_index = batch
    if_update = 0
    forecast_error = []
    for i in range(len(model_lst)):
        new = []
        for j in range(len(training)):
            new.append(0)
        forecast_error.append(new)
    while True:
        a, b = AR.anomaly_detection_least(training, test, threshold=anomaly_threshold, used_threshold=used_threshold, also_largest=True, model_lst=model_lst)
        # d = Old_AR.anomaly_detection_score(training, test, threshold=anomaly_threshold, models=model_lst, if_return_score=True)
        # e = max_index(d[1], output_num=10)
        # d = list(map(lambda x: d[0][x], e))
        # for i in d:
        #     if i not in a:
        #         a.append(i)
        #         print(i + start_index)
        anomaly_lst += list(map(lambda x: x + start_index, sorted(a)))
        # training = test
        c = [model.prediction_advanced(training, test, used_threshold=used_threshold) for model in model_lst]
        training = c[0][2]
        if_force = 0
        for i in range(len(model_lst)):
            if percentage_of_large(c[i][1], anomaly_threshold) > force_to_change:
                if_force = 1
                break
        if if_force == 1:
            model_lst = AR.model_estimator(test)
            for i in range(len(model_lst)):
                prediction[i] += c[i][0]
                forecast_error[i] += [math.sqrt(math.fabs(model_lst[i].noise_variance))] * len(test)
                anomaly_rate[i] += list(map(lambda x: x / math.sqrt(math.fabs(model_lst[i].noise_variance)), model_lst[i].examine(test)))
        else:
            for i in range(len(model_lst)):
                prediction[i] += c[i][0]
                anomaly_rate[i] += c[i][1]
                forecast_error[i] += c[i][3]
        left_data = left_data[batch:]
        test = left_data[:batch]
        start_index += batch
        if if_update == 2:
            model_lst = AR.model_estimator(training)
            if_update = 0
        elif if_update == 1:
            if max(b) >= change_threshold:
                pass
            else:
                if_update += 1
        else:
            if max(b) >= change_threshold:
                if_update += 1
            else:
                pass
        if not left_data:
            break
    return anomaly_lst, prediction, anomaly_rate, forecast_error


def percentage_of_large(lst, threshold):
    if len(lst) == 0:
        return 1
    i = 0
    for j in lst:
        if math.fabs(j) > threshold:
            i += 1
    return i / len(lst)


def max_index(lst, output_num=100):
    if output_num >= len(lst):
        return [i for i in range(len(lst))]
    s = sorted(lst)
    threshold = s[-output_num]
    output = []
    for i in range(len(lst)):
        if lst[i] >= threshold:
            output.append(i)
    return output


# Only when there are three selected models.
def real_time_anomaly_detection_modified(data, anomaly_threshold=2, used_threshold=10, change_threshold=50, force_to_change=0.5, batch=1000, output_num='auto'):
    if output_num == 'auto':
        output_num = int(len(data) / 1000)
    result = real_time_anomaly_detection(data, anomaly_threshold=anomaly_threshold, used_threshold=used_threshold, change_threshold=change_threshold, force_to_change=force_to_change, batch=batch)
    anomalies = list(map(lambda x: result[2][0][x]+result[2][1][x]+result[2][2][x], result[0]))
    output = max_index(anomalies, output_num=output_num)
    anomaly_index = list(map(lambda x: result[0][x], output))
    return anomaly_index, result[1], result[2], result[3]


# Electricity.

# Using steps-forward prediction.
def ARJ_electricity_demo_1(target_variable):
    data = Data_treatment.read_file(filename='Electricity_transpose.txt')
    raw_data = differencing(data[target_variable])
    # plt.plot(raw_data)
    training = raw_data[:100000]
    test = raw_data[100000:]
    # plt.plot(training)
    model_lst = AR.model_estimator(raw_data)
    anomaly = AR.anomaly_detection_score(training, test, threshold=2)
    print('Anomaly', anomaly)
    print('Number of anomalies', len(anomaly))
    plt.plot(test)
    # plt.scatter(anomaly, list(map(lambda x: test[x], anomaly)), color='k', s=10)
    for model in model_lst:
        print('Order', model.order)
        print('Coefficient', model.phi)
        print('Variance', model.noise_variance)
        plt.plot(model.prediction_advanced(training, test)[0])
        # print('Anomaly rate', model.prediction_advanced(training, test)[1])
    # output = prediction_error(raw_data, model_lst[0])
    # plt.plot(raw_data[38000:48000])
    # a = []
    # for i in range(100):
    #     a.append(100 * i)
    # plt.hist(anomaly, bins=a)
    plt.show()
    return

# Using real-time prediction, but for differentiated data.
def ARJ_electricity_demo_2(target_variable):
    data = Data_treatment.read_file(filename='Electricity_transpose.txt')
    raw_data = differencing(data[target_variable])
    anomaly_lst, prediction_of_3, anomaly_rate_of_3, forecast_error_of_3 = real_time_anomaly_detection(raw_data, anomaly_threshold=2)
    print('Anomalies', anomaly_lst)
    print('Number of anomalies', len(anomaly_lst))
    plt.plot(raw_data)
    for i in [0, 1, 2]:
        plt.plot(prediction_of_3[i])
    plt.scatter(anomaly_lst, list(map(lambda x: raw_data[x], anomaly_lst)), color='k', s=3)
    plt.show()
    a = []
    for i in range(10):
        a.append(i)
    for i in [0]:
        plt.hist(anomaly_rate_of_3[i], bins=a)
        print('Max', max(anomaly_rate_of_3[i]))
    plt.show()
    return


def demo_sample(target_variable, filename='Electricity_transpose.txt', anomaly_threshold=2, used_threshold=10, change_threshold=50, force_to_change=0.5, batch=1000, if_print=True, if_plot=True, if_differentiated=False, if_plot_predict=False, if_confident_interval=False, if_slice=None, if_top=False, point_radius=3, label='Electricity'):
    data, original = data_preprocess(target_variable, filename=filename)
    if if_top:
        if type(if_top) == int:
            anomaly_lst, prediction_of_3, anomaly_rate_of_3, forecast_error_of_3 = real_time_anomaly_detection_modified(data, anomaly_threshold=anomaly_threshold, used_threshold=used_threshold, change_threshold=change_threshold, force_to_change=force_to_change, batch=batch, output_num=if_top)
        else:
            anomaly_lst, prediction_of_3, anomaly_rate_of_3, forecast_error_of_3 = real_time_anomaly_detection_modified(data, anomaly_threshold=anomaly_threshold, used_threshold=used_threshold, change_threshold=change_threshold, force_to_change=force_to_change, batch=batch)
    else:
        anomaly_lst, prediction_of_3, anomaly_rate_of_3, forecast_error_of_3 = real_time_anomaly_detection(data, anomaly_threshold=anomaly_threshold, used_threshold=used_threshold, change_threshold=change_threshold, force_to_change=force_to_change, batch=batch)
    # print: anomalies, anomaly rates, amount of anomalies, properties of models
    if if_print:
        print('The time stamps of the anomalies:', anomaly_lst)
        print('Number of anomalies:', len(anomaly_lst))
        print('Percentage of anomalies:', int(len(anomaly_lst) * 10000 / len(data)) / 100, '%')
    # plot: anomalies, data, undifferentiated data, predictions of 3
    if if_plot:
        if if_differentiated:
            if if_slice:
                plt.plot(data[if_slice[0]:if_slice[1]], label=label)
            else:
                plt.plot(data, label=label)
            color_lst = ['r', 'g', 'y']
            if not if_slice:
                if if_plot_predict:
                    for i in range(len(color_lst)):
                        plt.plot(prediction_of_3[i], color=color_lst[i], label='Prediction ' + str(i))
                        if if_confident_interval:
                            plt.plot([prediction_of_3[i][j] + anomaly_threshold * forecast_error_of_3[i][j] for j in
                                      range(len(prediction_of_3[i]))], color=color_lst[i], linestyle='--')
                            plt.plot([prediction_of_3[i][j] - anomaly_threshold * forecast_error_of_3[i][j] for j in
                                      range(len(prediction_of_3[i]))], color=color_lst[i], linestyle='--')
                plt.scatter(anomaly_lst, list(map(lambda x: data[x], anomaly_lst)), color = 'r', s=point_radius, label='Anomaly')
            else:
                if if_plot_predict:
                    for i in range(len(color_lst)):
                        plt.plot(prediction_of_3[i][if_slice[0]:if_slice[1]], color=color_lst[i], label='Prediction ' + str(i))
                        if if_confident_interval:
                            plt.plot([prediction_of_3[i][j] + anomaly_threshold * forecast_error_of_3[i][j] for j in
                                      range(if_slice[0], if_slice[1])], color=color_lst[i], linestyle='--')
                            plt.plot([prediction_of_3[i][j] - anomaly_threshold * forecast_error_of_3[i][j] for j in
                                      range(if_slice[0], if_slice[1])], color=color_lst[i], linestyle='--')
                new_anomaly_lst = []
                for i in anomaly_lst:
                    if if_slice[0] <= i < if_slice[1]:
                        new_anomaly_lst.append(i - if_slice[0])
                plt.scatter(new_anomaly_lst, list(map(lambda x: data[x+if_slice[0]], new_anomaly_lst)), color='r', s=3 * point_radius, label='Anomaly')
                print('Number of anomalies in the given interval:', len(new_anomaly_lst))
        else:
            if if_slice:
                plt.plot(original[if_slice[0]:if_slice[1]], label=label)
            else:
                plt.plot(original, label=label)
            if not if_slice:
                plt.scatter(anomaly_lst, list(map(lambda x: original[x + 1], anomaly_lst)), color='r', s=point_radius, label='Anomaly')
            else:
                new_anomaly_lst = []
                for i in anomaly_lst:
                    if if_slice[0] - 1 <= i < if_slice[1] - 1:
                        new_anomaly_lst.append(i + 1 - if_slice[0])
                plt.scatter(new_anomaly_lst, list(map(lambda x: original[x+if_slice[0]], new_anomaly_lst)), color='r', s=3 * point_radius, label='Anomaly')
                print('Number of anomalies in the given interval:', len(new_anomaly_lst))
        plt.legend()
        plt.show()
    print('Here is the end of the demo.')
    return


def differencing(lst):
    result = []
    for i in range(len(lst) - 1):
        result.append(lst[i + 1] - lst[i])
    return result


def data_preprocess(target_variable, filename='Electricity_transpose.txt'):
    data = Data_treatment.read_file(filename=filename)
    return differencing(data[target_variable]), data[target_variable]


def old_demo(target_variable, filename='Electricity_transpose.txt'):
    d = data_preprocess(target_variable, filename=filename)[0]
    demo_training = d[:int(len(d) / 2)]
    print(demo_training)
    demo_test = d[int(len(d) / 2):]
    result = Old_AR.anomaly_detection_score(demo_training, demo_test, threshold=5)
    plt.plot(demo_test)
    plt.scatter(result, list(map(lambda x: demo_test[x], result)), color='r', s=3)
    plt.show()
    return


def test():
    para6 = AR.read_data('para6_108D.csv')[0]
    para6_resample = AR.resample(para6, 16)[0]
    para6_resample_diff = AR.Data(para6_resample).differencing(2)[0]
    result = real_time_anomaly_detection(para6_resample_diff, batch=30, anomaly_threshold=3)
    plt.plot(para6_resample_diff, label='engine data')
    plt.scatter(result[0], list(map(lambda x: para6_resample_diff[x], result[0])), color='r', s=20, label='anomaly')
    plt.legend()
    plt.show()
    return


# ARJ_electricity_demo_1(0)
# ARJ_electricity_demo_2(10)
# demo_sample(9, filename='Performance_1_transpose.txt', anomaly_threshold=3, point_radius=10, force_to_change=0.5)
# demo_sample(0, anomaly_threshold=5, point_radius=20, if_top=100)
# old_demo(0, filename='Performance_1_transpose.txt')
# test()

