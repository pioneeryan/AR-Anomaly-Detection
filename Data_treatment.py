import matplotlib.pyplot as plt


# Transpose the original data file so that each row is a variable.
def transpose(filename='ARJ1110-640-20120505-1_电源_time-16.txt', outname='Electricity_transpose.txt'):  # Tested.
    read_data_file = open(filename, 'r')
    write_data_file = open(outname, 'w')
    output = []
    i = 0
    for line in read_data_file:
        string = line.strip()
        lst = string.split()
        for j in range(len(lst)):
            lst[j] = lst[j].strip()
        new_lst = []
        for j in range(len(lst)):
            if lst[j] != '':
                new_lst.append(lst[j])
        lst = new_lst
        if i == 0:
            for j in range(len(lst)):
                output.append([lst[j]])
        else:
            for j in range(len(lst)):
                output[j].append(lst[j])
        i += 1
    for i in output:
        string = ''
        for j in range(len(i)):
            if j == len(i) - 1:
                string += str(i[j]) + '\n'
            else:
                string += str(i[j]) + ', '
        write_data_file.write(string)
    write_data_file.flush()
    write_data_file.close()
    return


# Generate a matrix of data with each sublist contains the values of a variable.
# No time. No variable name.
def read_file(filename='Electricity_transpose.txt'):  # Tested.
    read_data_file = open(filename, 'r')
    output = []
    i = 0
    for line in read_data_file:
        if i == 0:
            i += 1
            continue
        string = line.strip()
        lst = string.split(sep=',')
        for j in range(len(lst)):
            if j >= 1:
                lst[j] = float(lst[j].strip())
        output.append(lst[1:])
        i += 1
    print('The file ' + '\'' + filename + '\' has', len(output), 'variables, and the time series of each variable is of length', len(output[0]), '.')
    return output


# transpose(filename='ARJ1110-640-20120505-1_性能附件1_time-8.txt', outname='Performance_1_transpose.txt')
# plt.plot(read_file('Performance_1_transpose.txt')[1])
# plt.show()

