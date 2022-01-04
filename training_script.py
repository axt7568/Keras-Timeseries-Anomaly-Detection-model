# Keras Time Series Anomaly Detection using HPV Data
# Setup
import math
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import pickle
import time
start_time = time.time()

healthy_data_file_1 = "C:/Healthy.xlsx"
healthy_data_1 = pd.read_excel(healthy_data_file_1)

healthy_data_file_2 = "C:/Partial_Healthy.xlsx"
healthy_data_2 = pd.read_excel(healthy_data_file_2)

failure_data_file_1 = "C:/Failure.xlsx"
failure_data_1 = pd.read_excel(failure_data_file_1)

failure_data_file_2 = "C:/Failure.xlsx"
failure_data_2 = pd.read_excel(failure_data_file_2)

# healthy_test_data_0 = data.iloc[:,:]

# train_data = healthy_data_1
data_seperation_index = (int)(len(healthy_data_1)/2)
healthy_train_data_1 = healthy_data_1.iloc[:data_seperation_index,:]
healthy_test_data_1 = healthy_data_1.iloc[data_seperation_index:,:]
healthy_test_data_2 = healthy_data_2.iloc[:,:]
failure_test_data_1 = failure_data_1.iloc[:,:]
failure_test_data_2 = failure_data_2.iloc[:,:]



# Code to Train on a Single Feature
# Feature Number
# f_num = 3
# # train_data = data.iloc[:80000,:]
# healthy_test_data_0 = data.iloc[:,f_num]
# # train_data = healthy_data_1
# data = data.iloc[:,f_num]
# healthy_test_data_1 = healthy_data_1.iloc[:,f_num]
# healthy_test_data_2 = healthy_data_2.iloc[:,f_num]
# train_data = pd.concat([healthy_test_data_1])
# failure_test_data_1 = failure_data_1.iloc[:,f_num]
# failure_test_data_2 = failure_data_2.iloc[:,f_num]

train_data = pd.concat([healthy_train_data_1])

# Print shape of the data
print(healthy_test_data_1.shape)
print(healthy_test_data_2.shape)
print(failure_test_data_1.shape)
print(failure_test_data_2.shape)

# Quick Look at the data
print(healthy_test_data_1.head())
print(healthy_test_data_2.head())
print(failure_test_data_1.head())
print(failure_test_data_2.head())

# Visualize the data
# Timeseries Training data without anomalies
# fig, ax = plt.subplots()
hpv_train_data = train_data
# hpv_train_data.astype(float).plot(legend=False, ax=ax)
#plt.title("Raw Timeseries Training data without anomalies")
# fig.savefig("C:/Raw_Timeseries_Training_data_without_anomalies_f1.png")
# plt.show()

# # Timeseries Testing data without anomalies
# fig, ax = plt.subplots()
# hpv_good_test_data = hpv_good_test_data.astype(float)
# hpv_good_test_data.plot(legend=False, ax=ax)
# plt.title("Raw Timeseries Test data without anomalies")
# fig.savefig("C:/Raw_Timeseries_Test_data_without_anomalies.png")
# # plt.show()

"""
# Timeseries Test data wih anomalies
fig, ax = plt.subplots()
hpv_bad_test_data = hpv_bad_test_data.astype(float)
hpv_bad_test_data.plot(legend=False, ax=ax)
# plt.title("Raw Timeseries Test data with anomalies")
# # plt.show()

# Timeseries Test data wih anomalies
fig, ax = plt.subplots()
hpv_high_power_test_data = hpv_high_power_test_data.astype(float)
hpv_high_power_test_data.plot(legend=False, ax=ax)
# plt.title("Raw Timeseries High Power Test data without anomalies")
# # plt.show()
"""

# Prepare training data
# Normalize and save the mean and std we get

# for normalizing training data
training_mean = hpv_train_data.mean()
# training_std = 1
training_std = hpv_train_data.std()


for i in range(0, len(training_std)):
    if((abs(training_std[i]) - 0) < 0.001):
        # val = (hpv_train_data.iloc[:,i])
        val = training_mean[i]
        val = abs(val)
        if((val - 0) < 0.001):
            val = 1
        training_std[i] = val

# Code to Normalize Single Feature Data

# for i in range(0, 1):
#     if((abs(training_std) - 0) < 0.001):
#         # val = (hpv_train_data.iloc[:,i])
#         val = training_mean
#         val = abs(val)
#         if((val - 0) < 0.001):
#             val = 1
#         training_std = val

normalized_train_data = (abs(hpv_train_data - training_mean))/training_std
normalized_train_data=normalized_train_data.astype(float)

# Code to Normalize Data Between 0 and 1
# max_val = hpv_train_data.max()
# min_val = hpv_train_data.min()
# normalized_train_data = (hpv_train_data - min_val)/max_val
# normalized_train_data = hpv_train_data
# plt.hist(healthy_test_data_1, bins=50, color=['g'] , label="PA raw values")
# print("Number of training samples:", len(normalized_train_data))

# # Normalized good test data 0
# normalized_good_test_data_0 = (healthy_test_data_0 - training_mean) / training_std
#
# # Normalized good test data 1
# normalized_good_test_data_1 = (healthy_test_data_1 - training_mean) / training_std
#
# # Normalized good test data 2
# normalized_good_test_data_2 = (healthy_test_data_2 - training_mean) / training_std
#
# # Normalized bad test data 1
# normalized_bad_test_data_1 = (failure_test_data_1 - training_mean) / training_std
#
# # Normalized bad test data 2
# normalized_bad_test_data_2 = (failure_test_data_2 - training_mean) / training_std
#
# plt.hist(normalized_good_test_data_1, bins=50, color=['g'] , label="Healthy data 1")
# plt.hist(normalized_good_test_data_2, bins=50, color=['b'] , label="Healthy data 2")
# plt.hist(normalized_bad_test_data_1, bins=50, color=['r'] , label="Failure data 1")
# plt.hist(normalized_bad_test_data_2, bins=50, color=['k'] , label="Failure data 2")
#
# plt.title('Histogram of Normalized Values for all Distributions using just PA voltage')
# plt.legend()
# plt.xlabel('Distribution values')
# plt.ylabel('Number of Samples')
# # fig.savefig("C:/Raw_Distribution_Histogram_Block_Run_all_features_p250_S_1.png")
# plt.show()

# Plot Training normalized data
# fig, ax = plt.subplots()
# normalized_train_data=normalized_train_data.astype(float)
# normalized_train_data.plot(legend=False, ax=ax)
# plt.legend()
# plt.xlabel("Sample Number")
# plt.ylabel("Normalized Numerical Value")
# plt.title("Normalized Timeseries Training data without anomalies")
# fig.savefig("C:/Normalized_Timeseries_Training_data_without_anomalies_f1.png")
# plt.show()

#
# # Plot Test normalized data
#
# # Healthy Test Data 1
# normalized_good_test_data_1 = (healthy_test_data_1 - training_mean)/training_std
# fig, ax = plt.subplots()
# normalized_good_test_data_1=normalized_good_test_data_1.astype(float)
# normalized_good_test_data_1.plot(legend=False, ax=ax)
# plt.legend()
# plt.xlabel("Sample Number")
# plt.ylabel("Normalized Numerical Value")
# plt.title("Normalized Timeseries Test data without anomalies")
# fig.savefig("C:/Normalized_Timeseries_Test_data_without_anomalies_f1.png")
# # plt.show()
#
# # Healthy Test Data 2
# normalized_good_test_data_2 = (healthy_test_data_2 - training_mean)/training_std
# fig, ax = plt.subplots()
# normalized_good_test_data_2=normalized_good_test_data_2.astype(float)
# normalized_good_test_data_2.plot(legend=False, ax=ax)
# plt.legend()
# plt.xlabel("Sample Number")
# plt.ylabel("Normalized Numerical Value")
# plt.title("Normalized Timeseries Test data without anomalies")
# fig.savefig("C:/Normalized_Timeseries_Test_data_without_anomalies_f1.png")
# # plt.show()
#
# # Failure Test Data 1
# normalized_bad_test_data_1 = (failure_test_data_1 - training_mean)/training_std
# fig, ax = plt.subplots()
# normalized_bad_test_data_1=normalized_bad_test_data_1.astype(float)
# normalized_bad_test_data_1.plot(legend=False, ax=ax)
# plt.legend()
# plt.xlabel("Sample Number")
# plt.ylabel("Normalized Numerical Value")
# plt.title("Normalized Timeseries Test data without anomalies")
# fig.savefig("C:/Normalized_Timeseries_Test_data_without_anomalies_f1.png")
# # plt.show()
#
# # Failure Test Data 2
# normalized_bad_test_data_2 = (failure_test_data_2 - training_mean)/training_std
# fig, ax = plt.subplots()
# normalized_bad_test_data_2=normalized_bad_test_data_2.astype(float)
# normalized_bad_test_data_2.plot(legend=False, ax=ax)
# plt.legend()
# plt.xlabel("Sample Number")
# plt.ylabel("Normalized Numerical Value")
# plt.title("Normalized Timeseries Test data without anomalies")
# fig.savefig("C:/Normalized_Timeseries_Test_data_without_anomalies_f1.png")
# # plt.show()


# Create sequences
# create sequences combining TIME_STEPS contiguous data values from the training data
TIME_STEPS = 12

# Generated training sequences for use in the model
def create_sequences(values, time_steps=TIME_STEPS):
    output = []
    for i in range(0, len(values) - time_steps, time_steps):
        output.append(values[i : (i + time_steps)])
    return np.stack(output)

x_train = create_sequences(normalized_train_data.values)
print("Training input shape: ", x_train.shape)

# Build model
# Convolutional autoencoder model
kernel_size = 12
filter_size = 32
num_filters = np.shape(x_train)[1]

for i in range(1, 2):
    print("---------- RUN %d---------" %i)
    model = keras.Sequential(
        [
            layers.Input(shape=(x_train.shape[1], x_train.shape[2])),
            # layers.Input(shape=(12, 1)),
            layers.Conv1D(
                filters=filter_size, kernel_size=kernel_size, padding="same", strides=2, activation="relu"
            ),
            layers.Dropout(rate=0.60),
            layers.Conv1D(
                filters=filter_size/2, kernel_size=kernel_size, padding="same", strides=2, activation="relu"
            ),
            layers.Conv1DTranspose(
                filters=filter_size/2, kernel_size=kernel_size, padding="same", strides=2, activation="relu"
            ),
            layers.Dropout(rate=0.60),
            layers.Conv1DTranspose(
                filters=filter_size, kernel_size=kernel_size, padding="same", strides=2, activation="relu"
            ),
            layers.Conv1DTranspose(filters=num_filters, kernel_size=kernel_size, padding="same"),
        ]
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    model.summary()
    epoch = {}

    """
    # Print Training Shape and Weights
    print(x_train.shape[1])
    print(x_train.shape[2])
    conv = model.layers[0].kernel_initializer
    print(conv)
    """

    """
    # Print Weight Values
    for i in range(1,5):
        # print(model.layers[0].get_weights())
        print("index is =",i)
        print(model.layers[0].get_weights()[0])
    """

    # x_train = x_train.reshape()
    # Train the model
    # x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    history = model.fit(
        x_train,
        x_train,
        epochs=1000,
        batch_size=12,
        validation_split=0.1,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                min_delta=0,
                patience=10,
                verbose=1,
                mode="min",
        )]
    )

    n_epochs = len(history.history["loss"])
    epoch[i] = n_epochs


    # Plot Training and Validation Loss
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.legend()
    plt.savefig("C:/Training_vs_Validation_Loss.png")
    plt.show()

    healthy_test_data_0 = healthy_train_data_1

    # Save model
    model.save("saved_model.h5")
    with open("saved_pickle_file.pickle", 'wb') as f:
        pickle.dump([x_train, healthy_test_data_0, healthy_test_data_1, healthy_test_data_2, failure_test_data_1, failure_test_data_2, training_mean, training_std, hpv_train_data, TIME_STEPS, n_epochs], f)

print("--- %s seconds ---" % (time.time() - start_time))