import pickle
from keras.models import load_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import show
import time

start_time = time.time()

# Healthy Data 0
healthy_data_distribution_0 = {}
mean_healthy_data_0_dict = {}
sd_healthy_data_0_dict = {}

# Healthy Data 1
healthy_data_distribution_1 = {}
mean_healthy_data_1_dict = {}
sd_healthy_data_1_dict = {}

# Healthy Data 2
healthy_data_distribution_2 = {}
mean_healthy_data_2_dict = {}
sd_healthy_data_2_dict = {}

# Failure Data 1
failure_data_distribution_1 = {}
mean_failure_data_1_dict = {}
sd_failure_data_1_dict = {}

# Failure Data 2
failure_data_distribution_2 = {}
mean_failure_data_2_dict = {}
sd_failure_data_2_dict = {}

# Raw Distribution Histograms
red_distribution_sse_data = {}
green_distribution_sse_data = {}
blue_distribution_sse_data = {}

# Mean and SD dictionaries
mean_red = {}
sd_red = {}

mean_green = {}
sd_green = {}

mean_blue = {}
sd_blue = {}

# J value dictionaries
J_red_green = {}
J_red_blue = {}
J_blue_green = {}

# Save Epoch Count
epoch_count = {}

# Load model
for i in range(1, 2):
    model = load_model("saved_model.h5")
    model.summary()
    with open("saved_pickle_file.pickle", 'rb') as f:
        [x_train, healthy_test_data_0, healthy_test_data_1, healthy_test_data_2, failure_test_data_1,
         failure_test_data_2, training_mean, training_std, hpv_train_data, TIME_STEPS, n_epochs] = pickle.load(f)


    # Generated testing sequences for use in the model
    def create_sequences(values, time_steps=TIME_STEPS):
        output = []
        for i in range(0, len(values) - time_steps, 1):
            output.append(values[i: (i + time_steps)])
        return np.stack(output)


    # Get train MAE loss
    x_train_pred = model.predict(x_train)
    train_mae_loss = np.abs(x_train_pred - x_train)
    train_mae_loss = np.mean(train_mae_loss, axis=1)

    # Normalized good test data 0
    normalized_good_test_data_0 = (healthy_test_data_0 - training_mean) / training_std

    # Normalized good test data 1
    normalized_good_test_data_1 = (healthy_test_data_1 - training_mean) / training_std

    # Normalized good test data 2
    normalized_good_test_data_2 = (healthy_test_data_2 - training_mean) / training_std

    # Normalized bad test data 1
    normalized_bad_test_data_1 = (failure_test_data_1 - training_mean) / training_std

    # Normalized bad test data 2
    normalized_bad_test_data_2 = (failure_test_data_2 - training_mean) / training_std

    # # create sequences combining TIME_STEPS contiguous data values from the training data
    TIME_STEPS = 12

    x_good_data_test_0 = create_sequences(normalized_good_test_data_0)
    print("Test Data 0(Train Data) input shape: ", x_good_data_test_0.shape)

    x_good_data_test_1 = create_sequences(normalized_good_test_data_1)
    print("Test Data 1 input shape: ", x_good_data_test_1.shape)

    x_good_data_test_2 = create_sequences(normalized_good_test_data_2)
    print("Test Data 2 input shape: ", x_good_data_test_2.shape)

    x_bad_data_test_1 = create_sequences(normalized_bad_test_data_1)
    print("Test Data 3  input shape: ", x_bad_data_test_1.shape)

    x_bad_data_test_2 = create_sequences(normalized_bad_test_data_2)
    print("Test Data 4 input shape: ", x_bad_data_test_2.shape)

    # Reshape
    num_filters = np.shape(x_train)[1]
    x_good_data_test_0 = x_good_data_test_0.reshape(x_good_data_test_0.shape[0], x_good_data_test_0.shape[1], num_filters)
    print("Test input shape: ", x_good_data_test_0.shape)

    x_good_data_test_1 = x_good_data_test_1.reshape(x_good_data_test_1.shape[0], x_good_data_test_1.shape[1], num_filters)
    print("Test input shape: ", x_good_data_test_1.shape)

    x_good_data_test_2 = x_good_data_test_2.reshape(x_good_data_test_2.shape[0], x_good_data_test_2.shape[1], num_filters)
    print("Test input shape: ", x_good_data_test_2.shape)

    x_bad_data_test_1 = x_bad_data_test_1.reshape(x_bad_data_test_1.shape[0], x_bad_data_test_1.shape[1], num_filters)
    print("Test input shape: ", x_bad_data_test_1.shape)

    x_bad_data_test_2 = x_bad_data_test_2.reshape(x_bad_data_test_2.shape[0], x_bad_data_test_2.shape[1], num_filters)
    print("Test input shape: ", x_bad_data_test_2.shape)

    # Get test MAE loss
    # Healthy Test Data 0
    x_test_pred_good_data_0 = model.predict(x_good_data_test_0)
    test_good_data_0 = np.abs(x_test_pred_good_data_0 - x_good_data_test_0)
    test_good_data_0 = np.mean(test_good_data_0, axis=1)
    sse_good_test_data_0 = np.sum(test_good_data_0[:, :] ** 2, axis=1)

    # Healthy Test Data 1
    x_test_pred_good_data_1 = model.predict(x_good_data_test_1)
    test_good_data_1 = np.abs(x_test_pred_good_data_1 - x_good_data_test_1)
    test_good_data_1 = np.mean(test_good_data_1, axis=1)
    sse_good_test_data_1 = np.sum(test_good_data_1[:, :] ** 2, axis=1)

    # Healthy Test Data 2
    x_test_pred_good_data_2 = model.predict(x_good_data_test_2)
    test_good_data_2 = np.abs(x_test_pred_good_data_2 - x_good_data_test_2)
    test_good_data_2 = np.mean(test_good_data_2, axis=1)
    sse_good_test_data_2 = np.sum(test_good_data_2[:, :] ** 2, axis=1)

    # Failure Test Data 1
    x_test_pred_bad_data_1 = model.predict(x_bad_data_test_1)
    test_bad_data_1 = np.abs(x_test_pred_bad_data_1 - x_bad_data_test_1)
    test_bad_data_1 = np.mean(test_bad_data_1, axis=1)
    sse_bad_test_data_1 = np.sum(test_bad_data_1[:, :] ** 2, axis=1)

    # Failure Test Data 2
    x_test_pred_bad_data_2 = model.predict(x_bad_data_test_2)
    test_bad_data_2 = np.abs(x_test_pred_bad_data_2 - x_bad_data_test_2)
    test_bad_data_2 = np.mean(test_bad_data_2, axis=1)
    sse_bad_test_data_2 = np.sum(test_bad_data_2[:, :] ** 2, axis=1)

    """
    plt.hist(sse_good_test_data,bins=50, color='g', label='Test Data without Anomalies')
    plt.hist(sse_bad_test_data,bins=50, color='r', label='Test Data with Anomalies')
    plt.hist(sse_high_power_good_test_data,bins=50, color='b', label='High Power Test Data without Anomalies')
    # plt.axvline(x=threshold, color='k', linestyle='--')
    plt.title('Sum Squared Error Histogram Plot')
    plt.legend()
    plt.xlabel('Sum Squared Error Value')
    plt.ylabel('Number of Samples')
    plt.show()
    """

    # mean_R = np.mean(sse_bad_test_data)
    # sd_R = np.std(sse_bad_test_data)

    # mean_B = np.mean(sse_high_power_good_test_data)
    # sd_B = np.std(sse_high_power_good_test_data)

    # J_R_G = (mean_R - mean_G) / max(sd_R, sd_G)
    # J_R_B = (mean_R - mean_B) / max(sd_R, sd_B)
    # J_B_G = (max(mean_G, mean_B) - min(mean_G, mean_B)) / max(sd_G, sd_B)
    #
    # red_distribution_sse_data[i] = sse_bad_test_data
    # mean_red[i] = mean_R
    # sd_red[i] = sd_R

    # Healthy Data 0
    mean_healthy_data_0 = np.mean(sse_good_test_data_0)
    sd_healthy_data_0 = np.std(sse_good_test_data_0)
    healthy_data_distribution_0[i] = sse_good_test_data_0
    mean_healthy_data_0_dict[i] = mean_healthy_data_0
    sd_healthy_data_0_dict[i] = sd_healthy_data_0

    # Healthy Data 1
    mean_healthy_data_1 = np.mean(sse_good_test_data_1)
    sd_healthy_data_1 = np.std(sse_good_test_data_1)
    healthy_data_distribution_1[i] = sse_good_test_data_1
    mean_healthy_data_1_dict[i] = mean_healthy_data_1
    sd_healthy_data_1_dict[i] = sd_healthy_data_1

    # Healthy Data 2
    mean_healthy_data_2 = np.mean(sse_good_test_data_2)
    sd_healthy_data_2 = np.std(sse_good_test_data_2)
    healthy_data_distribution_2[i] = sse_good_test_data_2
    mean_healthy_data_2_dict[i] = mean_healthy_data_2
    sd_healthy_data_2_dict[i] = sd_healthy_data_2

    # failure Data 1
    mean_failure_data_1 = np.mean(sse_bad_test_data_1)
    sd_failure_data_1 = np.std(sse_bad_test_data_1)
    failure_data_distribution_1[i] = sse_bad_test_data_1
    mean_failure_data_1_dict[i] = mean_failure_data_1
    sd_failure_data_1_dict[i] = sd_failure_data_1

    # failure Data 2
    mean_failure_data_2 = np.mean(sse_bad_test_data_2)
    sd_failure_data_2 = np.std(sse_bad_test_data_2)
    failure_data_distribution_2[i] = sse_bad_test_data_2
    mean_failure_data_2_dict[i] = mean_failure_data_2
    sd_failure_data_2_dict[i] = sd_failure_data_2

    # blue_distribution_sse_data[i] = sse_high_power_good_test_data
    # mean_blue[i] = mean_B
    # sd_blue[i] = sd_B
    #
    # J_red_green[i] = J_R_G
    # J_red_blue[i] = J_R_B
    # J_blue_green[i] = J_B_G

    epoch_count[i] = n_epochs

# fig, ax = plt.subplots()
#
# # plt.hist(healthy_data_distribution_0.values(), color=['y','y','y','y','y','y','y','y','y','y'] , label="Good Test Data 1 SSE Loss Values")
# plt.hist(healthy_data_distribution_1.values(), color=['g','g','g','g','g','g','g','g','g','g'] , label="Training Data 1 SSE Loss Values")
# plt.hist(healthy_data_distribution_2.values(), color=['g','g','g','g','g','g','g','g','g','g'] , label="Training Data 2 SSE Loss Values")
# # plt.hist(healthy_data_distribution_2.values(), color=['b','b','b','b','b','b','b','b','b','b'] , label="Good Test Data 2 SSE Loss Values")
# plt.hist(failure_data_distribution_1.values(), color=['r','r','r','r','r','r','r','r','r','r'] , label="Failure Test Data 1 SSE Loss Values")
# plt.hist(failure_data_distribution_2.values(), color=['k','k','k','k','k','k','k','k','k','k'] , label="Failure Test Data 2 SSE Loss Values")
#
#
# # plt.axvline(x=threshold, color='k', linestyle='--', label="Threshold = "+str(threshold))
#
# plt.title('Histogram of Raw Values for all Distributions')
# plt.legend()
# plt.xlabel('Distribution values')
# plt.ylabel('Number of Samples')
# fig.savefig("C:/Raw_Distribution_Histogram_Block_Run.png")
# plt.show()


with open("saved_pickle_file.pickle",'wb') as f:
    pickle.dump(
        [mean_healthy_data_0_dict, mean_healthy_data_1_dict, mean_healthy_data_2_dict, mean_failure_data_1_dict,
         mean_healthy_data_2_dict, sd_healthy_data_0_dict, sd_healthy_data_1_dict, sd_healthy_data_2_dict,
         sd_failure_data_1_dict, sd_failure_data_2_dict, epoch_count, healthy_data_distribution_0,
         healthy_data_distribution_1, healthy_data_distribution_2, failure_data_distribution_1,
         failure_data_distribution_2], f)

print("--- %s seconds ---" % (time.time() - start_time))
