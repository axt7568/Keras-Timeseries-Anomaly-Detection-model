import pickle
from keras.models import load_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import show
import time
import scipy.io
start_time = time.time()

# Load Files
with open("saved_pickle_file.pickle", 'rb') as f:
    [mean_healthy_data_0_dict, mean_healthy_data_1_dict, mean_healthy_data_2_dict, mean_failure_data_1_dict, mean_healthy_data_2_dict, sd_healthy_data_0_dict, sd_healthy_data_1_dict, sd_healthy_data_2_dict, sd_failure_data_1_dict, sd_failure_data_2_dict, epoch_count, healthy_data_distribution_0, healthy_data_distribution_1, healthy_data_distribution_2, failure_data_distribution_1, failure_data_distribution_2] = pickle.load(f)

# Plot Histogram for all 3 Mean values
# min_red = np.amin(list(red_distribution_sse_data.values()))
# max_blue = np.amax(list(blue_distribution_sse_data.values()))
# max_green = np.amax(list(green_distribution_sse_data.values()))
# max_blue_green = max(max_blue, max_green)
# threshold = (min_red + max_blue_green)/2
# threshold = round(threshold, 4)
fig, ax = plt.subplots()

# # plt.hist(healthy_data_distribution_0.values(), color=['y','y','y','y','y','y','y','y','y','y'] , label="Good Test Data 1 SSE Loss Values")
# plt.hist(healthy_data_distribution_1.values(), color=['g','g','g','g','g','g','g','g','g','g'] , label="Training Data 1 SSE Loss Values")
# plt.hist(healthy_data_distribution_2.values(), color=['g','g','g','g','g','g','g','g','g','g'] , label="Training Data 2 SSE Loss Values")
# # plt.hist(healthy_data_distribution_2.values(), color=['b','b','b','b','b','b','b','b','b','b'] , label="Good Test Data 2 SSE Loss Values")
# plt.hist(failure_data_distribution_1.values(), color=['r','r','r','r','r','r','r','r','r','r'] , label="Failure Test Data 1 SSE Loss Values")
# plt.hist(failure_data_distribution_2.values(), color=['k','k','k','k','k','k','k','k','k','k'] , label="Failure Test Data 2 SSE Loss Values")

# Single run histogram
# plt.hist(healthy_data_distribution_0.values(), color=['y','y','y','y','y','y','y','y','y','y'] , label="Good Test Data 1 SSE Loss Values")
plt.hist(healthy_data_distribution_0.values(), bins=500, color=['y'] , label="Training Data 0 SSE Loss Values")
plt.hist(healthy_data_distribution_1.values(), bins=500, color=['g'] , label="Test Data 1 SSE Loss Values")
# plt.hist(healthy_data_distribution_2.values(), bins=500, color=['b'] , label="Test Data 2 SSE Loss Values")
plt.hist(failure_data_distribution_1.values(), bins=500, color=['r'] , label="Failure Test Data 1 SSE Loss Values")
plt.hist(failure_data_distribution_2.values(), bins=500, color=['k'] , label="Failure Test Data 2 SSE Loss Values")

# healthy_data_d0_loss_val = healthy_data_distribution_0[1]
# plt.plot(healthy_data_d0_loss_val, color="yellow", label="Train Data SSE Loss Values")
#
# healthy_data_d1_loss_val = healthy_data_distribution_1[1]
# plt.plot(healthy_data_d1_loss_val, color="green", label="Test Data 1 SSE Loss Values")
#
# healthy_data_d2_loss_val = healthy_data_distribution_2[1]
# plt.plot(healthy_data_d2_loss_val, color="blue", label="Test Data 2 SSE Loss Values")
#
# failure_data_d1_loss_val = failure_data_distribution_1[1]
# plt.plot(failure_data_d1_loss_val, color="red", label="Test Data 3 SSE Loss Values")
#
# failure_data_d2_loss_val = failure_data_distribution_2[1]
# plt.plot(failure_data_d2_loss_val, color="black", label="Test Data 4 SSE Loss Values")

# Test_Data_Time_Series_Error_Val = [healthy_data_d1_loss_val[:1000], healthy_data_d2_loss_val[:1000], failure_data_d1_loss_val[:1000], failure_data_d2_loss_val[:1000]]
# # Test_Data_Time_Series_Error_Val = [healthy_data_d1_loss_val, healthy_data_d2_loss_val, failure_data_d1_loss_val, failure_data_d2_loss_val]
# Train_Data_Time_Series_Error_Val = [healthy_data_d0_loss_val]

# with open("saved_pickle_file.pickle", 'wb') as f:
#     pickle.dump([Train_Data_Time_Series_Error_Val], f)
#
# with open("saved_pickle_file.pickle",'wb') as f:
#     pickle.dump([Test_Data_Time_Series_Error_Val], f)

# plt.axvline(x=threshold, color='k', linestyle='--', label="Threshold = "+str(threshold))
# plt.axvline
#
# plt.title('Histogram of Raw Values for all Distributions using all features')
# plt.title('SSE Histogram Plot')
plt.title('Test Data Timeseries Error Plot')
plt.legend()
plt.ylabel('SSE Loss Values')
# plt.xlabel('Distribution values')
# plt.ylabel('Number of Samples')
plt.xlabel('Sample Number')
# fig.savefig("Raw_Distribution_Histogram.png")
plt.show()
#
# Train_Data_Time_Series_Error_Val={'Train_Data_Time_Series_Error_Val':Train_Data_Time_Series_Error_Val}
# scipy.io.savemat('Train_Data_Time_Series_Error_Val.mat', Train_Data_Time_Series_Error_Val)
#
# Test_Data_Time_Series_Error_Val={'Test_Data_Time_Series_Error_Val':Test_Data_Time_Series_Error_Val}
# scipy.io.savemat('Test_Data_Time_Series_Error_Val.mat', Test_Data_Time_Series_Error_Val)



# plt.title('Test Dataset 02 Loss Values')
# plt.legend()
# plt.ylabel('SSE Loss value')
# plt.xlabel('Sample Number')
# # fig.savefig("Raw_Distribution_Histogram_Block_Run.png")
# plt.show()


print("--- %s seconds ---" % (time.time() - start_time))