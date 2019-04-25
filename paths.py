import os


cwd = ".."

# input paths
input_path = os.path.join(cwd, 'inputs')
train_data_path = os.path.join(input_path,"train.npy")
train_labels_path = os.path.join(input_path,"train_transcripts.npy")
valid_data_path = os.path.join(input_path,"dev.npy")
valid_labels_path = os.path.join(input_path,"dev_transcripts.npy")
test_data_path = os.path.join(input_path,"test.npy")
# # test_labels_path = os.path.join(environment_path,"wsj0_train_merged_labels.npy")


# # model_path = "../model/model_dict.pt"

output_path = os.path.join(cwd,"outputs")

# plot pathes
attention_path = os.path.join(output_path, 'attention_plots')
gradient_path = os.path.join(output_path, 'gradient_plots')