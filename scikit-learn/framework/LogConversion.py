
#returns log data in dictonary form
def get_log_data(path_to_log):

    output_dict = {}
    with open(path_to_log, "r") as f:
        for i, line in enumerate(f):
            line = line.strip() # remove /n on end
            for item in line.split(", "):
                key, value = tuple(item.split(" "))
                if i == 0:
                    output_dict[key] = [round(float(value), 3)]
                else:
                    output_dict[key].append(round(float(value), 3))

    return output_dict

def get_2D_loss_data(log_path, x_key="batch_number_accumulated", y_key="loss_over_batch"):
    log_data = get_log_data(log_path)
    return log_data[x_key], log_data[y_key]



if __name__ == "__main__":
    get_log_data("../../output/train_output/ResNetBinary/runs/20240312-213006/train_losses.txt")


