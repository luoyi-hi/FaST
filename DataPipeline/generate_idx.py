import numpy as np
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))


def generate_and_split_indices_sequential(data_name, input_step, output_step):

    data_npz = np.load(f"../main-master/datasets/{data_name}/his.npz")

    data = data_npz["data"]
    T, N, D = data.shape
    total_steps = input_step + output_step

    indices = np.arange(total_steps - 1, T)
    num_samples = len(indices)

    train_end = int(0.6 * num_samples)
    val_end = train_end + int(0.2 * num_samples)
    idx_train = indices[:train_end]
    idx_val = indices[train_end:val_end]
    idx_test = indices[val_end:]

    target_dir = os.path.join(
        "..", "main-master", "datasets", data_name, f"{input_step}_{output_step}"
    )
    os.makedirs(target_dir, exist_ok=True)

    np.save(
        f"../main-master/datasets/{data_name}/{input_step}_{output_step}/idx_train.npy",
        idx_train,
    )
    np.save(
        f"../main-master/datasets/{data_name}/{input_step}_{output_step}/idx_val.npy",
        idx_val,
    )
    np.save(
        f"../main-master/datasets/{data_name}/{input_step}_{output_step}/idx_test.npy",
        idx_test,
    )


if __name__ == "__main__":
    data_names = ["SD", "GBA", "GLA", "CA"]
    input_time_step = 96
    output_time_steps = [12, 48, 96, 192, 672]

    for output_time_step in output_time_steps:
        for data_name in data_names:
            generate_and_split_indices_sequential(
                data_name, input_time_step, output_time_step
            )
