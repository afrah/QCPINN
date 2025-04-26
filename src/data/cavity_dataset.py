import numpy as np
import torch
import h5py
import pandas as pd

dist = "Sobol"
domainP = 0.01
leftP = 0.15
rightP = 0.15
bottomP = 0.15
upP = 0.15
initialP = 0.15
sensorP = 0.005


def generate_sobol_sequence(low, high, n):
    soboleng = torch.quasirandom.SobolEngine(dimension=1)
    bounds = [low, high]

    input_tb = soboleng.draw(n)
    result = np.floor((bounds[0] + (bounds[1] - bounds[0]) * input_tb))
    result = [int(i) for i in result]
    return result


class CavityDatasetFromFile(object):
    def __init__(self, data_file, device):
        [domain, sensors, left, right, bottom, up, initial] = process_file(
            data_file, domainP, sensorP, leftP, rightP, bottomP, upP, initialP, dist
        )

        print(
            f"{domain.shape=}, {left.shape=}, {bottom.shape=}, {up.shape=}, {right.shape=}, {initial.shape=}, {sensors.shape=}"
        )

        max_size = np.max(
            [
                domain.shape[0],
                left.shape[0],
                bottom.shape[0],
                up.shape[0],
                initial.shape[0],
            ]
        )

        self.min_x = torch.tensor(
            np.min(domain[:, 0:3], axis=0), dtype=torch.float32
        ).to(device)
        self.max_x = torch.tensor(
            np.max(domain[:, 0:3], axis=0), dtype=torch.float32
        ).to(device)

        self.txy_domain = torch.tensor(domain[:, 0:3], dtype=torch.float32).to(device)
        self.uvp_domain = torch.tensor(domain[:, 3:6], dtype=torch.float32).to(device)

        self.txy_left = torch.tensor(left[:, 0:3], dtype=torch.float32).to(device)
        self.uvp_left = torch.tensor(left[:, 3:6], dtype=torch.float32).to(device)

        self.txy_right = torch.tensor(right[:, 0:3], dtype=torch.float32).to(device)
        self.uvp_right = torch.tensor(right[:, 3:6], dtype=torch.float32).to(device)

        self.txy_bottom = torch.tensor(bottom[:, 0:3], dtype=torch.float32).to(device)
        self.uvp_bottom = torch.tensor(bottom[:, 3:6], dtype=torch.float32).to(device)

        self.txy_up = torch.tensor(up[:, 0:3], dtype=torch.float32).to(device)
        self.uvp_up = torch.tensor(up[:, 3:6], dtype=torch.float32).to(device)

        self.txy_initial = torch.tensor(initial[:, 0:3], dtype=torch.float32).to(device)
        self.uvp_initial = torch.tensor(initial[:, 3:6], dtype=torch.float32).to(device)
        self.txy_sensors = torch.tensor(sensors[:, 0:3], dtype=torch.float32).to(device)
        self.uvp_sensors = torch.tensor(sensors[:, 3:6], dtype=torch.float32).to(device)

        # # Debugging purpose
        # print(f"CavityDatasetFromFile:__init__")
        # print(f"{self.txy_domain.shape=}, {self.uvp_domain.shape=}")
        # print(f"{self.txy_left.shape=}, {self.uvp_left.shape=}")
        # print(f"{self.txy_right.shape=}, {self.uvp_right.shape=}")
        # print(f"{self.txy_bottom.shape=}, {self.uvp_bottom.shape=}")
        # print(f"{self.txy_up.shape=}, {self.uvp_up.shape=}")
        # print(f"{self.txy_initial.shape=}, {self.uvp_initial.shape=}")

        self.size = max_size

    def __getitem__(self):
        return (
            dict(
                {
                    "txy_domain": self.txy_domain,
                    "txy_left": self.txy_left,
                    "txy_right": self.txy_right,
                    "txy_bottom": self.txy_bottom,
                    "txy_up": self.txy_up,
                    "txy_initial": self.txy_initial,
                    "txy_sensors": self.txy_sensors,
                }
            ),
            dict(
                {
                    "uvp_domain": self.uvp_domain,
                    "uvp_left": self.uvp_left,
                    "uvp_right": self.uvp_right,
                    "uvp_bottom": self.uvp_bottom,
                    "uvp_up": self.uvp_up,
                    "uvp_initial": self.uvp_initial,
                    "uvp_sensors": self.uvp_sensors,
                }
            ),
        )

    def __len__(self):
        return self.size


def process_file(
    data_file="../../data/cavity.mat",
    domainP=0.01,
    sensorP=0.005,
    leftP=0.15,
    rightP=0.15,
    bottomP=0.15,
    upP=0.15,
    initialP=0.15,
    dist="Sobol",
):
    # pleft, pRight, pBottom, pUp, pDomain, pInitial
    np.random.seed(42)
    percents = [domainP, sensorP, leftP, rightP, bottomP, upP, initialP]
    data = h5py.File(data_file, "r")  # load dataset from matlab

    domain = pd.DataFrame(data["cavity_internal"]).T.to_numpy()
    cavity_gamma0 = pd.DataFrame(data["cavity_gamma0"]).T.to_numpy()
    cavity_gamma1 = pd.DataFrame(data["cavity_gamma1"]).T.to_numpy()

    left = cavity_gamma0[cavity_gamma0[:, 1] == cavity_gamma0[:, 1].min()]
    right = cavity_gamma0[cavity_gamma0[:, 1] == cavity_gamma0[:, 1].max()]
    bottom = cavity_gamma0[
        (cavity_gamma0[:, 1] != cavity_gamma0[:, 1].min())
        & (cavity_gamma0[:, 1] != cavity_gamma0[:, 1].max())
    ]

    initial = domain[domain[:, 0] == domain[:, 0].min()]

    # total = domain.shape[0] + cavity_gamma0.shape[0] + cavity_gamma1.shape[0]

    data = [domain, domain, left, right, bottom, cavity_gamma1, initial]
    training_dataset = []
    for d, percent in zip(data, percents):
        if dist == "Sobol":
            idxi = generate_sobol_sequence(0, d.shape[0], int(d.shape[0] * percent))
        else:
            idxi = np.random.choice(
                d.shape[0], int(d.shape[0] * percent), replace=False
            )
        training_dataset.append(d[idxi, :])
        # [left, right, bottom, cavity_gamma1_df, domain, initial]
    domain = training_dataset[0]
    sensors = training_dataset[1]
    left = training_dataset[2]
    right = training_dataset[3]
    bottom = training_dataset[4]
    up = training_dataset[5]
    initial = training_dataset[6]

    ## sort training dataset by time to better capture cuasality?? not sure
    domain = domain[np.argsort(domain[:, 0])]
    sensors = sensors[np.argsort(sensors[:, 0])]
    left = left[np.argsort(left[:, 0])]
    right = right[np.argsort(right[:, 0])]
    bottom = bottom[np.argsort(bottom[:, 0])]
    up = up[np.argsort(up[:, 0])]

    # with open(dist_data_path, "wb") as file:
    #     pickle.dump([domain, sensors, left, right, bottom, up, initial], file)

    return [domain, sensors, left, right, bottom, up, initial]
