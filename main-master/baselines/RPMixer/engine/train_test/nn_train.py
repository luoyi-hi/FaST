# -*- coding: utf-8 -*-
"""
@author: 
"""


import os
import time
import joblib
import numpy as np
import torch
import torch.nn as nn
from .nn_test import nn_test
from .util import check_checkpoint


class EarlyStopping:
    """
    Early stopping

    Modified from https://github.com/cure-lab/LTSF-Linear
    """

    def __init__(self, patience=7):
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, loss_valid, model_pkl, result_pkl):
        if self.best_loss is None:
            self.best_loss = loss_valid
            self.model_pkl = model_pkl
            self.result_pkl = result_pkl

        elif loss_valid > self.best_loss:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = loss_valid
            self.counter = 0
            self.model_pkl = model_pkl
            self.result_pkl = result_pkl


def nn_train(
    model,
    model_path,
    result_path,
    config,
    get_loader,
    loss_fun,
    get_metric,
    is_inverse,
    device,
    n_worker,
):
    model.to(device)
    model.train()

    loader_train, dataset_train = get_loader("train", config, n_worker)
    print(f"len(dataset_train)={len(dataset_train)}")
    loader_valid, dataset_valid = get_loader("valid", config, n_worker)
    print(f"len(dataset_valid)={len(dataset_valid)}")
    loader_test, dataset_test = get_loader("test", config, n_worker)
    print(f"len(dataset_test)={len(dataset_test)}")

    lr = float(config["model"]["lr"])
    n_iter = len(loader_train)
    n_epoch = int(config["model"]["n_epoch"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    model_path_final = model_path.format(9999)
    result_path_final = result_path.format(9999)
    if os.path.isfile(model_path_final) and os.path.isfile(result_path_final):
        return

    early_stopper = EarlyStopping()
    check_checkpoint(model_path, n_epoch)
    start_epoch = 0
    loss_train = np.zeros(n_epoch)
    toc_train = np.zeros(n_epoch)
    for i in range(start_epoch, n_epoch):
        model_path_i = model_path.format(i)
        result_path_i = result_path.format(i)
        if os.path.isfile(model_path_i):
            print(f"loading {model_path_i}")
            model_pkl = torch.load(model_path_i, map_location="cpu")
            loss_train = model_pkl["loss_train"]
            toc_train = model_pkl["toc_train"]
            loss_valid = model_pkl["loss_valid"]
            loss_epoch = loss_train[i]
            toc_epoch = toc_train[i]

            model.load_state_dict(model_pkl["model_state_dict"])
            model.to(device)
            model.train()

            optimizer.load_state_dict(model_pkl["optimizer_state_dict"])

            if os.path.isfile(result_path_i):
                result_pkl = joblib.load(result_path_i)
                metric_test = result_pkl["metric_test"]
            else:
                valid_time = time.time()
                metric_valid = nn_test(
                    loader_valid,
                    dataset_valid,
                    model,
                    get_metric,
                    device,
                    is_inverse,
                    False,
                )
                print(f"valid time={time.time()-valid_time}")
                test_time = time.time()
                metric_test = nn_test(
                    loader_test,
                    dataset_test,
                    model,
                    get_metric,
                    device,
                    is_inverse,
                    False,
                )
                print(f"test time={time.time()-test_time}")
                result_pkl = {}
                result_pkl["loss_valid"] = loss_valid
                result_pkl["metric_test"] = metric_test
                result_pkl["metric_valid"] = metric_valid
                result_pkl["n_param"] = model.get_n_param()
                joblib.dump(result_pkl, result_path_i)

                model_pkl = {}
                model_pkl["loss_train"] = loss_train
                model_pkl["loss_valid"] = loss_valid
                model_pkl["metric_test"] = metric_test
                model_pkl["metric_valid"] = metric_valid
                model_pkl["toc_train"] = toc_train
                model_pkl["model_state_dict"] = model.state_dict()
                model_pkl["optimizer_state_dict"] = optimizer.state_dict()
                torch.save(model_pkl, model_path_i)

            mae_test = np.mean(metric_test["mae"])

            print(
                (
                    f"epoch {i + 1}/{n_epoch}, "
                    f"train loss={loss_epoch:0.4f}, "
                    f"train time={toc_epoch:0.2f}, "
                    f"valid mae={loss_valid:0.4f}, "
                    f"test mae={mae_test:0.4f}."
                )
            )

            early_stopper(loss_valid, model_pkl, result_pkl)
            # if early_stopper.early_stop or i == n_epoch - 1:
            #     torch.save(early_stopper.model_pkl, model_path_final)
            #     joblib.dump(early_stopper.result_pkl, result_path_final)
            #     print(f"early stopped at epoch {i + 1}.")
            #     return

            if i == n_epoch - 1:
                torch.save(early_stopper.model_pkl, model_path_final)
                joblib.dump(early_stopper.result_pkl, result_path_final)
                print(f"early stopped at epoch {i + 1}.")
                return
            continue

        tic = time.time()
        loss_epoch = 0
        model.train()
        for batch_x, batch_y in loader_train:
            optimizer.zero_grad()

            batch_x = batch_x.to(device, dtype=torch.float32)
            batch_y = batch_y.to(device, dtype=torch.float32)

            pred = model.forward(batch_x)
            loss = loss_fun(dataset_train, pred, batch_y, device)

            loss.backward()
            optimizer.step()
            loss_epoch += loss.item()

        loss_epoch /= n_iter
        toc_epoch = time.time() - tic

        loss_train[i] = loss_epoch
        toc_train[i] = toc_epoch
        valid_loss_time = time.time()
        loss_valid = nn_test(
            loader_valid, dataset_valid, model, get_metric, device, is_inverse, True
        )
        print(f"valid loss time={time.time()-valid_loss_time}")
        valid_time = time.time()
        metric_valid = nn_test(
            loader_valid, dataset_valid, model, get_metric, device, is_inverse, False
        )
        print(f"valid time={time.time()-valid_time}")
        test_time = time.time()
        metric_test = nn_test(
            loader_test, dataset_test, model, get_metric, device, is_inverse, False
        )
        print(f"test time={time.time()-test_time}")
        model_pkl = {}
        model_pkl["loss_train"] = loss_train
        model_pkl["loss_valid"] = loss_valid
        model_pkl["metric_test"] = metric_test
        model_pkl["metric_valid"] = metric_valid
        model_pkl["toc_train"] = toc_train
        model_pkl["model_state_dict"] = model.state_dict()
        model_pkl["optimizer_state_dict"] = optimizer.state_dict()
        torch.save(model_pkl, model_path_i)

        result_pkl = {}
        result_pkl["loss_valid"] = loss_valid
        result_pkl["metric_test"] = metric_test
        result_pkl["metric_valid"] = metric_valid
        result_pkl["n_param"] = model.get_n_param()
        joblib.dump(result_pkl, result_path_i)

        mae_test = np.mean(metric_test["mae"])

        print(
            (
                f"epoch {i + 1}/{n_epoch}, "
                f"train loss={loss_epoch:0.4f}, "
                f"train time={toc_epoch:0.2f}, "
                f"valid mae={loss_valid:0.4f}, "
                f"test mae={mae_test:0.4f}."
            )
        )

        early_stopper(loss_valid, model_pkl, result_pkl)

        # if early_stopper.early_stop or i == n_epoch - 1:
        #     torch.save(early_stopper.model_pkl, model_path_final)
        #     joblib.dump(early_stopper.result_pkl, result_path_final)
        #     print(f"early stopped at epoch {i + 1}.")
        #     return
        if i == n_epoch - 1:
            torch.save(early_stopper.model_pkl, model_path_final)
            joblib.dump(early_stopper.result_pkl, result_path_final)
            print(f"early stopped at epoch {i + 1}.")
            return
