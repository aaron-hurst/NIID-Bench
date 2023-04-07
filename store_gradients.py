import datetime
import json
import logging
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from model import FcNet, SimpleCNN, SimpleCNNMNIST, ModerateCNN, ModerateCNNMNIST
from resnetcifar import ResNet50_cifar10
from utils import partition_data, get_dataloader, compute_accuracy
from vggmodel import vgg11, vgg16

RESULTS_DIR = os.path.join(
    # "C:",
    # os.sep,
    # "Users",
    # "au686379",
    # "OneDrive - Aarhus Universitet",
    # "Documents",
    # "04 Research",
    # "results",
    # "federated_learning",
    "results"
)

PARAMS = {
    "algorithm": "fedavg",
    "model": "simple-cnn",
    "dataset": "femnist",
    "partition": "homo",  # TODO implement author/source-based non-iid environment
    "batch_size": 64,
    "lr": 0.01,
    "reg": 1e-5,  # L2 regularization strength
    "rho": 0,  # parameter controlling the momentum SGD
    "epochs": 1,  # number of local epochs
    "n_clients": 2,
    "clients_sample_ratio": 1,  # fraction of clients that participate in each round
    "comm_rounds": 50,  # maximum number of communication rounds
    "init_seed": 0,  # random seed
    "dropout_p": 0,  # dropout probability
    "data_dir": "./data/",  # where to store downloaded datasets
    "model_dir": "./models/",  # where to store models
    "device": "cpu",
    "optimiser": "sgd",
}


def init_nets(dataset, model, n_parties, dropout_p):
    nets = {net_i: None for net_i in range(n_parties)}
    for net_i in range(n_parties):
        if model == "mlp":
            if dataset == "covtype":
                input_size = 54
                output_size = 2
                hidden_sizes = [32, 16, 8]
            elif dataset == "a9a":
                input_size = 123
                output_size = 2
                hidden_sizes = [32, 16, 8]
            elif dataset == "rcv1":
                input_size = 47236
                output_size = 2
                hidden_sizes = [32, 16, 8]
            elif dataset == "SUSY":
                input_size = 18
                output_size = 2
                hidden_sizes = [16, 8]
            net = FcNet(input_size, hidden_sizes, output_size, dropout_p)
        elif model == "vgg":
            net = vgg11()
        elif model == "simple-cnn":
            if dataset in ("cifar10", "cinic10", "svhn"):
                net = SimpleCNN(
                    input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=10
                )
            elif dataset in ("mnist", "femnist", "fmnist"):
                net = SimpleCNNMNIST(
                    input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=10
                )
            elif dataset == "celeba":
                net = SimpleCNN(
                    input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=2
                )
        elif model == "vgg-9":
            if dataset in ("mnist", "femnist"):
                net = ModerateCNNMNIST()
            elif dataset in ("cifar10", "cinic10", "svhn"):
                # print("in moderate cnn")
                net = ModerateCNN()
            elif dataset == "celeba":
                net = ModerateCNN(output_dim=2)
        elif model == "resnet":
            net = ResNet50_cifar10()
        elif model == "vgg16":
            net = vgg16()
        else:
            print("not supported yet")
            exit(1)
        nets[net_i] = net

    model_meta_data = []
    layer_type = []
    for k, v in nets[0].state_dict().items():
        model_meta_data.append(v.shape)
        layer_type.append(k)
    return nets, model_meta_data, layer_type


def train_net(
    net,
    net_id,
    train_dataloader,
    test_dataloader,
    epochs,
    lr,
    args_optimizer,
    out_dir,
    comm_round,
    device="cpu",
):
    # Compute and report pre-training accuracy
    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc = compute_accuracy(net, test_dataloader, device=device)
    logger.info(f">> Pre-training training accuracy: {train_acc:.4f}")
    logger.info(f">> Pre-training test accuracy: {test_acc:.4f}")

    # Initialise optimiser
    if args_optimizer == "adam":
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, net.parameters()),
            lr=lr,
            weight_decay=PARAMS["reg"],
        )
    elif args_optimizer == "amsgrad":
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, net.parameters()),
            lr=lr,
            weight_decay=PARAMS["reg"],
            amsgrad=True,
        )
    elif args_optimizer == "sgd":
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, net.parameters()),
            lr=lr,
            momentum=PARAMS["rho"],
            weight_decay=PARAMS["reg"],
        )
    criterion = nn.CrossEntropyLoss().to(device)

    cnt = 0
    if type(train_dataloader) == type([1]):
        pass
    else:
        train_dataloader = [train_dataloader]

    for epoch in range(epochs):
        epoch_loss_collector = []
        for tmp in train_dataloader:
            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(device), target.to(device)  # move data tensor to GPU
                optimizer.zero_grad()  # sets all gradients in the optimiser to zero
                x.requires_grad = True  # gradients will be computed
                target.requires_grad = False  # gradients not computed
                target = target.long()  # convert to long int
                out = net(x)  # nn.Module.__call__, essentially runs "forward"
                loss = criterion(out, target)  # evaluate loss
                loss.backward()  # backprop, running this makes the gradients available

                optimizer.step()  # updates the model
                cnt += 1
                epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info(f"Epoch: {epoch} loss: {epoch_loss:.3f}")

    # Export model weights and weight gradients
    out_dir_gradients = os.path.join(
        out_dir,
        "gradients",
        f"round_{comm_round:04d}",
        f"client_{net_id:02d}",
    )
    out_dir_weights = os.path.join(
        out_dir,
        "weights",
        f"round_{comm_round:04d}",
        f"client_{net_id:02d}",
    )
    os.makedirs(out_dir_weights, exist_ok=True)
    os.makedirs(out_dir_gradients, exist_ok=True)
    for param_name, param in zip(net.state_dict(), net.parameters()):
        torch.save(param, os.path.join(out_dir_weights, param_name + ".pt"))
        torch.save(param.grad, os.path.join(out_dir_gradients, param_name + ".pt"))

    # Compute and report accuracy
    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc = compute_accuracy(net, test_dataloader, device=device)
    logger.info(f">> Post-training training accuracy: {train_acc:.4f}")
    logger.info(f">> Post-training test accuracy: {test_acc:.4f}")

    net.to("cpu")
    logger.info("** Training complete **")
    return train_acc, test_acc


def local_train_net(
    nets,
    selected,
    net_dataidx_map,
    dataset,
    data_dir,
    batch_size,
    n_epoch,
    out_dir,
    comm_round,
    test_dl=None,
    device="cpu",
):
    avg_acc = 0.0
    for net_id, net in nets.items():
        if net_id not in selected:
            continue
        dataidxs = net_dataidx_map[net_id]
        net.to(device)  # move the model to cuda device
        logger.info(f"Training network {net_id}. N_training: {len(dataidxs)}")

        # Data loaders
        train_dl_local, _, _, _ = get_dataloader(
            dataset, data_dir, batch_size, 32, dataidxs
        )

        # Perform training
        trainacc, test_acc = train_net(
            net,
            net_id,
            train_dl_local,
            test_dl,
            n_epoch,
            PARAMS["lr"],
            PARAMS["optimiser"],
            out_dir,
            comm_round,
            device=device,
        )
        logger.info(f"Net {net_id} final test accuracy {test_acc:.4f}")
        avg_acc += test_acc

    # Compute average accuracy across all
    avg_acc /= len(selected)
    logger.info(f"Average accuracy for current round: {avg_acc:.4f}")

    nets_list = list(nets.values())
    return nets_list


if __name__ == "__main__":
    # Configure logging
    ts = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    out_dir = os.path.join(
        RESULTS_DIR, PARAMS["algorithm"], PARAMS["model"], PARAMS["dataset"], ts
    )
    os.makedirs(out_dir, exist_ok=True)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    log_file_name = f"experiment_log-{ts}.log"
    logging.basicConfig(
        filename=os.path.join(out_dir, log_file_name),
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        level=logging.DEBUG,
        filemode="w",
    )
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    device = torch.device(PARAMS["device"])
    seed = PARAMS["init_seed"]
    logger.info(f"Device: {device}")
    logger.info("#" * 100)

    # Export parameters
    with open(os.path.join(out_dir, "arguments.json"), "w") as f:
        json.dump(PARAMS, f, indent=4)

    # Setup additional directories
    os.makedirs(PARAMS["model_dir"], exist_ok=True)
    os.makedirs(PARAMS["data_dir"], exist_ok=True)

    # Initialise random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    # Partition data
    # TODO inplement source/author-based non-iid environment
    logger.info("Partitioning data")
    X_train, y_train, X_test, y_test, net_dataidx_map, _ = partition_data(
        PARAMS["dataset"],
        PARAMS["data_dir"],
        out_dir,
        PARAMS["partition"],
        PARAMS["n_clients"],
    )
    n_classes = len(np.unique(y_train))
    train_dl_global, test_dl_global, train_ds_global, test_ds_global = get_dataloader(
        PARAMS["dataset"], PARAMS["data_dir"], PARAMS["batch_size"], 32
    )  # 32 is the text batch size
    print("len train_dl_global:", len(train_ds_global))
    data_size = len(test_ds_global)

    # Initialise client and server models
    logger.info("Initializing nets")
    nets, local_model_meta_data, layer_type = init_nets(
        PARAMS["dataset"], PARAMS["model"], PARAMS["n_clients"], PARAMS["dropout_p"]
    )
    global_models, global_model_meta_data, global_layer_type = init_nets(
        PARAMS["dataset"], PARAMS["model"], 1, 0
    )
    global_model = global_models[0]
    global_para = global_model.state_dict()
    for net_id, net in nets.items():
        net.load_state_dict(global_para)  # initialise clients with same initial model

    # Run Federated Learning
    for round in range(PARAMS["comm_rounds"]):
        logger.info(f"Starting communication round: {round}")

        # Store current global model weights (start of communication round)
        out_dir_global_model = os.path.join(
            out_dir,
            "weights",
            f"round_{round:04d}",
        )
        os.makedirs(out_dir_global_model, exist_ok=True)
        for param_name, param in zip(global_para, global_model.parameters()):
            torch.save(param, os.path.join(out_dir_global_model, param_name + ".pt"))

        # Randomly select a subset of clients
        arr = np.arange(PARAMS["n_clients"])
        np.random.shuffle(arr)
        selected = arr[: int(PARAMS["n_clients"] * PARAMS["clients_sample_ratio"])]

        # Update model for selected clients
        global_para = global_model.state_dict()
        for idx in selected:
            nets[idx].load_state_dict(global_para)

        # Train local models
        local_train_net(
            nets,
            selected,
            net_dataidx_map,
            PARAMS["dataset"],
            PARAMS["data_dir"],
            PARAMS["batch_size"],
            PARAMS["epochs"],
            out_dir,
            round,
            test_dl=test_dl_global,
            device=device,
        )

        # Update global model
        # NOTE This currently works by model average. It does not use the gradients.
        # I would like to use (i.e. communicate and update using) gradients because I
        # think that this will be more compressible for GD.
        total_data_points = sum([len(net_dataidx_map[r]) for r in selected])
        fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in selected]
        for idx in range(len(selected)):
            net_para = nets[selected[idx]].cpu().state_dict()
            if idx == 0:
                for key in net_para:
                    global_para[key] = net_para[key] * fed_avg_freqs[idx]
            else:
                for key in net_para:
                    global_para[key] += net_para[key] * fed_avg_freqs[idx]
        global_model.load_state_dict(global_para)
        logger.info("Global n_training: %d" % len(train_dl_global))
        logger.info("Global n_test: %d" % len(test_dl_global))
        global_model.to(device)
        train_acc = compute_accuracy(global_model, train_dl_global, device=device)
        test_acc, conf_matrix = compute_accuracy(
            global_model, test_dl_global, get_confusion_matrix=True, device=device
        )

        logger.info(f">> Global Model Train accuracy: {train_acc:.4f}")
        logger.info(f">> Global Model Test accuracy: {test_acc:.4f}")
