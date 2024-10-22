from .data import *


def get_dataset_transforms(args):
    forget_class_int = -1
    if args.dataset == "cifar100":
        dataSet = DatasetCifar100FullClassUnlearning
        transform_train = transform_train_cifar
        transform_valid = transform_valid_cifar
        if hasattr(args, "forget_class"): forget_class_int = CIFAR100_CLASS_DICT[args.forget_class]

    elif args.dataset == "cifar20":
        dataSet = DatasetCifar20FullClassUnlearning
        transform_train = transform_train_cifar
        transform_valid = transform_valid_cifar
        if hasattr(args, "forget_class"): forget_class_int = CIFAR100_CLASS_DICT[args.forget_class]
    else:
        raise NotImplementedError(f"Not implement this dataset : {args.dataset}")

    return dataSet, transform_train, transform_valid, forget_class_int


def set_resumeCKPT(args):
    mode = "init" if "retrain" in args.code_file else "last"
    if args.resumeCKPT is None:
        if args.dataset == "cifar20":
            args.resumeCKPT = f"ckpt/ResNet18/cifar20/Vanilla/24_08_05_22_40_04.65_sgd_lr-0.1_epoch-100_batchsize-128_wd-0.0005/model-{mode}.pth"
        elif args.dataset == "cifar100":
            args.resumeCKPT = f"ckpt/ResNet18/cifar100/Vanilla/24_08_05_21_55_49.20_sgd_lr-0.1_epoch-100_batchsize-128_wd-0.0005/model-{mode}.pth"
        else:
            raise NotImplementedError


def get_dataset_info(dataloader):
    if isinstance(dataloader.dataset, ConcatDataset):
        info = f"Dataset Length:{len(dataloader.dataset)}\tMax Label 1:{min(dataloader.dataset.label1)}\tMin Label 1:{max(dataloader.dataset.label1)}\tMax Label 2:{min(dataloader.dataset.label2)}\tMin Label 2:{max(dataloader.dataset.label2)}"
        info += "\n" + str(np.bincount(dataloader.dataset.label1))
        info += "\n" + str(np.bincount(dataloader.dataset.label2))
    else:
        min_label = min(dataloader.dataset.label)
        max_label = max(dataloader.dataset.label)
        info = f"Dataset Length:{len(dataloader.dataset)}\tMax Label:{max_label}\tMin Label:{min_label}"
        info += "\n" + str(np.bincount(dataloader.dataset.label))
    return info + "\n"


def get_w_wbias(init_model, pretrain_model, retrain_model):
    w_0 = init_model['net']['net.linear.weight']  # 10 x 512
    w_p = pretrain_model['net']['net.linear.weight']  # 10 x 512
    w_r = retrain_model['net']['net.linear.weight']  # 10 x 512
    b_0 = init_model['net']['net.linear.bias']  # 10
    b_p = pretrain_model['net']['net.linear.bias']  # 10
    b_r = retrain_model['net']['net.linear.bias']  # 10
    w_0_wbias = torch.cat([w_0, b_0.reshape(-1, 1)], dim=1)  # 10 x 513
    w_p_wbias = torch.cat([w_p, b_p.reshape(-1, 1)], dim=1)  # 10 x 513
    w_r_wbias = torch.cat([w_r, b_r.reshape(-1, 1)], dim=1)  # 10 x 513
    w_0_wbias = torch.t(w_0_wbias)
    w_p_wbias = torch.t(w_p_wbias)
    w_r_wbias = torch.t(w_r_wbias)

    return w_0_wbias, w_p_wbias, w_r_wbias