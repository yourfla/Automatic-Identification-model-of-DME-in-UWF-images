import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from model import SwinTransformerV2 as create_model
from utils import train_one_epoch, evaluate
from my_dataset import AugmentedDiabeticMacularEdemaDataset  # 导入 MyDataSet 类


def read_split_data(data_path, incorrect_samples=None):
    # 读取Excel文件
    df = pd.read_excel(os.path.join(data_path, 'Train-c.xlsx'))

    # 获取图像路径和标签
    images_path = df['image'].tolist()
    images_label = df['diabetic macular edema'].tolist()

    # 将路径前缀加上根目录
    images_path = [os.path.join(data_path, p) for p in images_path]

    # 拆分训练和验证集

    train_images_path, val_images_path, train_images_label, val_images_label = train_test_split(
        images_path, images_label, test_size=0.2, random_state=42)

    return train_images_path, train_images_label, val_images_path, val_images_label, incorrect_samples



def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if not os.path.exists("./weights"):
        os.makedirs("./weights")

    tb_writer = SummaryWriter()

    incorrect_samples = []

    train_images_path, train_images_label, val_images_path, val_images_label, incorrect_samples = read_split_data(args.data_path, incorrect_samples)

    img_size = 512
    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 随机仿射变换
            transforms.RandomPerspective(distortion_scale=0.5, p=0.5, interpolation=3),  # 随机透视变换
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.Resize(int(img_size * 1.143)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    # 实例化训练数据集
    train_dataset = AugmentedDiabeticMacularEdemaDataset(
        csv_file=os.path.join(args.data_path, 'Train-c.xlsx'),
        root_dir=args.data_path,
        transform=data_transform["train"],
        num_augments=args.num_augments,
        incorrect_samples=incorrect_samples,
        repeat_factor=5
    )

    # 实例化验证数据集
    val_dataset = AugmentedDiabeticMacularEdemaDataset(
        csv_file=os.path.join(args.data_path, 'Train-c.xlsx'),
        root_dir=args.data_path,
        transform=data_transform["val"],
        num_augments=args.num_augments,
        incorrect_samples=None,  # 如果没有错误样本可以传 None
        repeat_factor=1  # 验证集通常不需要重复
    )

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # 数据加载的进程数
    print(f'每个进程使用 {nw} 个数据加载工作者')

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    model = create_model(img_size=512, num_classes=args.num_classes).to(device)  # 修改输入图像大小

    if args.weights != "":
        assert os.path.exists(args.weights), f"权重文件: '{args.weights}' 不存在."
        weights_dict = torch.load(args.weights, map_location=device)["model"]
        for k in list(weights_dict.keys()):
            if "head" in k:
                del weights_dict[k]
        # 处理形状不匹配的参数
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in weights_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)

    if args.freeze_layers:
        for name, para in model.named_parameters():
            if "head" not in name:
                para.requires_grad_(False)
            else:
                print(f"训练 {name}")

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=5E-2)

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)

        val_loss, val_acc = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)

        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        torch.save(model.state_dict(), f"./weights/model-{epoch}.pth")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--num_augments', type=int, default=20)
    # 数据集所在根目录
    parser.add_argument('--data-path', type=str, default="./data/train-c")

    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights', type=str, default='./swinv2_tiny_patch4_window16_256.pth', help='initial weights path')
    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)