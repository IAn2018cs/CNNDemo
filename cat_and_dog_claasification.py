import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from model.ResNet import *

root_dir = 'kagglecatsanddogs_5340/PetImages'

loss_fig_root = "loss_fig"
model_parameter_root = "model_parameter"
temp_data_root = "temp_data"

cat_dir = os.path.join(root_dir, 'Cat')
dog_dir = os.path.join(root_dir, "Dog")
cat_image_num = 4000
dog_image_num = 4000
# 图片训练的尺寸
fig_size = (224, 224)


# 采样作为训练集
def sample(path, number):
    for root, dirs, files in os.walk(path):
        temp = np.random.choice(files, size=number, replace=False)
        return [os.path.join(root, image_path) for image_path in temp]


def show_image(image_path):
    image = cv2.imread(image_path)
    if image is not None:
        resized_image = cv2.resize(image, fig_size)
        train_image = np.transpose(resized_image, (2, 0, 1))

        print("image: ", image.shape, "train image: ", train_image.shape)
        cv2.imshow("target image", resized_image)
        cv2.imshow("image", image)

        # 等待按下任意键
        cv2.waitKey(0)
        # 关闭窗口
        cv2.destroyAllWindows()


def generate_train_data(cat_list, dog_list, mode="train"):
    # 对应的标签和数据
    data_list = []
    y_label = []
    for item in cat_list:
        image_ = cv2.imread(item)
        if image_ is None:
            print("猫 图片坏了", item)
            continue
        data_list.append(np.transpose(cv2.resize(image_, fig_size), (2, 0, 1)))
        y_label.append(np.array(1))
    for item in dog_list:
        image_ = cv2.imread(item)
        if image_ is None:
            print("狗 图片坏了", item)
            continue
        data_list.append(np.transpose(cv2.resize(image_, fig_size), (2, 0, 1)))
        y_label.append(np.array(0))

    data_set = torch.tensor(np.stack(data_list, axis=0))
    data_label = torch.tensor(np.stack(y_label, axis=0))
    torch.save(data_set, os.path.join(temp_data_root, f"{mode}_data.pth"))
    torch.save(data_label, os.path.join(temp_data_root, f"{mode}_label.pth"))

    return data_set, data_label


def train(train_data, train_data_label, epoch, net, optimizer, criterion, batch_size, iterations=10):
    total_loss = 0
    data_length = len(train_data_label)
    with tqdm(total=iterations, desc='Training Progress', unit="iter") as bar:
        for item in range(iterations):
            bar.update()
            indices = torch.randperm(data_length)[:batch_size]
            x = train_data[indices, :, :, :].to(torch.float32).to(device)
            label = train_data_label[indices].to(torch.long).to(device)
            y_hat = net(x)
            loss = criterion(y_hat, label)
            total_loss += loss.detach().cpu()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    print(f"epoch:{epoch + 1}   loss:{(total_loss / iterations) : 3f}")
    return total_loss / iterations


def test(device, retry=False):
    test_batch_size = 16
    test_data_path = os.path.join(temp_data_root, "test_data.pth")
    test_label_path = os.path.join(temp_data_root, "test_label.pth")
    if os.path.exists(test_data_path) and os.path.exists(test_label_path) and not retry:
        test_data = torch.load(test_data_path)
        test_label = torch.load(test_label_path)
    else:
        cat_list = sample(cat_dir, cat_image_num)
        dog_list = sample(dog_dir, dog_image_num)
        test_data, test_label = generate_train_data(cat_list, dog_list, mode="test")

    net_parameter = torch.load(os.path.join(model_parameter_root, f"{net_name}.pt"), map_location=device)
    net.load_state_dict(net_parameter)
    net.eval()
    net.to(device)
    accuracy_num = 0
    for i in range(int(len(test_label) / test_batch_size)):
        if i != int(len(test_label) / test_batch_size):
            x = test_data[i * test_batch_size: (i + 1) * test_batch_size, :, :, :]
            y = test_label[i * test_batch_size:(i + 1) * test_batch_size]
        else:
            x = test_data[i * test_batch_size: -1, :, :, :]
            y = test_label[i * test_batch_size: -1]
        result = F.softmax(net(x.to(torch.float32).to(device))).cpu()
        result = torch.max(result, dim=1)[1] == y
        accuracy_num += torch.sum(result).item()
    print(f"accuracy:{accuracy_num / len(test_label) * 100 : 3f}%")


def predict(image_path, device):
    net_parameter = torch.load(os.path.join(model_parameter_root, f"{net_name}.pt"), map_location=device)
    net.load_state_dict(net_parameter)
    net.eval()
    net.to(device)
    if os.path.exists(image_path):
        image_ = cv2.resize(cv2.imread(image_path), fig_size)
        cv2.imshow("image", image_)
        cv2.waitKey(0)
        image_ = np.transpose(image_, (2, 0, 1))
        image_ = np.expand_dims(image_, axis=0)
        image_ = torch.tensor(image_).to(torch.float32).to(device)
        _, result = torch.max(net(image_), dim=1)
        result = "猫" if result else "狗"
        print(f"预测结果为 {result}")
    else:
        print("path is not exist")


if __name__ == '__main__':
    # cat_image_file_path_list = sample(cat_dir, cat_image_num)
    # dog_image_file_path_list = sample(dog_dir, dog_image_num)

    "展示图像"
    # for i in range(100):
    #     show_image(dog_image_file_path_list[i])
    "生成图像数据"
    # generate_train_data(cat_image_file_path_list, dog_image_file_path_list, mode="train")

    # 将网络转移到gpu中去
    net = ResNet()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    train_data = torch.load(os.path.join(temp_data_root, "train_data.pth"), map_location=device)
    train_data_label = torch.load(os.path.join(temp_data_root, "train_label.pth"))

    "测试一下网络是否可行"
    # train_data = train_data[0:64, :, :, :].to(torch.float).to(device)
    # for layer in net.net:
    #     train_data = forward(layer, train_data)
    #     print(layer.__class__.__name__, train_data.shape)

    training = False
    if training:
        optimizer = torch.optim.Adam(net.parameters(), lr=0.03)
        criterion = torch.nn.CrossEntropyLoss()
        batch_size = 128
        assert len(train_data) == len(
            train_data_label), f"数据维度应该一致 train_data:{train_data.shape},y_label:{train_data_label.shape}"
        data_length = len(train_data)
        loss_list = []
        for epoch in range(400):
            loss = train(train_data=train_data, train_data_label=train_data_label,
                         batch_size=batch_size, epoch=epoch, net=net, optimizer=optimizer, criterion=criterion)
            loss_list.append(loss)

        torch.save(net.state_dict(), os.path.join(model_parameter_root, f'{net_name}.pt'))

        plt.title(f"BatchSize: {batch_size}")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.plot(range(len(loss_list)), loss_list)
        plt.savefig(os.path.join(loss_fig_root, f"{net_name}--loss.jpg"))
        plt.show()

    test(device, retry=False)

    # image_path = 'kagglecatsanddogs_5340/PetImages/Cat/2010.jpg'
    # predict(image_path, device)
