from nn.NetTrainer import Net, transform
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
from PIL import Image


def getImagePred(input_img):
    # Load the image
    net = Net()
    net.load_state_dict(torch.load('nn\my_model.pth'))
    print("net loaded")
    img = Image.open(input_img)

    # Convert to grayscale and resize
    img = img.convert('L').resize((28, 28))

    # Convert to a tensor and normalize
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])
    tensor = transform(img)

    # Add a batch dimension
    tensor = tensor.unsqueeze(0)
    tensor_copy = torch.clone(tensor)
    # Create the model and run the inference

    tensor_copy = tensor_copy.squeeze()
    tensor_copy = tensor_copy.numpy()

    # Display the image
    plt.imshow(tensor_copy, cmap='gray')
    plt.show()

    output = net(tensor)

    # Print the predicted image, torch.max(output.data, 1)[1] returns a 1d tensor
    return int(torch.max(output.data, 1)[1])


if __name__ == "__main__":
    print("readasdsadsa")
# train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
# test_dataset = datasets.MNIST('data', train=False, download=True, transform=transform)

# Returns 10000 images total /// total_images//batchsize is total images checked
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=True)

# with torch.no_grad():
#     for data in test_loader:
#         images, labels = data
#         outputs = net(images)
#         print(torch.max(outputs.data, 1)[1])
#         for i in range(len(images)):
#              print(i)
#              plt.imshow(images[i][0], cmap='gray')
#              plt.show()




# for data in test_loader:
#         #print(data)
#         images, labels = data
#         for i in range(len(images)):
#             plt.imshow(images[i][0], cmap='gray')
#             plt.show()
#             print(f"Label: {labels[i]}")
#         print(f"images : {images}")
#         print(f"labels : {labels}")