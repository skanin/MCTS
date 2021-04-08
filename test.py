# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# import time 
# import torchvision.models as models
# import torchvision.datasets as datasets
# import torchvision.transforms as transforms


# model = models.resnet50()
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

# dataset = datasets.FakeData(
#     size=1000,
#     transform=transforms.ToTensor())
# loader = DataLoader(
#     dataset,
#     num_workers=1,
#     pin_memory=True
# )

# # model.to('cuda')


# if __name__ == '__main__':
#     before_time = time.time()

#     for data, target in loader:
#         data = data# .to('cuda', non_blocking=True)
#         target = target# .to('cuda', non_blocking=True)
#         optimizer.zero_grad()
#         output = model(data)
#         loss = criterion(output, target)
#         loss.backward()
#         optimizer.step()

#     print(f'Done: {time.time() - before_time}')

import keyboard

while True:
    try:
        while keyboard.is_pressed('w') and keyboard.is_pressed('d'):
            print('You are pressing W and D')
            break 
    except:
        break 