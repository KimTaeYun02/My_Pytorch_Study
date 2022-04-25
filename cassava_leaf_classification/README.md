# Cassava Leaf Disease Classification(Use GPU server)

#### Dataset
https://www.kaggle.com/competitions/cassava-leaf-disease-classification

#### Preprocessing(transform)
Use torchvision.transforms.Compose
'''python
transform = transforms.Compose([
            transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256,256)),
            transforms.RandomHorizontalFlip(p = 0.5),
            transforms.RandomRotation(90),
            transforms.RandomVerticalFlip(p = 0.5),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])
'''

#### Model
Use Pretrained model(resnet 152)
'''python
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1., momentum=0.9)
step_size = 4*len(train_loader)
clr = cyclical_lr(step_size, min_lr=3e-4, max_lr=3e-3)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [clr])
'''

#### Best Val Acc
Best Val Acc: 0.6265






