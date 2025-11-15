import torch 
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from losses.sliced_score_loss import sliced_score_matching_loss, annealed_sliced_loss
from models.unet import ScoreUNet
from tqdm import tqdm

training_iteration = 100
batch_size = 128
model_path = "./score_unet.pth"
out_name = "ScoreUNet_MNIST"

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))])

train_dataset = torchvision.datasets.MNIST(root="./data",
                                           train=True,
                                           download="True",
                                           transform=transform)

trainloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=0)

test_dataset = torchvision.datasets.MNIST(root="./data",
                                           train=False,
                                           download="True",
                                           transform=transform)

testloader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=0)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = ScoreUNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

sigma_values = torch.exp(torch.linspace(torch.log(torch.tensor(10.0)),
                                        torch.log(torch.tensor(1.0)),
                                        10)).to(device)

writer = SummaryWriter(log_dir=f"./runs/{out_name}")
for epoch in tqdm(range(training_iteration)):
    for x, _ in trainloader:
        x = x.to(device)

        optimizer.zero_grad()

        sigmas = sigma_values[torch.randint(0, len(sigma_values), (x.shape[0],))]

        noise = torch.randn_like(x) * sigmas.view(-1,1,1,1)
        x_noise = x + noise
        
        loss = annealed_sliced_loss(model, x_noise, sigmas)
        loss.backward()
        optimizer.step()

  

    writer.add_scalar("Loss/train", loss.item(), epoch)
    if epoch%5 == 0:
        print(f"Epoch = {epoch} | Loss = {loss}")
        torch.save(model.state_dict(), f"./trained_models/{out_name}.pth")

writer.close()
torch.save(model.state_dict(), f"./trained_models/{out_name}.pth")
        



