import pandas as pd
import torch
import torch.nn as nn
from sklearn import preprocessing
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import random
from tabulate import tabulate
from tqdm import tqdm


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
test_batch_size = 16

print(device)
data = pd.read_csv('data/test_dataset.csv')

class RecommendationModel(nn.Module):
    def __init__(self, n_users, n_movies):
        """
        Args:
            n_users (int): number of unique users
            n_movies (int): number of unique movies
        """
        super().__init__()

        self.user_embed = nn.Embedding(n_users, 96)
        self.movie_embed = nn.Embedding(n_movies, 64)

        self.fc1 = nn.Linear(160, 32)
        self.drop1 = nn.Dropout(p=0.5)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(32, 32)
        self.drop2 = nn.Dropout(p=0.5)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, 1)


    def forward(self, users, movies):
        """
        Forward function

        Args:
            users (torch.Tensor): ids of users
            movies (torch.Tensor): ids of movies

        Returns:
            float: rating of the input movie
        """

        # Embedding of user id
        user_id_embeds = self.user_embed(users)
        # Embedding of movie id
        movie_id_embeds = self.movie_embed(movies)

        x = torch.cat([user_id_embeds, movie_id_embeds], dim=1)

        x = self.fc1(x)
        x = self.drop1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        x = self.relu2(x)
        x = self.fc3(x)

        return x
    
class MovieDataset(Dataset):
    """
    Custom Movie Dataset. 
    """
    def __init__(self, user_ids, movie_ids, ratings):
        """
        Args:
            user_ids: ids of users
            movie_ids: ids of movies
            ratings: ratings of the corresponding movies
        """
        self.user_ids = user_ids
        self.movie_ids = movie_ids
        self.ratings = ratings

    def __len__(self):
        """
        Returns length of the dataset

        Returns:
            _type_: int
        """
        return len(self.user_ids)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): index of the required item

        Returns:
            _type_: dict
            user_ids: ids of the users
            movie_ids: ids of the movies
            ratings: ratings of the corresponding movies
        """
        sample = {
            "user_ids": torch.tensor(self.user_ids[idx], dtype=torch.long),
            "movie_ids": torch.tensor(self.movie_ids[idx], dtype=torch.long),
            "ratings": torch.tensor(self.ratings[idx], dtype=torch.float32)
        }
        return sample

test_dataset = MovieDataset(
    user_ids=data.user_id.values,
    movie_ids=data.movie_id.values,
    ratings=data.rating.values,
)


test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=test_batch_size,
    shuffle=False,
    num_workers=2,
    drop_last=True
)

model = RecommendationModel(
    n_users=943,
    n_movies=1682,
).to(device)
model_path = '../models/best_model.pth'
model.load_state_dict(torch.load(model_path))

loss_func = nn.MSELoss()

test_running_loss = 0

model.eval()
test_running_loss = 0
with torch.no_grad():
    for test_data in tqdm(test_loader):
        output = model(
            test_data["user_ids"].to(device),
            test_data["movie_ids"].to(device)
                    )
        rating = test_data["ratings"].view(test_batch_size, -1).to(torch.float32).to(device)
        loss = torch.sqrt(loss_func(output, rating))
        test_running_loss += loss.sum().item()
print(f"Test loss: {test_running_loss/len(test_loader)}")