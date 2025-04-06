import torch
import torch.nn as nn
import os

def searchForMaxIteration(folder):
    saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder)]
    return max(saved_iters)


class LightModel(nn.Module):
    def __init__(self, num_images=1600):
        super().__init__()
        self.light_map = nn.Parameter(torch.ones(num_images, 224, 224).cuda())
        self.light_gamma = nn.Parameter(torch.zeros(num_images, 2).cuda())
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.optimizer = torch.optim.Adam([
            {'params': self.light_map, 'lr': 0.001, "name": "appear_map"},
            {'params': self.light_gamma, 'lr': 0.001, "name": "appear_ab"},
            {'params': self.cnn.parameters(), 'lr': 0.001, "name": "cnn_params"},
        ], betas=(0.9, 0.99))
    def save_weights(self, model_path, iteration):
        out_weights_path = os.path.join(model_path, "light_model/iteration_{}".format(iteration))
        os.makedirs(out_weights_path, exist_ok=True)
        print(f"save light model. path: {out_weights_path}")
        torch.save(self.state_dict(), os.path.join(out_weights_path, 'light.pth'))

    def load_weights(self, model_path, iteration=-1):
        if iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(model_path, "light_model"))
        else:
            loaded_iter = iteration
        weights_path = os.path.join(model_path, "light_model/iteration_{}/light.pth".format(loaded_iter))
        state_dict = torch.load(weights_path)
        self.load_state_dict(state_dict)