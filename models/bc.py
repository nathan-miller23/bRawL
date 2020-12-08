import json
import numpy as np
import torch
from torch import nn
from torch.utils import data
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from pathlib import Path
# from ppo_model import RllibPPOModel
from gym import spaces
import melee
from melee import SSBMEnv

device = "cuda" if torch.cuda.is_available() else "cpu"

def states_from_file(file):
    """Returns states, actions, dim_states, num_actions"""
    json_data = json.loads(Path(file).read_text())
    json_data = [x for x in json_data if 'next_state' in x]
    states = []
    actions = []
    for sa in json_data:
        states.append(np.array(sa["state"]["ai_1"]))
        actions.append(int(sa["ai_1"]))
    dim_states = states[0].shape
    num_actions = max(actions) + 1
    return states, actions, dim_states, num_actions
    

def states_from_folder(folder):
    """Returns states, actions, dim_states, num_actions"""
    states = []
    actions = []
    dim_states = 0
    num_actions = 0
    for file in Path(folder).iterdir():
        new_states, new_actions, _dim_states, _num_actions = states_from_file(file)
        states += new_states
        actions += new_actions
        dim_states = _dim_states
        num_actions = max(num_actions, _num_actions)
    return states, actions, dim_states, num_actions

def datasets(states, actions, batch_size):
    train_states, test_states, train_actions, test_actions = train_test_split(states, actions, test_size=0.15)
    tensor_train_states = torch.Tensor(train_states)
    tensor_train_actions = torch.Tensor(train_actions)
    tensor_test_states = torch.Tensor(test_states)
    tensor_test_actions = torch.Tensor(test_actions)

    train_dataset = data.TensorDataset(tensor_train_states, tensor_train_actions)
    train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True) 
    test_dataset = data.TensorDataset(tensor_test_states, tensor_test_actions)
    test_dataloader = data.DataLoader(test_dataset, batch_size=batch_size) 
    return train_dataloader, test_dataloader

class BCAgent(nn.Module):
    def __init__(self, agent):
        super().__init__()
        self.agent = agent
    
    def forward(self, x):
        return self.agent(x)
    
    def save_state(self, path):
        torch.save(self.agent.state_dict(), path)
    
    def load_state(self, path):
        self.agent.load_state_dict(torch.load(path))

class LinearAgent(BCAgent):
    def __init__(self, num_states, num_actions):
        agent = nn.Sequential(
            nn.Linear(num_states, 40),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(40, 40),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(40, num_actions)
        ).to(device)
        super().__init__(agent)


class LinearBufferAgent(BCAgent):
    """For use with 2D buffer"""
    def __init__(self, buffer_len, num_states, num_actions, hidden_size):
        self.buffer_len = buffer_len
        agent = nn.Sequential(
            nn.Linear(num_states * buffer_len, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, num_actions)
        ).to(device)
        super().__init__(agent)
    
    def forward(self, x):
        # Add batch dimension to gym input
        if len(x.shape) < 3:
            x = x.unsqueeze(0)
        # Unroll buffer into flat array w/o disrupting batching
        return self.agent(torch.flatten(x, start_dim=1))

class ConvBufferAgent(BCAgent):
    """For use with 2D buffer"""
    def __init__(self, buffer_len, num_states, num_actions, hidden_size):
        self.buffer_len = buffer_len
        agent = nn.Sequential(
            nn.Conv1d(1, 8, 3),
            nn.Conv1d(8, 1, 3),
            nn.Linear(num_states * buffer_len, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, num_actions)
        ).to(device)
        super().__init__(agent)
    
    def forward(self, x):
        # Add batch dimension to gym input
        if len(x.shape) < 3:
            x = x.unsqueeze(0)
        # Unroll buffer into flat array w/o disrupting batching
        return self.agent(torch.flatten(x, start_dim=1))
    
# class ConvAgent(BCAgent):
#     def __init__(self, buffer_len, num_states, num_actions):
#         size_hidden_layers = 1024
#         num_convs = 2
#         num_filters = 8
#         obs_space_shape = num_states
#         policy_modules = []
#         if num_convs > 0:
#             policy_modules.append(nn.Conv1d(obs_space_shape, num_filters, kernel_size=3, padding=1))
#             policy_modules.append(torch.nn.LeakyReLU())
#         for _ in range(num_convs-1):
#             policy_modules.append(torch.nn.Conv1d(num_filters, num_filters, kernel_size=3, padding=1))
#             policy_modules.append(torch.nn.LeakyReLU())
#         policy_modules.append(nn.Flatten())
        
#         #modules.append(nn.InstanceNorm1d(39))
#         in_size = num_filters * obs_space.shape[1] if num_convs > 0 else obs_space.shape[0] * obs_space.shape[1]
        
#         policy_modules.append(nn.Linear(in_size, size_hidden_layers))
#         policy_modules.append(torch.nn.LeakyReLU())
        
#         for i in range(num_hidden_layers - 1):
#             policy_modules.append(torch.nn.Linear(size_hidden_layers, size_hidden_layers))
#             policy_modules.append(torch.nn.LeakyReLU())

#         self._num_outputs = num_outputs
#         self.shared = nn.Sequential(*policy_modules)
#         self.policy_out = nn.Linear(size_hidden_layers, self._num_outputs)
#         self.value_out = nn.Linear(size_hidden_layers, 1)

class RLLibAgent(BCAgent):
    def __init__(self, num_states, num_actions):
        agent = RllibPPOModel(
            obs_space=spaces.Discrete(num_states),
            action_space=spaces.Discrete(num_actions),
            num_outputs=1, model_config={
              "_time_major": 0,
              "custom_model_config": {
                  "NUM_HIDDEN_LAYERS": 3,
                  "SIZE_HIDDEN_LAYERS": 128,
                  "NUM_FILTERS": 25,
                  "NUM_CONV_LAYERS": 3,
                  "HIDDEN_OUTPUT_SIZE": 4900
              }
            }, name=0)
        super().__init__(agent)
    
    def forward(self, x):
        return self.agent({"obs": x})[0]

def train(bc_agent, train_dataloader, test_dataloader, num_actions, num_epochs):
    optimizer = torch.optim.Adam(bc_agent.parameters(), lr=0.001)
    loss_func = nn.CrossEntropyLoss(weight=torch.Tensor([0.1] + [1] * (num_actions - 1)).to(device))
    train_total_losses = []
    test_total_losses = []
    for epoch in range(num_epochs):
        train_total_loss = 0
        train_steps = 0
        for s, a in train_dataloader:
            s = s.float().to(device)
            a = a.long().to(device)
        
            loss = loss_func(bc_agent(s), a)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_steps += 1
            train_total_loss += loss.detach().cpu().numpy()
        train_total_losses.append(train_total_loss / train_steps)
        
        test_total_loss = 0
        test_steps = 0
        for s, a in test_dataloader:
            s = s.float().to(device)
            a = a.long().to(device)
            loss = loss_func(bc_agent(s), a)
            
            test_steps += 1
            test_total_loss += loss.detach().cpu().numpy()
        test_total_losses.append(test_total_loss / test_steps)
        print(f"Epoch {epoch:<5} Train Loss: {train_total_losses[-1]:<5.2f} Test Loss: {test_total_losses[-1]:<5.2f}")
    plt.plot(range(len(train_total_losses)), train_total_losses)
    plt.plot(range(len(test_total_losses)), test_total_losses)
    plt.show()

def play(bc_agent, cpu_level, buffer_size, frame_skip, **kwargs):
    with torch.no_grad():
        env = SSBMEnv('../mocker/dolphin-emu.app/Contents/MacOS',
                    ssbm_iso_path='../mocker/m.iso',
                    cpu=True, cpu_level=cpu_level, every_nth=frame_skip,
                    buffer_size=buffer_size, **kwargs)
        obs = env.reset()
        done = False
        while not done:
            logits = bc_agent(torch.FloatTensor(obs['ai_1'], device=device))
            action = torch.distributions.categorical.Categorical(logits=logits).sample()
            joint_action = {'ai_1': action}
            obs, reward, done, info = env.step(joint_action)
            done = done['__all__']