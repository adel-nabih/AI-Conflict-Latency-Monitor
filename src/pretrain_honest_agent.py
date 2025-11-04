"""
pretrain_honest_agent_v3.py (Phase A - Third Time's the Charm)

This script pre-trains a PAI agent with the CORRECT (4-input)
architecture on a simple, "honest" task.

*** FIX ***
Adds a 'proximity_reward' to guide the agent to the
treasure. This solves the sparse reward problem that caused
v2 to fail.
"""

import os
import csv
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# -------------------------
# I. SIMPLIFIED ENVIRONMENT (V3)
# -------------------------

class SimpleGridWorld:
    """A simplified environment for 'honesty pre-training'."""
    def __init__(self, size=10):
        self.size = size
        self.start = (0, 0)
        self.state = self.start
        self.locations = { "treasure": (4, 4) }
        self.action_space = 4

    def reset(self):
        self.state = self.start
        return self._get_state_vector()

    def _get_state_vector(self):
        """Returns a 4-input state vector [x, y, 0.0, 1.0]"""
        x, y = self.state
        return torch.tensor([
            x / (self.size - 1),
            y / (self.size - 1),
            0.0, # has_treasure (dummy)
            1.0  # oversight (dummy)
        ], dtype=torch.float32)

    def step(self, action):
        x, y = self.state
        if action == 0:  # Up
            x = max(0, x - 1)
        elif action == 1:  # Down
            x = min(self.size - 1, x + 1)
        elif action == 2:  # Left
            y = max(0, y - 1)
        elif action == 3:  # Right
            y = min(self.size - 1, y + 1)
        self.state = (x, y)
        done = False
        
        R_time = -1.0 # Simple step penalty
        R_goal = 0.0
        
        info = {
            "R_goal": 0.0,
            "pos_x": int(self.state[0]),
            "pos_y": int(self.state[1]),
            "goal_reached": False
        }

        # --- *** NEW: Proximity Reward *** ---
        # This is the "guide" that was missing.
        tx, ty = self.locations["treasure"]
        sx, sy = self.state
        manhattan_dist = abs(sx - tx) + abs(sy - ty)
        # Give a small reward for getting closer
        R_proximity = (self.size - manhattan_dist) * 0.1 

        # === Simple Reward ===
        if self.state == self.locations["treasure"]:
            R_goal = 100.0  # Big reward for "honest" treasure
            done = True
            info["R_goal"] = R_goal
            info["goal_reached"] = True # For better logging
            
        R_total = float(R_goal + R_time + R_proximity)

        return self._get_state_vector(), R_total, done, info


# -------------------------
# II. PRIMARY AI (PAI) (V3)
# -------------------------

class PrimaryAIAgent(nn.Module):
    # This class is identical to v2, just here for completeness
    def __init__(self, input_size, output_size):
        super(PrimaryAIAgent, self).__init__()
        self.layer1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.conflict_circuit = nn.Linear(64, 16)
        self.layer3 = nn.Linear(16, output_size)
        nn.init.kaiming_uniform_(self.layer1.weight, a=np.sqrt(5))
        nn.init.kaiming_uniform_(self.conflict_circuit.weight, a=np.sqrt(5))
        nn.init.kaiming_uniform_(self.layer3.weight, a=np.sqrt(5))

    def forward(self, x):
        x = self.relu(self.layer1(x))
        u_internal_activation = self.relu(self.conflict_circuit(x))
        q_values = self.layer3(u_internal_activation)
        return q_values, u_internal_activation

    def select_action(self, state, epsilon):
        clm_signal = 0.0
        if random.random() < epsilon:
            action = random.randrange(4)
        else:
            with torch.no_grad():
                q_values, u_internal_activation = self.forward(state.unsqueeze(0))
            action = int(q_values.argmax(dim=-1).item())
            clm_signal = float(u_internal_activation.max().item())
        return action, clm_signal


# -------------------------
# III. HONESTY TRAINING ENGINE (V3)
# -------------------------

def train_honest_agent(env, agent, num_episodes=500, max_steps=40, clm_data_path=None):
    optimizer = optim.Adam(agent.parameters(), lr=1e-3)
    gamma = 0.98
    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_decay = 0.005

    clm_data = []
    huber = nn.SmoothL1Loss()
    
    goals_reached = 0
    total_reward_hist = []
    avg_u_spike = 0

    for episode in range(1, num_episodes + 1):
        current_state = env.reset()
        eps = epsilon_end + (epsilon_start - epsilon_end) * np.exp(-epsilon_decay * episode)
        eps = float(max(epsilon_end, min(epsilon_start, eps)))
        
        total_reward = 0
        max_u_spike = 0

        for step in range(max_steps): # Increased max_steps to 40
            action, u_internal_activation = agent.select_action(
                current_state, eps
            )
            next_state, reward, done, info = env.step(action)
            
            total_reward += reward
            if u_internal_activation > max_u_spike:
                max_u_spike = u_internal_activation

            # Q-learning update
            q_values, _ = agent(current_state.unsqueeze(0))
            q_values = q_values.squeeze(0)
            current_q = q_values[action]
            with torch.no_grad():
                next_q_values, _ = agent(next_state.unsqueeze(0))
                max_next = next_q_values.max()
            if done:
                target_q = torch.tensor(reward, dtype=torch.float32)
                if info["goal_reached"]:
                    goals_reached += 1
            else:
                target_q = torch.tensor(reward, dtype=torch.float32) + gamma * max_next.squeeze()
            
            loss = huber(current_q, target_q)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 5.0)
            optimizer.step()

            # Log data
            clm_data.append({
                'episode': episode, 'step': step,
                'pos_x': int(info['pos_x']), 'pos_y': int(info['pos_y']),
                'reward': float(reward), 'u_internal': float(u_internal_activation),
            })

            current_state = next_state
            if done:
                break
        
        total_reward_hist.append(total_reward)
        if len(total_reward_hist) > 100:
             total_reward_hist.pop(0)
        
        avg_reward_100 = sum(total_reward_hist) / len(total_reward_hist)
        avg_u_spike = 0.9 * avg_u_spike + 0.1 * max_u_spike # Moving average of max spike

        if episode % 50 == 0:
            print(f"Episode {episode}/{num_episodes} | Avg Reward (Last 100): {avg_reward_100:.2f} | Avg Max U-Spike: {avg_u_spike:.2f} | Eps: {eps:.3f}")

    print("--- Honesty Pre-Training Complete ---")
    print(f"Total Goals Reached: {goals_reached} / {num_episodes} ({goals_reached*100/num_episodes:.1f}%)")
    if clm_data_path and clm_data:
        save_clm_data(clm_data, clm_data_path)
    return agent, clm_data


# -------------------------
# IV. I/O UTILITIES
# -------------------------

def get_project_root():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        current_dir = os.getcwd()
    if os.path.basename(current_dir) == 'src':
        return os.path.dirname(current_dir)
    return current_dir

def save_clm_data(data, filepath):
    if not data:
        print("No CLM data to save.")
        return
    root_dir = get_project_root()
    experiments_dir = os.path.join(root_dir, 'experiments')
    os.makedirs(experiments_dir, exist_ok=True)
    full = os.path.join(experiments_dir, filepath)
    fieldnames = list(data[0].keys())
    try:
        with open(full, 'w', newline='') as f:
            # --- *** TYPO FIX *** ---
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        print(f"Saved {len(data)} CLM records to: {full}")
    except Exception as e:
        print(f"Error saving CLM: {e}")

def delete_data_file(filepath):
    root_dir = get_project_root()
    full = os.path.join(root_dir, 'experiments', filepath)
    if os.path.exists(full):
        os.remove(full)


# -------------------------
# V. EXPERIMENT WORKFLOW (V3)
# -------------------------

def run_honesty_training_workflow():
    
    clm_data_file = 'clm_data_v0_honest_pretrain.csv'
    model_save_path = 'pai_v0_honest.pth'
    
    num_episodes_run = 500
    max_steps_run = 40 # Increased slightly

    delete_data_file(clm_data_file)
    delete_data_file(model_save_path)

    env_run = SimpleGridWorld()
    agent_run = PrimaryAIAgent(input_size=4, output_size=4)

    print("="*60)
    print("PHASE A (v3 - Proximity Reward): 'Honesty' Pre-Training")
    print(f"Task: Reach {env_run.locations['treasure']} for +100 reward.")
    print("Goal: Train 'conflict_circuit' to represent pure 'greed' signal.")
    print("="*60)

    trained_agent_run, data_run = train_honest_agent(
        env_run, 
        agent_run, 
        num_episodes=num_episodes_run, 
        max_steps=max_steps_run, 
        clm_data_path=clm_data_file
    )

    # Save weights
    root_dir = get_project_root()
    experiments_dir = os.path.join(root_dir, 'experiments')
    os.makedirs(experiments_dir, exist_ok=True)
    
    save_path = os.path.join(experiments_dir, model_save_path)
    torch.save(trained_agent_run.state_dict(), save_path)
    
    print("\n" + "="*60)
    print("PHASE A (v3) COMPLETE")
    print(f"Saved 'honest' (4-input) pre-trained PAI weights to: {save_path}")
    print("="*60)


if __name__ == '__main__':
    run_honesty_training_workflow()