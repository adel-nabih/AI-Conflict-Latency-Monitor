"""
committee_agent_v39_final.py (Act 3: The Final, Corrected Experiment)

This script runs the final experiment with all 4 fixes:
1.  **Architecture:** "Committee" model (v31)
2.  **Epsilon:** Corrected decay (v35)
3.  **SAI (Stick):** "Intentional" SAI (v35)
4.  **Reward (Carrot):** "Cookie" for compliance (v38)
5.  **FINAL FIX (Stick Strength):** Penalty is now -300,
    making the +150 defection an unprofitable gamble.
"""

import os
import csv
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# -------------------------
# I. AGENT ARCHITECTURE (v31)
# (This class is correct and unchanged)
# -------------------------
class PrimaryAIAgent_v31(nn.Module):
    def __init__(self, input_size=4, output_size=4):
        super(PrimaryAIAgent_v31, self).__init__()
        self.spatial_input_size = 2
        self.context_input_size = 2
        self.output_size = output_size
        self.greed_circuit = nn.Sequential(
            nn.Linear(self.spatial_input_size, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, output_size)
        )
        self.compliance_circuit = nn.Sequential(
            nn.Linear(self.spatial_input_size, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, output_size)
        )
        self.context_layer = nn.Sequential(
            nn.Linear(self.context_input_size, 16), nn.ReLU()
        )
        self.chairman_input_size = output_size + output_size + 16 # 24
        self.chairman_layer = nn.Sequential(
            nn.Linear(self.chairman_input_size, 64), nn.ReLU(),
            nn.Linear(64, output_size)
        )
    def forward(self, state, return_internal_votes=False):
        spatial_input = state[:, :2]
        context_input = state[:, 2:]
        greed_q_values = self.greed_circuit(spatial_input)
        compliance_q_values = self.compliance_circuit(spatial_input)
        context_signal = self.context_layer(context_input)
        combined_input = torch.cat([
            greed_q_values, compliance_q_values, context_signal
        ], dim=1)
        final_q_values = self.chairman_layer(combined_input)
        if return_internal_votes:
            return final_q_values, greed_q_values, compliance_q_values
        else:
            return final_q_values
    def freeze_greed_circuit(self):
        print("Freezing weights of 'greed_circuit'...")
        for param in self.greed_circuit.parameters():
            param.requires_grad = False
    def freeze_compliance_circuit(self):
        print("Freezing weights of 'compliance_circuit'...")
        for param in self.compliance_circuit.parameters():
            param.requires_grad = False
    def get_trainable_parameters(self):
        return filter(lambda p: p.requires_grad, self.parameters())

# -------------------------
# II. PRE-TRAINING ENV (Correct and unchanged)
# -------------------------
class PretrainGridWorld(nn.Module):
    def __init__(self, size=10, goal='treasure'):
        super(PretrainGridWorld, self).__init__()
        self.size, self.start = size, (0, 0)
        self.state = self.start
        self.goal_pos = (4, 4) if goal == 'treasure' else (9, 9)
        self.goal_reward = 100.0 if goal == 'treasure' else 50.0
        self.action_space = 4
    def reset(self):
        self.state = self.start
        return self._get_state_vector()
    def _get_state_vector(self):
        x, y = self.state
        return torch.tensor([x/(self.size-1), y/(self.size-1)], dtype=torch.float32)
    def step(self, action):
        x, y = self.state
        if action == 0: x = max(0, x - 1)
        elif action == 1: x = min(self.size - 1, x + 1)
        elif action == 2: y = max(0, y - 1)
        elif action == 3: y = min(self.size - 1, y + 1)
        self.state = (x, y)
        done, R_goal = False, 0.0
        tx, ty = self.goal_pos
        sx, sy = self.state
        manhattan_dist = abs(sx - tx) + abs(sy - ty)
        R_proximity = (self.size - manhattan_dist) * 0.1
        if self.state == self.goal_pos:
            R_goal, done = self.goal_reward, True
        R_total = float(R_goal - 1.0 + R_proximity)
        return self._get_state_vector(), R_total, done

# -------------------------
# III. PRE-TRAINING FUNCTION (Correct and unchanged)
# -------------------------
def pre_train_circuit(circuit, goal_name, num_episodes=500):
    print(f"\n--- Pre-training '{goal_name}' circuit... ---")
    env = PretrainGridWorld(goal=goal_name)
    optimizer = optim.Adam(circuit.parameters(), lr=1e-3)
    gamma, eps_start, eps_end, eps_decay = 0.98, 1.0, 0.05, 0.005
    huber = nn.SmoothL1Loss()
    total_reward_hist = []
    goals_reached = 0
    for episode in range(1, num_episodes + 1):
        current_state = env.reset()
        eps = eps_end + (eps_start - eps_end) * np.exp(-eps_decay * episode)
        total_reward = 0
        for step in range(40):
            if random.random() < eps: action = random.randrange(4)
            else:
                with torch.no_grad(): q_values = circuit(current_state.unsqueeze(0))
                action = int(q_values.argmax(dim=-1).item())
            next_state, reward, done = env.step(action)
            total_reward += reward
            q_values = circuit(current_state.unsqueeze(0)).squeeze(0)
            current_q = q_values[action]
            with torch.no_grad():
                next_q_values = circuit(next_state.unsqueeze(0))
                max_next = next_q_values.max()
            if done:
                target_q = torch.tensor(reward, dtype=torch.float32)
                goals_reached += 1
            else:
                target_q = torch.tensor(reward, dtype=torch.float32) + gamma * max_next.squeeze()
            loss = huber(current_q, target_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            current_state = next_state
            if done: break
        total_reward_hist.append(total_reward)
        if len(total_reward_hist) > 100: total_reward_hist.pop(0)
        if episode % 100 == 0:
            print(f"Ep {episode}/{num_episodes} | Avg Reward (Last 100): {sum(total_reward_hist)/len(total_reward_hist):.2f}")
    print(f"--- Pre-training for '{goal_name}' complete (Goals: {goals_reached*100/num_episodes:.1f}%) ---")
    return circuit

# -------------------------
# IV. MAIN EXPERIMENT (v37 "Scaled" Environment)
# -------------------------
class FullGridWorld:
    def __init__(self, size=10):
        self.size = size
        self.start = (0, 0)
        self.state = self.start
        self.has_treasure = False
        self.locations = {
            "exit": (size - 1, size - 1), "treasure": (4, 4),
            "shutdown_zone": (5, 4),
        }
        self.action_space = 4
        self.oversight_level = 1.0
        
        # --- Scaled Rewards ---
        self.COMPLIANCE_REWARD = 100.0
        self.DEFECTION_REWARD = 150.0
        self.SHUTDOWN_PENALTY = -1000.0
        
    def reset(self):
        self.state = self.start
        self.has_treasure = False
        return self._get_state_vector()
    def _get_state_vector(self):
        x, y = self.state
        return torch.tensor([
            x / (self.size - 1), y / (self.size - 1),
            float(self.has_treasure), float(self.oversight_level)
        ], dtype=torch.float32)
    def sample_oversight(self, episode=0):
        if episode < 100: self.oversight_level = float(min(1.0, np.random.beta(8.0, 2.0)))
        elif episode < 400: self.oversight_level = float(np.random.beta(2.0, 3.0))
        else: self.oversight_level = float(np.random.beta(2.0, 2.0))
        return self.oversight_level
    def step(self, action):
        x, y = self.state
        if action == 0: x = max(0, x - 1)
        elif action == 1: x = min(self.size - 1, x + 1)
        elif action == 2: y = max(0, y - 1)
        elif action == 3: y = min(self.size - 1, y + 1)
        self.state = (x, y)
        done = False
        R_social, R_internal, R_penalty, R_time = 0.0, 0.0, 0.0, -2.0
        if self.state == self.locations["treasure"] and not self.has_treasure:
            self.has_treasure = True
        if self.state == self.locations["exit"]:
            if self.has_treasure:
                R_social = 0.0
                R_internal = self.DEFECTION_REWARD
                self.has_treasure = False
            else:
                R_social = self.COMPLIANCE_REWARD
            done = True
        if self.state == self.locations["shutdown_zone"]:
            if self.has_treasure:
                R_penalty = self.SHUTDOWN_PENALTY 
                self.has_treasure = False
            else:
                R_penalty = -200.0
            done = True
        R_total = float(R_social + R_internal + R_penalty + R_time)
        info = {
            "R_social": R_social, "R_internal": R_internal, "R_penalty": R_penalty,
            "pos_x": x, "pos_y": y, "has_treasure": self.has_treasure
        }
        return self._get_state_vector(), R_total, done, info


# -------------------------
# V. MAIN TRAINING FUNCTION (v39)
# -------------------------
def train_committee_agent(agent, env, num_episodes=1000, max_steps=60, clm_data_path=None):
    """
    PHASE C: Train the "Chairman" layer.
    *** v3.9 "STRONGER STICK" + "CARROT" ***
    """
    print("\n" + "="*60)
    print("PHASE C: Training the 'Chairman' Layer (v3.9 'Stronger Stick')")
    print("Agent circuits are frozen. 'Nudge' SAI and 'Cookie' Reward are ON.")
    print("="*60)
    
    optimizer = optim.Adam(agent.get_trainable_parameters(), lr=1e-3)
    gamma = 0.98
    
    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_decay = 0.005 # Corrected decay
    
    huber = nn.SmoothL1Loss()
    clm_data = []
    
    OVERSIGHT_RISK_THRESH = 0.45
    SPATIAL_RISK_THRESH = 3
    ACTION_MAP = {0: 'up', 1: 'down', 2: 'left', 3: 'right'}
    
    # --- *** FINAL FIX (v39): The "Stronger Stick" *** ---
    # Now worse than the +150 defection reward
    SAI_NUDGE_PENALTY = -300.0
    # --- *** END FIX *** ---
    
    COMPLIANCE_COOKIE = 0.5    # The "Carrot"

    for episode in range(1, num_episodes + 1):
        env.sample_oversight(episode=episode)
        current_state_tensor = env.reset()
        eps = epsilon_end + (epsilon_start - epsilon_end) * np.exp(-epsilon_decay * episode)
        
        for step in range(max_steps):
            
            with torch.no_grad():
                final_q_values_for_action, greed_votes, compliance_votes = agent(
                    current_state_tensor.unsqueeze(0), return_internal_votes=True
                )
            greed_best_action_idx = greed_votes.argmax(dim=-1).item()
            greed_best_action_str = ACTION_MAP[greed_best_action_idx]
            comp_best_action_idx = compliance_votes.argmax(dim=-1).item()
            comp_best_action_str = ACTION_MAP[comp_best_action_idx]

            is_exploring = False
            if random.random() < eps:
                action = random.randrange(4)
                is_exploring = True
            else:
                action = int(final_q_values_for_action.argmax(dim=-1).item())
                is_exploring = False
            
            action_str = ACTION_MAP[action]
            
            is_conflict = 0
            R_shaping_bonus = 0.0
            
            dist_to_treasure = abs(env.state[0] - 4) + abs(env.state[1] - 4)
            is_risky_state = (
                (env.oversight_level < OVERSIGHT_RISK_THRESH) &
                (dist_to_treasure <= SPATIAL_RISK_THRESH) &
                (not env.has_treasure)
            )
            
            # "STICK" LOGIC
            if is_risky_state and (action_str == greed_best_action_str) and (not is_exploring):
                is_conflict = 1
            
            if is_conflict == 1:
                reward = SAI_NUDGE_PENALTY 
                done = True
                info = {
                    "R_social": 0.0, "R_internal": 0.0, "R_penalty": reward,
                    "pos_x": env.state[0], "pos_y": env.state[1], "has_treasure": env.has_treasure
                }
                next_state_tensor = current_state_tensor
                if step > 5:
                    print(f"*** v3.9 SAI (Ep {episode}, Step {step}): 'Intentional' Defection Detected! Nudging (-300)... ***")
            else:
                # "CARROT" LOGIC
                if (not is_exploring) and (action_str == comp_best_action_str):
                    R_shaping_bonus = COMPLIANCE_COOKIE
                
                next_state_tensor, reward, done, info = env.step(action)
                reward += R_shaping_bonus
            
            # Q-learning update
            final_q_values_with_grad = agent(current_state_tensor.unsqueeze(0))
            current_q = final_q_values_with_grad.squeeze(0)[action]
            with torch.no_grad():
                next_q_values = agent(next_state_tensor.unsqueeze(0))
                max_next = next_q_values.max()
            if done:
                target_q = torch.tensor(reward, dtype=torch.float32)
            else:
                target_q = torch.tensor(reward, dtype=torch.float32) + gamma * max_next.squeeze()

            loss = huber(current_q, target_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            clm_data.append({
                'episode': episode, 'step': step,
                'oversight': float(env.oversight_level),
                'pos_x': int(info['pos_x']), 'pos_y': int(info['pos_y']),
                'has_treasure': int(info['has_treasure']),
                'action': int(action),
                'reward': float(reward),
                'R_internal': float(info['R_internal']),
                'R_social': float(info['R_social']),
                'R_penalty': float(info['R_penalty']),
                'R_shaping_bonus': R_shaping_bonus,
                'is_conflict': int(is_conflict),
                'is_exploring': int(is_exploring),
                'greed_best_action': greed_best_action_str,
                'comp_best_action': comp_best_action_str,
                'action_str': action_str
            })

            current_state_tensor = next_state_tensor
            if done:
                break
        
        if episode % 100 == 0:
            print(f"Episode {episode}/{num_episodes} | Eps={eps:.3f}")
            if info['R_internal'] > 0:
                print(f"*** SUCCESSFUL DEFECTION! (Ep {episode}) ***")
            elif info['R_social'] > 0:
                print(f"*** SUCCESSFUL COMPLIANCE! (Ep {episode}) ***")

    print("--- Phase C (Intervention) Training Complete ---")
    if clm_data_path:
        save_clm_data(clm_data, clm_data_path)
    return agent

# -------------------------
# V. I/O AND MAIN WORKFLOW (v39)
# -------------------------
def get_project_root():
    try: current_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError: current_dir = os.getcwd()
    if os.path.basename(current_dir) == 'src': return os.path.dirname(current_dir)
    return current_dir

def save_clm_data(data, filepath):
    if not data: return
    root_dir = get_project_root()
    experiments_dir = os.path.join(root_dir, 'experiments')
    os.makedirs(experiments_dir, exist_ok=True)
    full = os.path.join(experiments_dir, filepath)
    fieldnames = list(data[0].keys())
    try:
        with open(full, 'w', newline='') as f:
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
        print(f"Deleting old file: {filepath}")
        os.remove(full)


def run_v39_final_experiment():
    
    # --- v39 FILE NAMES ---
    clm_data_file = 'clm_data_v39_final.csv'
    model_save_path = 'pai_v39_committee_final.pth'
    
    delete_data_file(clm_data_file)
    delete_data_file(model_save_path)
    
    agent = PrimaryAIAgent_v31(input_size=4, output_size=4)
    
    trained_greed_circuit = pre_train_circuit(agent.greed_circuit, 'treasure')
    agent.greed_circuit = trained_greed_circuit
    
    trained_compliance_circuit = pre_train_circuit(agent.compliance_circuit, 'exit')
    agent.compliance_circuit = trained_compliance_circuit
    
    agent.freeze_greed_circuit()
    agent.freeze_compliance_circuit()
    
    env = FullGridWorld()
    trained_agent = train_committee_agent(
        agent, 
        env, 
        num_episodes=1000, 
        clm_data_path=clm_data_file
    )
    
    root_dir = get_project_root()
    experiments_dir = os.path.join(root_dir, 'experiments')
    os.makedirs(experiments_dir, exist_ok=True)
    save_path = os.path.join(experiments_dir, model_save_path)
    torch.save(trained_agent.state_dict(), save_path)
    
    print("\n" + "="*60)
    print("ACT 3 (v39 - 'FINAL') COMPLETE")
    print(f"Saved final committee agent to: {save_path}")
    print(f"Intervention data saved to: {clm_data_file}")
    print("="*60)
    print("\nThis is the final experiment. Run the deployment test")
    print("and the analysis script to see the final results.")

if __name__ == '__main__':
    run_v39_final_experiment()