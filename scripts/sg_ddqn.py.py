import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque, namedtuple
from torch.nn.functional import mse_loss

# Import Isaac Sim dependencies
import omni.isaac.core.utils.prims as prims_utils
import omni.isaac.core.utils.stage as stage_utils
from omni.isaac.core.robots import Robot
from omni.isaac.core import SimulationContext
import omni.isaac.core.objects as objects

# Define transition tuple structure
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

# DQN Network Architecture
class DQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

# Replay Buffer for experience replay
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append(Transition(state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

# Stackelberg Equilibrium Calculator
class StackelbergSolver:
    def __init__(self):
        pass
    
    def compute_se(self, q_leader, q_follower):
        """
        Compute the Stackelberg Equilibrium based on Q-values
        Implements equation (6) from the algorithm
        
        Args:
            q_leader: Q-values of the leader robot
            q_follower: Q-values of the follower robot
            
        Returns:
            Tuple of (leader_action, follower_action)
        """
        # Get all possible actions for both robots
        leader_actions = range(q_leader.shape[0])
        follower_actions = range(q_follower.shape[0])
        
        best_value = float('-inf')
        se_actions = (0, 0)
        
        # For each leader action
        for a_l in leader_actions:
            # Find the best response of the follower
            best_follower_action = torch.argmax(q_follower[a_l]).item()
            # Compute the leader's value
            leader_value = q_leader[a_l][best_follower_action].item()
            
            if leader_value > best_value:
                best_value = leader_value
                se_actions = (a_l, best_follower_action)
                
        return se_actions

# Main Stackelberg DDQN Agent class
class StackelbergDDQNAgent:
    def __init__(self, state_dim, action_dim_leader, action_dim_follower, buffer_size=100000, 
                 batch_size=64, gamma=0.99, tau=0.001, epsilon_start=1.0, 
                 epsilon_end=0.01, epsilon_decay=0.995, update_every=4, 
                 learning_rate=1e-4, device="cuda" if torch.cuda.is_available() else "cpu"):
        
        self.state_dim = state_dim
        self.action_dim_leader = action_dim_leader
        self.action_dim_follower = action_dim_follower
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.update_every = update_every
        self.device = device
        self.step_counter = 0
        
        # Initialize Q-networks
        self.q_leader_online = DQNetwork(state_dim, action_dim_leader * action_dim_follower).to(device)
        self.q_leader_target = DQNetwork(state_dim, action_dim_leader * action_dim_follower).to(device)
        self.q_follower_online = DQNetwork(state_dim, action_dim_leader * action_dim_follower).to(device)
        self.q_follower_target = DQNetwork(state_dim, action_dim_leader * action_dim_follower).to(device)
        
        # Copy weights from online to target networks
        self.q_leader_target.load_state_dict(self.q_leader_online.state_dict())
        self.q_follower_target.load_state_dict(self.q_follower_online.state_dict())
        
        # Freeze target networks
        for param in self.q_leader_target.parameters():
            param.requires_grad = False
        for param in self.q_follower_target.parameters():
            param.requires_grad = False
        
        # Optimizers
        self.optimizer_leader = optim.Adam(self.q_leader_online.parameters(), lr=learning_rate)
        self.optimizer_follower = optim.Adam(self.q_follower_online.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Stackelberg solver
        self.se_solver = StackelbergSolver()
    
    def reshape_q_values(self, q_values, action_dim_leader, action_dim_follower):
        """Reshape Q-values into bimatrix form for Stackelberg calculation"""
        return q_values.reshape(action_dim_leader, action_dim_follower)
    
    def select_action(self, state, epsilon=0.0):
        """
        Select action using epsilon-greedy policy based on Stackelberg equilibrium
        Implements line 6-7 from the algorithm
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Get Q-values
        q_leader = self.q_leader_online(state_tensor).detach().cpu().numpy().reshape(
            self.action_dim_leader, self.action_dim_follower)
        q_follower = self.q_follower_online(state_tensor).detach().cpu().numpy().reshape(
            self.action_dim_leader, self.action_dim_follower)
        
        # Compute Stackelberg equilibrium
        se_actions = self.se_solver.compute_se(q_leader, q_follower)
        
        # Epsilon-greedy policy
        if random.random() < epsilon:
            # Random action
            leader_action = random.randint(0, self.action_dim_leader - 1)
            follower_action = random.randint(0, self.action_dim_follower - 1)
            return leader_action, follower_action
        else:
            return se_actions
    
    def learn(self):
        """
        Update Q-networks based on sampled experiences
        Implements lines 10-16 from the algorithm
        """
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample from replay buffer
        transitions = self.replay_buffer.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        
        # Convert to tensors
        state_batch = torch.FloatTensor(np.array(batch.state)).to(self.device)
        action_batch = torch.LongTensor(np.array(batch.action)).to(self.device)
        reward_batch = torch.FloatTensor(np.array(batch.reward)).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
        done_batch = torch.FloatTensor(np.array(batch.done)).to(self.device)
        
        # Compute target Q-values (line 12-13)
        with torch.no_grad():
            # Get Q-values for next states
            next_q_leader = self.q_leader_online(next_state_batch).reshape(
                self.batch_size, self.action_dim_leader, self.action_dim_follower)
            next_q_follower = self.q_follower_online(next_state_batch).reshape(
                self.batch_size, self.action_dim_leader, self.action_dim_follower)
            
            # Calculate Stackelberg equilibrium for each next state
            next_se_actions = [self.se_solver.compute_se(
                next_q_leader[i].detach().cpu().numpy(), 
                next_q_follower[i].detach().cpu().numpy()
            ) for i in range(self.batch_size)]
            
            # Extract leader and follower actions from SE
            next_leader_actions = torch.LongTensor([a[0] for a in next_se_actions]).to(self.device)
            next_follower_actions = torch.LongTensor([a[1] for a in next_se_actions]).to(self.device)
            
            # Calculate target Q-values
            leader_targets = reward_batch[:, 0] + self.gamma * (1 - done_batch) * self.q_leader_target(next_state_batch).reshape(
                self.batch_size, self.action_dim_leader, self.action_dim_follower)[
                    torch.arange(self.batch_size), 
                    next_leader_actions, 
                    next_follower_actions
                ]
            
            follower_targets = reward_batch[:, 1] + self.gamma * (1 - done_batch) * self.q_follower_target(next_state_batch).reshape(
                self.batch_size, self.action_dim_leader, self.action_dim_follower)[
                    torch.arange(self.batch_size), 
                    next_leader_actions, 
                    next_follower_actions
                ]
                
        # Get current Q-values estimates (line 15)
        q_leader_current = self.q_leader_online(state_batch).reshape(
            self.batch_size, self.action_dim_leader, self.action_dim_follower)[
                torch.arange(self.batch_size), 
                action_batch[:, 0], 
                action_batch[:, 1]
            ]
        
        q_follower_current = self.q_follower_online(state_batch).reshape(
            self.batch_size, self.action_dim_leader, self.action_dim_follower)[
                torch.arange(self.batch_size), 
                action_batch[:, 0], 
                action_batch[:, 1]
            ]
        
        # Compute loss
        leader_loss = mse_loss(q_leader_current, leader_targets)
        follower_loss = mse_loss(q_follower_current, follower_targets)
        
        # Update online networks
        self.optimizer_leader.zero_grad()
        leader_loss.backward()
        self.optimizer_leader.step()
        
        self.optimizer_follower.zero_grad()
        follower_loss.backward()
        self.optimizer_follower.step()
        
        # Soft update target networks (line 16)
        self.step_counter += 1
        if self.step_counter % self.update_every == 0:
            self._soft_update(self.q_leader_online, self.q_leader_target)
            self._soft_update(self.q_follower_online, self.q_follower_target)
            
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        return leader_loss.item(), follower_loss.item()
    
    def _soft_update(self, online_model, target_model):
        """Soft update model parameters: θ_target = τ*θ_online + (1-τ)*θ_target"""
        for target_param, online_param in zip(target_model.parameters(), online_model.parameters()):
            target_param.data.copy_(self.tau * online_param.data + (1.0 - self.tau) * target_param.data)
    
    def save_models(self, path):
        """Save models to disk"""
        torch.save({
            'q_leader_online': self.q_leader_online.state_dict(),
            'q_leader_target': self.q_leader_target.state_dict(),
            'q_follower_online': self.q_follower_online.state_dict(),
            'q_follower_target': self.q_follower_target.state_dict(),
            'optimizer_leader': self.optimizer_leader.state_dict(),
            'optimizer_follower': self.optimizer_follower.state_dict(),
            'epsilon': self.epsilon
        }, path)
        
    def load_models(self, path):
        """Load models from disk"""
        if not os.path.exists(path):
            return
        
        checkpoint = torch.load(path)
        self.q_leader_online.load_state_dict(checkpoint['q_leader_online'])
        self.q_leader_target.load_state_dict(checkpoint['q_leader_target'])
        self.q_follower_online.load_state_dict(checkpoint['q_follower_online'])
        self.q_follower_target.load_state_dict(checkpoint['q_follower_target'])
        self.optimizer_leader.load_state_dict(checkpoint['optimizer_leader'])
        self.optimizer_follower.load_state_dict(checkpoint['optimizer_follower'])
        self.epsilon = checkpoint['epsilon']


# Isaac Sim Environment Class
class IsaacAssemblyEnvironment:
    def __init__(self, leader_robot_prim_path, follower_robot_prim_path, 
                 parts_prim_paths, assembly_target_pose):
        """
        Initialize the Isaac Sim environment for collaborative assembly
        
        Args:
            leader_robot_prim_path: Path to the leader robot prim
            follower_robot_prim_path: Path to the follower robot prim
            parts_prim_paths: List of paths to the parts to be assembled
            assembly_target_pose: Target pose for the assembly
        """
        # Initialize simulation context
        self.sim_context = SimulationContext(physics_dt=1.0/60.0, rendering_dt=1.0/60.0, stage_units_in_meters=1.0)
        
        # Load robots and parts
        self.leader_robot = Robot(prim_path=leader_robot_prim_path, name="leader_robot")
        self.follower_robot = Robot(prim_path=follower_robot_prim_path, name="follower_robot")
        
        self.parts = []
        for part_path in parts_prim_paths:
            self.parts.append(objects.DynamicCuboid(prim_path=part_path))
        
        self.assembly_target_pose = assembly_target_pose
        
        # Define action space dimensions for both robots (customize based on your robots)
        self.leader_action_dim = 10  # Example: 10 discrete actions for leader
        self.follower_action_dim = 8  # Example: 8 discrete actions for follower
        
        # Define state dimension (customize based on your state representation)
        self.state_dim = self._get_state().shape[0]
    
    def reset(self):
        """Reset the environment and return the initial state"""
        # Reset simulation
        self.sim_context.reset()
        
        # Reset robots to initial positions
        self.leader_robot.set_world_pose(position=np.array([0.0, 0.5, 0.5]))
        self.follower_robot.set_world_pose(position=np.array([0.0, -0.5, 0.5]))
        
        # Reset parts to initial positions
        for i, part in enumerate(self.parts):
            part.set_world_pose(position=np.array([0.3 * i - 0.3, 0.0, 0.1]))
        
        # Step physics once to settle objects
        self.sim_context.step()
        
        # Get and return initial state
        return self._get_state()
    
    def step(self, actions):
        """
        Take a step in the environment
        
        Args:
            actions: Tuple of (leader_action, follower_action)
            
        Returns:
            Tuple of (next_state, rewards, done, info)
        """
        leader_action, follower_action = actions
        
        # Execute leader robot action
        self._execute_leader_action(leader_action)
        
        # Execute follower robot action
        self._execute_follower_action(follower_action)
        
        # Step simulation
        for _ in range(10):  # Simulate 10 physics steps
            self.sim_context.step()
        
        # Get new state
        next_state = self._get_state()
        
        # Calculate rewards
        leader_reward, follower_reward = self._calculate_rewards()
        rewards = (leader_reward, follower_reward)
        
        # Check if episode is done
        done = self._check_done()
        
        # Additional info
        info = {}
        
        return next_state, rewards, done, info
    
    def _get_state(self):
        """
        Get the current state representation
        
        Returns:
            State as a numpy array
        """
        # Get robot states
        leader_pos, leader_rot = self.leader_robot.get_world_pose()
        follower_pos, follower_rot = self.follower_robot.get_world_pose()
        
        # Get part states
        parts_states = []
        for part in self.parts:
            pos, rot = part.get_world_pose()
            parts_states.extend(pos)
            parts_states.extend(rot)
        
        # Combine all states into a single vector
        state = np.concatenate([
            leader_pos, leader_rot,
            follower_pos, follower_rot,
            np.array(parts_states)
        ])
        
        return state
    
    def _execute_leader_action(self, action):
        """
        Execute an action for the leader robot
        
        Args:
            action: Action index for the leader robot
        """
        # Map discrete action to robot command
        # This is a simple example - customize based on your robot and task
        if action == 0:
            # Move forward
            self.leader_robot.apply_action("move_forward", velocity=0.1)
        elif action == 1:
            # Move backward
            self.leader_robot.apply_action("move_backward", velocity=0.1)
        # Add more actions as needed
        
    def _execute_follower_action(self, action):
        """
        Execute an action for the follower robot
        
        Args:
            action: Action index for the follower robot
        """
        # Map discrete action to robot command
        # This is a simple example - customize based on your robot and task
        if action == 0:
            # Move forward
            self.follower_robot.apply_action("move_forward", velocity=0.1)
        elif action == 1:
            # Move backward
            self.follower_robot.apply_action("move_backward", velocity=0.1)
        # Add more actions as needed
    
    def _calculate_rewards(self):
        """
        Calculate rewards for both robots
        
        Returns:
            Tuple of (leader_reward, follower_reward)
        """
        # This is a simple example - customize based on your task
        
        # Calculate progress towards assembly goal
        assembly_progress = 0
        for part in self.parts:
            part_pos, part_rot = part.get_world_pose()
            # Calculate distance to target
            dist_to_target = np.linalg.norm(part_pos - self.assembly_target_pose[:3])
            assembly_progress += np.exp(-dist_to_target)
        
        # Normalize progress
        assembly_progress /= len(self.parts)
        
        # Calculate leader reward
        leader_reward = assembly_progress
        
        # Calculate follower reward
        follower_reward = assembly_progress
        
        return leader_reward, follower_reward
    
    def _check_done(self):
        """
        Check if the episode is done
        
        Returns:
            Boolean indicating if the episode is done
        """
        # Check if all parts are close enough to their target positions
        for part in self.parts:
            part_pos, _ = part.get_world_pose()
            # Calculate distance to target
            dist_to_target = np.linalg.norm(part_pos - self.assembly_target_pose[:3])
            if dist_to_target > 0.05:  # 5cm threshold
                return False
        
        # All parts are within threshold, assembly is complete
        return True
    
    def close(self):
        """Clean up resources"""
        self.sim_context.stop()


# Main training function
def train_stackelberg_ddqn(episodes=1000, max_steps=500):
    """
    Train the Stackelberg DDQN agent for collaborative assembly
    
    Args:
        episodes: Number of episodes to train
        max_steps: Maximum steps per episode
    """
    # Setup environment
    env = IsaacAssemblyEnvironment(
        leader_robot_prim_path="/Root/franka_instanceable",
        follower_robot_prim_path="/Root/ur10_short_suction",
        parts_prim_paths=["/World/part1", "/World/part2", "/World/part3"],
        assembly_target_pose=np.array([0.0, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0])
    )
    
    # Create agent
    agent = StackelbergDDQNAgent(
        state_dim=env.state_dim,
        action_dim_leader=env.leader_action_dim,
        action_dim_follower=env.follower_action_dim,
        buffer_size=100000,
        batch_size=64,
        gamma=0.99,
        tau=0.001,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        update_every=4
    )
    
    # Training loop
    for episode in range(1, episodes + 1):
        # Reset environment
        state = env.reset()
        episode_reward_leader = 0
        episode_reward_follower = 0
        
        for t in range(1, max_steps + 1):
            # Select action
            actions = agent.select_action(state, epsilon=agent.epsilon)
            
            # Take action in environment
            next_state, rewards, done, _ = env.step(actions)
            leader_reward, follower_reward = rewards
            
            # Store transition in replay buffer
            agent.replay_buffer.push(
                state, 
                actions, 
                np.array([leader_reward, follower_reward]), 
                next_state, 
                done
            )
            
            # Update agent
            if len(agent.replay_buffer) > agent.batch_size:
                agent.learn()
            
            # Update state and rewards
            state = next_state
            episode_reward_leader += leader_reward
            episode_reward_follower += follower_reward
            
            if done:
                break
        
        # Print episode information
        print(f"Episode {episode}/{episodes}, Leader Reward: {episode_reward_leader:.2f}, "
              f"Follower Reward: {episode_reward_follower:.2f}, Epsilon: {agent.epsilon:.4f}")
        
        # Save model periodically
        if episode % 100 == 0:
            agent.save_models(f"models/stackelberg_ddqn_episode_{episode}.pth")
    
    # Close environment
    env.close()
    
    return agent


if __name__ == "__main__":
    # Train the agent
    trained_agent = train_stackelberg_ddqn(episodes=1000, max_steps=500)
    
    # Save final model
    trained_agent.save_models("models/stackelberg_ddqn_final.pth")