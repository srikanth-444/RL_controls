import asyncio
import grpc
import generated.rollout_pb2 as rollout_pb2
import generated.rollout_pb2_grpc as rollout_pb2_grpc
import torch
from tinyphysics import TinyPhysicsModel
from policy.ppopolicy import PPOPolicy
from typing import List
from envirnoment.PPOENV import PPOEnv
from tqdm import tqdm
import random
from tinyphysics import TinyPhysicsModel, CONTROL_START_IDX, STEER_RANGE, CONTEXT_LENGTH,DEL_T
import io
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from time import time
import subprocess
import boto3

CLUSTER_NAME = "sunny-panda-ya1vzc"
SERVICE_NAME = "RL-service-uxjht2q6"
ecs = boto3.client('ecs')

class PPOTrainer:
    def __init__(self, model: TinyPhysicsModel,policy:PPOPolicy, data_path: str, gamma=0.99, lam=0.95, clip_eps=0.2, epochs=10, batch_size=256, lr=3e-4,debug: bool = False) -> None:
        self.model = model
        self.device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.policy=policy.to(self.device)
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.env_list = self.create_env(data_path)  # Assuming model has a data_path attribute
        self.policy_logs = {
                "mean": [],
                "std": [],
                "value": [],
                "entropy": [],
                "log_prob": [],
                "reward": [],
                "policy_loss": [],
                "value_loss": [],
                "total_loss": []
            }

    def create_env(self,data_path: str) -> List[str]:
        data_path = Path(data_path)
        if data_path.is_file():
            return [str(data_path)]
        if data_path.is_dir():
            files = sorted(data_path.glob('*.csv'))[:5000]
        return [str(f) for f in files]


    def update_policy(self,obs, actions, old_log_probs, returns, advantages):
        obs = torch.cat(obs, dim=0).to(device=self.device)
        actions = torch.cat(actions, dim=0).to(device=self.device)
        old_log_probs = torch.cat(old_log_probs, dim=0).to(device=self.device)
        returns = torch.cat(returns, dim=0).to(device=self.device)
        advantages = torch.cat(advantages, dim=0).to(device=self.device)
        
        for epoch in range(self.epochs):
                # Optional: sample mini-batches here for large datasets
                mean, std, values = self.policy(obs)

                if std.item()<0.3:
                    subprocess.Popen([r"D:\Users\popur\PPO_trainner\aws_shutdown.cmd"], shell=True)  # Shutdown the machine if std is too low

                    return True
                
                dist = torch.distributions.Normal(mean, std)
                
                
                log_probs = dist.log_prob(actions)
                entropy = dist.entropy().mean()
                with torch.no_grad():
                    self.policy_logs["mean"].append(mean.mean().item())
                    self.policy_logs["std"].append(std.mean().item())
                    self.policy_logs["value"].append(values.mean().item())
                    self.policy_logs["entropy"].append(entropy.item())
                    self.policy_logs["log_prob"].append(log_probs.mean().item())

                ratio = (log_probs - old_log_probs).exp()
                surr1 = ratio * advantages.unsqueeze(-1)
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages.unsqueeze(-1)
                policy_loss = -torch.min(surr1, surr2).mean()
                self.policy_logs["policy_loss"].append(policy_loss.item())

                value_loss = ((returns - values) ** 2).mean()
                self.policy_logs["value_loss"].append(value_loss.item())

                total_loss = 100*policy_loss + 0.5 * value_loss - 0.02 * entropy

                self.policy_logs["total_loss"].append(total_loss.item())

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
        return False  # Return False to indicate training was successful


    def sample_env_batch(self, batch_size=4):
        return random.sample(self.env_list, batch_size) 
    def evaluate_policy(self,env_path: str, render: bool = True):
        env = PPOEnv(self.model, Path(env_path), self.policy, debug=False)
        env.reset()
        env.rollout_buffer = []
        done = False
        rewards = []

        while not done:
            state, target, _ = env.get_state_target_futureplan(env.step_idx)

            last_action = env.action_history[-1]
            last_lataccel = env.current_lataccel_history[-1]

            a_ego=[s.a_ego for s in env.state_history[-CONTEXT_LENGTH:]]
            v_ego=[s.v_ego for s in env.state_history[-CONTEXT_LENGTH:]]
            roll_lataccel=[s.roll_lataccel for s in env.state_history[-CONTEXT_LENGTH:]]
            input=np.column_stack((env.action_history[-CONTEXT_LENGTH:],
                                roll_lataccel[-CONTEXT_LENGTH:],
                                v_ego[-CONTEXT_LENGTH:],
                                a_ego[-CONTEXT_LENGTH:],
                                env.current_lataccel_history[-CONTEXT_LENGTH:],
                                env.target_lataccel_history[-CONTEXT_LENGTH:]))
            input_tensor = torch.tensor(input, dtype=torch.float32).flatten().unsqueeze(0)
            input_tensor = input_tensor.to(self.device)  # Move to device

            # Get action from policy
            with torch.no_grad():
                mean, std, _ = env.policy(input_tensor)
                dist = torch.distributions.Normal(mean, std)
                action = dist.mean.item()  # or sample with dist.sample().item()

            env.action_history.append(action)
            env.sim_step(env.step_idx)
            env.state_history.append(state)
            env.target_lataccel_history.append(target)
            env.step_idx += 1

            if len(env.current_lataccel_history) >= 2:
                jerk = (env.current_lataccel_history[-1] - env.current_lataccel_history[-2]) / DEL_T
            else:
                jerk = 0.0

            reward = -((env.current_lataccel - target)**2 * 5000 + jerk**2 * 100)
            rewards.append(reward)

            done = env.step_idx >= len(env.data)

        

        # ✅ Plot current vs target lateral acceleration
        if render:
            # ✅ Print total reward
            print(f"Evaluation Total Reward: {np.mean(rewards):.2f}")
            import matplotlib.pyplot as plt
            steps = list(range(CONTROL_START_IDX, len(env.current_lataccel_history)))
            plt.figure(figsize=(20, 5))
            plt.plot(steps, env.current_lataccel_history[CONTROL_START_IDX:], label="Current LatAccel")
            plt.plot(steps, env.target_lataccel_history[CONTROL_START_IDX:], label="Target LatAccel", linestyle="--")
            plt.xlabel("Step")
            plt.ylabel("Lateral Acceleration")
            plt.title("Policy Behavior Evaluation")
            plt.legend()
            plt.grid()
            plt.show()  

        return env.current_lataccel_history[CONTROL_START_IDX:], env.target_lataccel_history[CONTROL_START_IDX:],np.mean(rewards)  # Return current lataccel, target lataccel, and average reward
    def plot_training_dynamics(self):

        keys = list(self.policy_logs.keys())
        plt.figure(figsize=(20, 12))

        for i, key in enumerate(keys):
            plt.subplot(3, 3, i + 1)
            plt.plot(self.policy_logs[key])
            plt.title(key.capitalize())
            plt.grid()

        plt.tight_layout()
        plt.show()
    

    async def train(self, num_rollouts=1000):
        
        pbar = tqdm(range(num_rollouts), desc='Training PPO')
        all_accel = []
        target = []
        for rollout_idx in pbar:
            start_time = time()
            all_obs = []
            all_actions = []
            all_old_log_probs = []
            all_returns = []
            all_advantages = []
            all_rollout_rewards = []
            buffer = io.BytesIO()
            torch.save(self.policy.state_dict(), buffer)
            weights_bytes = buffer.getvalue()

            # Get a list of environments (paths)
            env_paths = self.sample_env_batch(batch_size=self.batch_size)
            # print(env_paths)

            # Launch parallel async requests
            tasks = [
                        retry_request(lambda: run_single_request(self.stub, str(env_path), weights_bytes,self.gamma, self.lam))
                        for env_path in env_paths
                    ]
            results = await asyncio.gather(*tasks,return_exceptions=True)
            time_taken = time() - start_time
            count=0
            for result in results:
                if result is None or isinstance(result, Exception):
                    
                    count += 1
                    continue
                else:
                    obs_tensor, actions_tensor, old_log_probs_tensor, returns_tensor, advantages_tensor, reward = result

                    all_obs.append(obs_tensor)
                    all_actions.append(actions_tensor)
                    all_old_log_probs.append(old_log_probs_tensor)
                    all_returns.append(returns_tensor)
                    all_advantages.append(advantages_tensor)
                    all_rollout_rewards.append(reward)
            
            avg_reward = sum(all_rollout_rewards) / len(all_rollout_rewards)
            self.policy_logs["reward"].append(avg_reward)

            # Policy update
            if(self.update_policy(all_obs, all_actions, all_old_log_probs, all_returns, all_advantages)):
                print("Training stopped due to low std deviation in policy output.")
                break
            torch.save(self.policy.state_dict(), "model_weights.pth")
            if(rollout_idx % 10 == 0):
                accel,targ,_=self.evaluate_policy(self.env_list[0], render=False)  # Evaluate on the first environment
                all_accel.append(accel)
                target= targ
            pbar.set_postfix({'networkcall_time': time_taken,'num_failed_rollouts': count, "value_loss": self.policy_logs["value_loss"][-1],})
        plt.figure(figsize=(20, 5))
        base_alpha = 0.1
        max_alpha = 1.0
        for i in range(len(all_accel)):
            N = len(all_accel) - 1
            alpha = base_alpha * ((max_alpha / base_alpha) ** (i / N))
            plt.plot(all_accel[i], color='tab:blue', alpha=alpha)
                 
        plt.plot(target, label="Target LatAccel",color='tab:orange',linestyle="--")
        plt.xlabel("Step")
        plt.ylabel("Lateral Acceleration")
        plt.title("Policy Behavior Evaluation")
        plt.legend()
        plt.grid()
        plt.show()             

async def run_single_request(stub, path, weights,gama,lam):
    request = rollout_pb2.RolloutRequest(data_path=path,weights=weights, gama=gama, lam=lam)  
    response = await stub.RunRollout(request)

    obs_tensor = torch.tensor(response.obs.data).view(-1, 20*6)  # reshape here
    actions_tensor = torch.tensor(response.actions.data,dtype=torch.float32).unsqueeze(-1)  # ensure actions are 2D
    old_log_probs_tensor = torch.tensor(response.old_log_probs.data, dtype=torch.float32).unsqueeze(-1)  # ensure log_probs are 2D
    returns_tensor = torch.tensor(response.returns.data).unsqueeze(-1)  # ensure returns are 2D
    advantages_tensor = torch.tensor(response.advantages.data)
    rewards= torch.tensor(response.rewards.data)  # ensure rewards are 2D

    total_reward = rewards.mean().item()

    return (
        obs_tensor,
        actions_tensor,
        old_log_probs_tensor,
        returns_tensor,
        advantages_tensor,
        total_reward
    )

async def retry_request(fn, retries=3, delay=1):
    for i in range(retries):
        try:
            return await fn()
        except Exception as e:
            if i < retries - 1:
                await asyncio.sleep(delay)
            else:
                return e
def any_task_running():
    tasks = ecs.list_tasks(cluster=CLUSTER_NAME, serviceName=SERVICE_NAME)
    if not tasks["taskArns"]:
        return False

    details = ecs.describe_tasks(cluster=CLUSTER_NAME, tasks=tasks["taskArns"])
    return any(task["lastStatus"] == "RUNNING" for task in details["tasks"])

async def main():
    print("Waiting for a task to start running...")
    while not any_task_running():
        time.sleep(5)

    print("✅ Task running — doing my work now...")
    async with grpc.aio.secure_channel('envrollout.click:50051',grpc.ssl_channel_credentials()) as channel:
        stub = rollout_pb2_grpc.RolloutServiceStub(channel)

        model = TinyPhysicsModel("./models/tinyphysics.onnx", debug=False)
        policy = PPOPolicy(input_dim=(20 * 6))
        data_dir = Path("./data")

        trainer = PPOTrainer(model=model, policy=policy, data_path=data_dir, debug=False)
        trainer.stub = stub  # 🔁 attach the stub to the trainer


        await trainer.train()
        total_rewards = []
        for env_path in tqdm(trainer.env_list[:100]):
            _,_,rewards=trainer.evaluate_policy(env_path, render=False)
            total_rewards.append(rewards)
        print(f"Average Reward over 100 environments: {np.mean(total_rewards):.2f}")
        trainer.evaluate_policy(r'.\data\00000.csv', render=True)
        trainer.plot_training_dynamics()

if __name__ == '__main__':
    asyncio.run(main())

