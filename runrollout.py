import sys
sys.path.append('/app/generated')
import generated.rollout_pb2 as rollout_pb2
import generated.rollout_pb2_grpc as rollout_pb2_grpc

import grpc
from concurrent import futures
import torch
from envirnoment.PPOENV import PPOEnv
from policy.ppopolicy import PPOPolicy
from tinyphysics import TinyPhysicsModel
from pathlib import Path
import io


def compute_returns_and_advantages( buffer):
        rewards = [entry['reward'] for entry in buffer]
        values = [entry['value'] for entry in buffer]
        dones = [entry['done'] for entry in buffer]

        returns = []
        advantages = []
        gae = 0
        next_value = 0

        for i in reversed(range(len(rewards))):
            gamma=0.91
            lam=0.95
            delta = rewards[i] + gamma * next_value * (1 - dones[i]) - values[i]
            gae = delta + gamma * lam * (1 - dones[i]) * gae
            advantages.insert(0, gae)
            next_value = values[i]
            returns.insert(0, gae + values[i])

        return returns, advantages
def run_rollout( env: PPOEnv) -> None:
            env.reset()
            env.rollout_buffer = []  # Reset the rollout buffer
            done = False
            while not done:
                _, _, _, done = env.step()
            buffer = env.rollout_buffer
            returns, advantages =compute_returns_and_advantages(buffer)
            # Normalize advantages
            advantages = torch.tensor(advantages, dtype=torch.float32)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Convert buffer to tensors
            obs = torch.cat([entry["obs"] for entry in buffer], dim=0)
            actions = torch.tensor([entry["action"] for entry in buffer], dtype=torch.float32).unsqueeze(-1)
            old_log_probs = torch.tensor([entry["log_prob"] for entry in buffer], dtype=torch.float32).unsqueeze(-1)
            returns = torch.tensor(returns, dtype=torch.float32).unsqueeze(-1)
            
            return obs, actions, old_log_probs, returns, advantages

class RolloutServicer(rollout_pb2_grpc.RolloutServiceServicer):
    def __init__(self):
        self.model = TinyPhysicsModel("./models/tinyphysics.onnx", debug=False)
        self.policy = PPOPolicy(input_dim=(20 * 6))

    def RunRollout(self, request, context):
        buffer = io.BytesIO(request.weights)
        state_dict = torch.load(buffer, map_location="cpu")
        self.policy.load_state_dict(state_dict)
        path_str = request.data_path.replace("\\", "/")  # Normalize path for Windows compatibility
        env = PPOEnv(self.model, Path(path_str), self.policy, debug=False)
        obs, actions, old_log_probs, returns, advantages = run_rollout(env)

        def flatten(tensor):
            return rollout_pb2.Tensor1D(data=tensor.flatten().tolist())

        return rollout_pb2.RolloutResponse(
            obs=flatten(obs),
            actions=flatten(actions),
            old_log_probs=flatten(old_log_probs),
            returns=flatten(returns),
            advantages=flatten(advantages),
        )

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    rollout_pb2_grpc.add_RolloutServiceServicer_to_server(RolloutServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("gRPC server running at port 50051...")
    server.wait_for_termination()


if __name__ == '__main__':
    print("Starting Rollout gRPC server...")
    serve()
    
    