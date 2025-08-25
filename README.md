# PPO Environment Rollout with AWS Fargate

This repository showcases a **Proximal Policy Optimization (PPO)** reinforcement learning setup deployed via **AWS Fargate**.  
The goal is to demonstrate a **CI/CD pipeline** for scaling environment rollouts in the cloud using GRPC Communication protocol.

---

## Overview

- **PPO Algorithm**: Efficient policy optimization for reinforcement learning.
- **Environment Rollouts**: Run in serverless containers using AWS Fargate.
- **CI/CD Pipeline**: Automates deployment, execution, and monitoring of rollouts.
- **Scalability**: Multiple containers execute environments in parallel, improving throughput.

### Architecture Diagram
```
+----------------+ +-------------------+ +----------------+
| PPO Trainer | <----> | Env Rollouts | <----> | AWS Fargate |
| Local       |        | (Containers) |        | (Containers)|
+----------------+ +-------------------+ +----------------+
```

- **PPO Trainer**: Central component that collects rollouts and updates policy.
- **Env Rollouts**: Containerized environments deployed dynamically via Fargate.
- **AWS Fargate**: Serverless compute that runs rollout tasks in parallel.

---

## Pipeline Description

1. **Code Commit**: Push updates to the repository.
2. **CI/CD Trigger**: Pipeline builds Docker images for PPO and environment rollout.
3. **Deployment**: AWS Fargate tasks are launched with the latest rollout code.
4. **Rollout Execution**: Environments execute, generating state-action samples.
5. **Policy Update**: Collected rollouts are processed and fed to PPO trainer (optional visualization/logging).

---

## Features

- Serverless environment rollout using Fargate.
- Parallel execution of multiple environment containers.
- Full CI/CD automation for reproducible PPO deployments.
- Showcases cloud-native reinforcement learning architecture.

---

## Notes

- This repository is **for showcasing purposes only**.  
- The code is **not runnable by default** outside of a configured AWS Fargate environment.
- Focus is on demonstrating CI/CD pipeline, container deployment, and rollout orchestration.

---

## Author

Srikanth Popuri

