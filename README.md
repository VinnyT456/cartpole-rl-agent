# CartPole DQN Agent

This repository implements a Deep Q-Network (DQN) agent trained to solve the classic CartPole balancing task from OpenAI Gym. It includes reward shaping, prioritized experience replay, and evaluation scripts to monitor model progress and generalization.

---

## 🧠 Key Features

* ✅ **Double Deep Q-Network** with ELU activations
* 🎯 **Prioritized Experience Replay (PER)** with dynamic TD-error updates
* 📈 **Custom reward shaping** based on pole angle and corrective actions
* 📉 **Live training tracking** with average rewards and model checkpointing
* 🧪 **Testing script** to evaluate generalization over multiple episodes

---

## 📁 Directory Structure

```
📁 CartPole-DQN-Agent
├── model.py            # DQN model definition
├── train.py            # Training loop and experience replay logic
├── test.py             # Offline testing over 1000 episodes
├── best_cartpole_model.pth # Best model checkpoint
├── requirements.txt    # Dependencies
└── README.md           # Project overview
```

---

## 🚀 Getting Started

### 1. Setup Environment

```bash
pip install -r requirements.txt
```

### 2. Train the Agent

```bash
python train.py
```

* Trains agent for 1000 episodes
* Applies epsilon-greedy policy with decay
* Saves the best model based on highest episode reward

### 3. Test the Agent

```bash
python test.py
```

* Loads the trained model
* Evaluates performance across 1000 episodes
* Reports average reward and success rate

---

## 📊 Evaluation (Post-Training)

| Metric          | Result (Approx.) |
| --------------- | ---------------- |
| Avg. Reward     | 500.00           |
| Episodes Solved | 100%             |

---

## ⚙️ Model Overview

* **Architecture:**

  * QNetwork: `state_dim -> 32 -> 16 -> 8 -> action_dim`
* **Activation:** ELU
* **Loss:** Weighted MSE with PER
* **Optimizer:** Adam
* **Discount Factor:** Gamma = 0.99
* **Learning Rate:** lr = 1e-3
* **Batch Size** batch_size = 128
* **Replay Buffer:** With priorities and TD-error adjustment
* **Exploration:** Epsilon decay from 1.0 to 0.01

---

## 🔧 Enhancements

* **Custom reward shaping** based on pole lean angle and corrective moves
* **Transition rejection** if episode ends in <10 steps
* **Max-priority injection** for episodes scoring a perfect 500

---

## 🔮 Potential Next Steps

* Normalize or scale pole angle inputs
* Experiment with different activation functions or layer sizes
* Visualize learned policy behavior over time

---

## 🤝 Acknowledgements

* OpenAI Gym (CartPole-v1)
* PyTorch
* Guidance and insights from ChatGPT

---
