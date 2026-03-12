
#  MNIST Neural Network Simulator (for MC)

This project is a hybrid of machine learning and redstone engineering: a fully simulated neural network trained on the **MNIST handwritten digits dataset**, with the ultimate goal of recreating its behavior **within Minecraft** using redstone logic.

Linkedin Post - https://www.linkedin.com/feed/update/urn:li:activity:7314837008021950464/ 

Inspired by the groundbreaking work of [MattBatWings](https://www.youtube.com/@MattBatWings), this project aims to bridge the gap between educational AI tools and Minecraft's mechanical sandbox.

##  Project Goal

Simulate a basic neural network architecture that can classify MNIST digits. Visualize and translate the network's logic into redstone-compatible circuits. Recreate the entire pipeline (input, weights, activation, output) using Minecraft mechanics like redstone torches, comparators, pistons, and observers.

##  Technologies Used

- Python (NumPy, Matplotlib)
- MNIST Dataset (28x28 grayscale images)
- Feedforward Neural Network Architecture
- Manual backpropagation (for redstone logic compatibility)
- ASCII and pixel-to-block map visualization
- Minecraft Java Edition (Creative Mode with redstone building)

##  Architecture

- Input Layer: 784 nodes (28x28 grayscale)
- Hidden Layer(s): Customizable, default 16–32 neurons
- Output Layer: 10 nodes (digits 0–9)
- Activation Function: Sigmoid
- Loss Function: Mean Squared Error
- Optimizer: Manual backpropagation to simulate redstone compatibility

##  How to Run the Simulation

Clone the repo:
```bash
git clone https://github.com/your-username/minecraft-mnist-sim.git
cd minecraft-mnist-sim
