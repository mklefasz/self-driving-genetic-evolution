# self-driving-genetic-evolution

 
A 2D simulation built from scratch in Pygame utilizing a custom multi-layer neural network and a genetic algorithm (Neuroevolution) to train autonomous cars to navigate tracks via raycasted distance sensors.

##  Tech Stack & Concepts
* **Simulation Engine:** Pygame (Physics, Raycasting, and Rendering Loop)
* **Math & Processing:** NumPy (Vectorized matrix operations)
* **AI Architecture:** Multi-Layer Perceptron (MLP) built from scratch (Forward propagation via $\tanh$ activation)
* **Optimization Engine:** Genetic Algorithm (Neuroevolution featuring Elitism, Roulette Wheel/Random Selection, and Single-Point Crossover)

##  Architecture Breakdown

### 1. Perceptual System (Raycasting Sensors)
The vehicle scans its environment using $5$ proximity sensors arrayed at fixed angles ($0^\circ$, $\pm30^\circ$, $\pm90^\circ$). Each sensor acts as a raycaster, calculating distance boundaries based on background pixel color checks.

### 2. Custom Neural Network Layer
Instead of relying on heavy frameworks like PyTorch, the network's forward propagation pass is handled explicitly via raw matrix math:
$$Z_1 = X \cdot W_1 + B_1$$
$$A_1 = \tanh(Z_1)$$
$$Z_2 = A_1 \cdot W_2 + B_2$$
$$A_2 = \tanh(Z_2)$$

The resulting output vector maps directly to continuous vehicle control commands (Steering angle velocity adjustments and Throttle/Braking).

### 3. Evolutionary Optimization (Genetic Algorithm)
* **Fitness Evaluation:** Vehicles are penalized for stagnating/low velocity and rewarded dynamically based on absolute distance traveled within the generation time limit.
* **Crossover:** Pairs of high-performing parent genomes combine weight arrays at dynamic splicing intervals.
* **Mutation:** Introduces stochastic variations via Gaussian noise additions based on configurable mutation rates.

 
