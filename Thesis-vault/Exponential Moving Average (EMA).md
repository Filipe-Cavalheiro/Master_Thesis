The **Exponential Moving Average (EMA)** is a simple recursive filter used to estimate the underlying state of a signal while reducing the effect of noise. Unlike a simple moving average, which gives equal weight to a fixed window of past observations, the EMA assigns **exponentially decaying weights** to older data. This makes it particularly useful in systems where _recent information should influence decisions more strongly than older information_.

Mathematically, the EMA is defined recursively as:

$$
x_{t} = \alpha y_{t} + (1-\alpha)x_{t-1}  
$$

where:

- $y_t$ is the new observation,
    
- $x_t$ is the filtered estimate,
    
- $x_{t-1}$ is the previous estimate,
    
- $\alpha \in (0,1]$ is the smoothing parameter.
    

The parameter $\alpha$ controls the **responsiveness vs. stability trade-off**:

- Larger $\alpha$ → faster response to new data but more sensitivity to noise.
    
- Smaller $\alpha$ → smoother estimate but slower adaptation.
    

In **control systems**, the EMA can be interpreted as a **first-order low-pass filter**. It filters measurement noise while preserving slow system dynamics. Because it is recursive and requires only the previous state, it is computationally lightweight and well suited for real-time control loops and embedded systems.

In **distributed consensus systems**, EMA-like updates appear naturally in **iterative averaging algorithms**. Each agent updates its estimate by blending its previous belief with new information (from sensors, neighbours, or observations). The exponential weighting implicitly encodes **trust decay over time**: older information gradually loses influence as new data arrives.

From a dynamical systems perspective, the EMA forms a **stable linear update rule**. The factor $1-\alpha$ acts as a memory coefficient, determining how strongly past estimates persist. This makes the EMA a useful building block for **adaptive estimators**, **online learning algorithms**, and **consensus protocols**, where the goal is to gradually converge to a stable shared estimate in the presence of noise and evolving information.

In summary, the exponential moving average can be viewed as a minimal **consensus-with-memory mechanism**: each update blends prior belief and new evidence through a controlled gain, balancing responsiveness, stability, and robustness.