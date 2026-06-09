
The paper @assloujQuadcopterTrackingUsing2023 compares the Mellinger controller and Snap controller having from it obtained the Mellinger controller equations.

The Mellinger controller is a differential-flatness-based quadrotor trajectory tracking controller introduced in @mellingerMinimumSnapTrajectory2011 It is designed to track aggressive trajectories while remaining computationally lightweight and suitable for real-time implementation. In the paper, the controller is reformulated using rotation matrices instead of Euler angles to avoid ambiguity between Euler angle conventions.

The controller is structured as a **cascade** of:
1. a **position controller**
2. an **attitude controller**.
![[Pasted image 20260512084758.png]]


The position controller computes a desired force vector:  
$$
F_{\mathrm{des}} = 

- K_p (p - p^*)
    
- K_v (\dot r - \dot r^*)
    

- mg\hat{k}
    
- m\ddot r^*  $$
where:
- $p$ is the current position,
- $p^*$ is the desired trajectory position,
- $K_p$ and $K_v$ are positive definite gain matrices,
- $m$ is the quadrotor mass,
- $g$ is gravity,
- $\hat{k}$ is the inertial vertical axis.
The desired thrust is obtained by projecting the desired force onto the body (z)-axis:  
$$  
p_{\mathrm{des}} = F_{\mathrm{des}} \cdot \hat{k}_B  
$$
where $(\cdot)_{des}$ means the controller generated desired states, and the desired body direction is:  
$$  
\hat{k}_{B,\mathrm{des}} = \frac{F_{\mathrm{des}}}{|F_{\mathrm{des}}|}  
$$Using the heading vector $\hat h$, the desired body-frame basis vectors are constructed as:  
$$  
\hat{i}_{B,\mathrm{des}} = 
\frac{  
(\hat{k}\times \hat{h}) \times \hat{k}_{B,\mathrm{des}}  
}{  
|(\hat{k}\times \hat{h}) \times \hat{k}_{B,\mathrm{des}}|  
}  
$$
$$  
\hat{j}_{B,\mathrm{des}} = 
\hat{k}_{B,\mathrm{des}}  
\times  
\hat{i}_{B,\mathrm{des}}  
$$The desired orientation matrix $R_{\mathrm{des}}$ is then compared against the current orientation $R$ to define the attitude error:  
$$  
e_R = 
\frac12  
\left(  
R_{\mathrm{des}}^T R - R^T R_{\mathrm{des}}  
\right)^\vee  
$$
where $(\cdot)^\vee$ denotes the vee-map from skew-symmetric matrices to vectors and $(\cdot)^T$ the transpose matrix .

The attitude controller computes the desired torque:  
$$  
\tau_{\mathrm{des}} =  - K_R e_R  - K_\omega e_\omega  
$$
with angular velocity error:  
$$  
e_\omega = \omega - \omega^* 
$$  
where:
- $K_R$ and $K_\omega$ are diagonal gain matrices,
- $\omega$ is the current angular velocity,
- $\omega^* $ is the trajectory angular velocity.

A major advantage of the Mellinger controller is its modular cascaded architecture, which allows it to be adapted into an attitude controller more easily than the Snap controller. However, the paper reports that the Mellinger controller performs best when tuned with complex poles in the attitude subsystem, whereas the Snap controller achieved lower tracking error overall.

The paper also highlights an important practical limitation: instability is often caused not by the controller design itself, but by actuator saturation. Aggressive trajectories may require rotor speeds outside physically achievable bounds, especially negative thrust, which standard quadrotors cannot generate.

