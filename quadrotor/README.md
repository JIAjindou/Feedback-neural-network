## Feedback Favors the Generalization of Neural ODEs: Robotics Application

**Learning Aerodynamic Effects + Model Predictive Control using FNN**

This repository contains the code related to the application in **quadrotor flight control** with the proposed feedback neural network. The aerodynamic effects are firstly learned using a neural ODE augmented model, and then embedded into a MPC controller with multi-step prediction method that utilizes feedbacks for higher precision accuracy and control performance.

## Files introduction

| Directory                 | Introduction                                                 | Details                                                      |
| ------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Folder: **aux_module**    | Modules required for visualization and polynomial trajectory generation. | ---                                                          |
| Folder: **img**           | Storage of the visualization results.                        | ---                                                          |
| Folder: **learning_NODE** | The learning framework that allow training neural odes or neural ode augmented models with auxiliary inputs using mini-batching. | **core.py** contains differentiable numerical integrators, defined loss functions and data-processing functions. **trainer.py** contains the main loop of learning together with the analytic gradient computation algorithm. |
| Folder: **learning_MLP**  | Learning a neural ODE in label-feature fashion (state derivatives are given) | Implementation with PyTorch                                  |
| Folder: **model**         | Models used for simulations and learning.                    | **model_learn.py**: the neural ODE augmented model for learning, **model_nominal.py**: a nominal model with no learning-based parts, **model_sim.py**: a model used for simulation, including more realistic settings. |
| Folder: **mpc**           | Include **solver.py** that contains various kinds of MPCs for validation. | ---                                                          |
| Folder: **sim_mpc_trajs** | Storage of the simulated trajectories.                       | ---                                                          |
| **1_mk_traindata.py**     | Make dataset for training.                                   | ---                                                          |
| **2_model_learn.py**      | Train the neural ODE augmented dynamics.                     | ---                                                          |
| **3_model_eval.py**       | Evaluation of models in training and test sets.              | The test sets are 3 randomly generated trajectories.         |
| **4_visualize_learn.py**  | Plot training trajectories and test results.                 | ---                                                          |
| **5_mpc_cases.py**        | Trajectory tracking simulation with different MPC setups.    | ---                                                          |
| **6_visualize_mpc.py**    | Plot simulated trajectories of different MPC setups.         | ---                                                          |

