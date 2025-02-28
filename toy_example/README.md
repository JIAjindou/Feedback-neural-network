# Feedback Infavors the Generalization of Neural ODEs

This document describes how to use the codes of feedback neural networks presented in the paper [1]. A toy example is first introduced to illustrate the basic idea of feedback neural networks. Then all codes to reproduce three implemented tests in the paper [1] are provided. The implemented tests include trajectory prediction of a spiral_curve, trajectory prediction of an irregular object, and model predictive control of the quadrotor.

## Installation

```python
pip install numpy==1.23.5
pip install torch
pip install torchdiffeq
pip install matplotlib
```

## Toy example

The toy example is placed in the **toy_example** repository.

#### 1. Neural ODE with a linear feedback

Consider a spiral curve:
$$
\left[ {\begin{array}{*{20}{c}}
{\dot {x}}\\
{\dot {y}}
\end{array}} \right] = \left[ {\begin{array}{*{20}{c}}
{ {- 0.1}}&{2}\\
{ - 2}&{- 0.1}
\end{array}} \right]\left[ {\begin{array}{*{20}{c}}
{x}\\
{y}
\end{array}} \right]
$$
Neural ODE [2] can learn its latent dynamics accurately with collected dataset $\{x_i,y_i\}$:

<img src="png_readme\Neural_ODE_in_training.png" alt="Neural_ODE_in_training" style="zoom:25%;" />

However,  as the learned ODE model is employed in a new case:
$$
\left[ {\begin{array}{*{20}{c}}{\dot {x}}\\{\dot {y}}\end{array}} \right] = \left[ {\begin{array}{*{20}{c}}{ {- 0.05}}&{3}\\{ - 3}&{- 0.05}\end{array}} \right]\left[ {\begin{array}{*{20}{c}}{x}\\{y}\end{array}} \right]+\left[ {\begin{array}{*{20}{c}}{10}\\{10}\end{array}} \right]
$$
the learning performance of latent dynamics will degrade:

<img src="png_readme\Neural_ODE_in_testing.png" alt="Neural_ODE_in_testing" style="zoom:25%;" />

With the proposed linear feedback form, the trained model by neural ODE can generalize to the new case very well:

<img src="png_readme\FNN_in_testing.png" alt="FNN_in_testing" style="zoom:25%;" />

which further improves the trajectory prediction performance:

<img src="png_readme\FNN_prediction.png" alt="FNN_prediction" style="zoom: 33%;" />



Run followed code in the terminal to obtain the above results.

```
python linear_feedback.py --viz
```

#### 2. Neural ODE with a neural feedback

A linear feedback form can promptly improve the adaptability of neural ODEs in unseen scenarios. However, two improvements could be further made. At first, it would be more practical if the gain tuning procedure could be avoided. Moreover, the linear feedback form can be extended to a nonlinear one to adopt more intricate scenes, as experienced in the control field. Thus in this part, we try to learn a neural feedback.

Domain randomization can improve the generalization ability for sim-to-real missions. However, the training objective is forced to focus on the average performance among different parameters, such that the prediction ability on the previous nominal task (1) will degraded:

<img src="png_readme\Neural_ODE_withDR_in_training.png" alt="Neural_ODE_withDR_in_training" style="zoom: 40%;" />

Run followed code in the terminal to obtain the above result.

```
python neural_ODE_withDR.py --viz
```

Run followed code in the terminal to train the neural Feedback.

```
python neural_feedback.py --viz
```

With the neural feedback, not only the accuracy performance on  the previous nominal task (1) can be maintained:

<img src="png_readme\Neural_ODE_withNF_in_training.png" alt="Neural_ODE_withNF_in_training" style="zoom:40%;" />

but also the generalization on new cases can be achieved:

<img src="png_readme\Neural_ODE_withNF_in_testing.png" alt="Neural_ODE_withNF_in_testing" style="zoom: 20%;" />

Run followed code in the terminal to obtain the above result.

```
python neural_feedback_test.py --viz
```

## Implemented experiments

#### 1. Trajectory prediction of a spiral_curve

Github repository link: https://github.com/anonymity-tempo/Feedback_NN_spiral_curve.git

Apart from the description in the above **Toy example**, more tests including gain decay strategy and ablation study with different uncertainties and gains are provided. Details refer to the following file description.

| Files                        | description                                                  |
| ---------------------------- | ------------------------------------------------------------ |
| Folder: **past_vision**      | Store intermediate versions of .py files.                    |
| Folder: **png**              | The auto-storage directory in which the program runs. Store intermediate test results. |
| Folder: **final_png**        | Store final test results, preparing for the paper.           |
| Folder: **trained_model**    | Trained NN models.                                           |
| **a_one_step_pre.py**        | One-step prediction program, to test the feedback mechanism. |
| **a_multi_steps_pre.py**     | Multi-steps prediction program.                              |
| **b_ablation_L_data.py**     | Collect data with different degrees of uncertainty and different L levels. The different L needs to be set artificially. |
| **b_ablation_L_heatmap.py**  | The plot program of b_ablation_L_data.py.                    |
| **c_neural_ODE_nominal.py**  | Train Neural ODE on the nominal task and store the trained model (Neural_ODE.pt). |
| **c_neural_ODE_domran.py**   | Train Neural ODE through domain randomization and plot the degraded performance on the nominal task. |
| **d_FeedbackNN.py**          | Train feedback neurons through domain randomization, store the trained model (FeedbackNN.pt), and plot the mataining performance on the nominal task. |
| **d_FeedbackNN_converge.py** | The convergence procedure is revealed in training dataset    |
| **d_FeedbackNN_test.py**     | Test trained model  FeedbackNN.pt on randomized tasks, to show its generalization. |

#### 2. Trajectory prediction of an irregular object

Github repository link: https://github.com/anonymity-tempo/Feedback_NN_irregular_object.git

Precise trajectory prediction of a free-flying irregular object is a challenging task due to the complicated aerodynamic effects.  We test the effectiveness of the proposed method on an open-source dataset, in comparison with the model-based method and the learning-based method. The objective of this mission is to accurately predict the object's position after $0.5\ s$. 

Details of this project refer to the following file description.

| Files                            | Introduction                                                 |
| -------------------------------- | ------------------------------------------------------------ |
| Folder: **Dataset_construction** | Construct training datasets (21 trajectories) and testing datasets (9 trajectories) from the real sampled data (Yakult_empty.mat) with a matlab .m program (Data_for_NeuralODE.m). |
| Folder: **png**                  | Storage training and testing results.                        |
| Folder: **trained_model**        | Storage trained models of Neural ODE.                        |
| **training.py**                  | Train the neural ODE with training datasets.                 |
| **testing.py**                   | Test different methods with testing datasets.                |

#### 3. Model predictive control of the quadrotor

Github repository link: https://github.com/anonymity-tempo/Feedback_NN_MPC.git

MPC works in the form of receding-horizon trajectory optimizations with a dynamic model and then determines the current optimal control input. Approving optimization results highly rely on accurate dynamical models. The proposed feedback neural network is employed on the quadrotor trajectory tracking scenario concerning model uncertainties and external disturbances, to demonstrate its online adaptive capability. 

The adjoint sensitive method is employed in [2] to train neural ODEs without considering external inputs. We provide an alternative training strategy in the presence of external inputs, from the view of optimal control.

Details of this project refer to the following file description.

| Directory                 | Introduction                                                 |
| ------------------------- | ------------------------------------------------------------ |
| Folder: **aux_module**    | Modules required for visualization and polynomial trajectory generation. |
| Folder: **img**           | Storage of visualization results.                            |
| Folder: **learning**      | The learning framework that allow training neural odes or neural ode augmented models with auxiliary inputs using minibatching. |
| Folder: **model**         | Models used for simulations and learning.                    |
| Folder: **mpc**           | Include **solver.py** containing a standard MPC and a MPC with mutli-step prediction algorithm. |
| Folder: **sim_mpc_trajs** | Storage of simulated trajectories of different MPC setups.   |
| **1_mk_traindata.py**     | Make dataset for training.                                   |
| **2_dynamics_learn.py**   | Learn the neural ODE augmented dynamics.                     |
| **3_visualize_learn.py**  | Plot training trajectories, test results and loss evaluation. |
| **4_mpc_cases.py**        | Trajectory tracking simulation with different MPC setups.    |
| **5_visualize_mpc.py**    | Plot simulated trajectories of different MPC setups.         |

## Reference

[1] Feedback Infavors the Generalization of Neural ODEs, ArXiv, 2024.

[2] Ricky TQ Chen, Yulia Rubanova, Jesse Bettencourt, and David K Duvenaud. Neural ordinary differential equations. In Proceedings of Advances in Neural Information Processing Systems, volume 31, 2018.

