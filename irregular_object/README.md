

# Feedback Favors the Generalization of Neural ODEs

This project uesd the developed feedback neural networks to predicte the trajectory of a free-flying irregular object.  

| Files                            | Introduction                                                 | Details                                  |
| -------------------------------- | ------------------------------------------------------------ | ---------------------------------------- |
| Folder: **Dataset_construction** | Construct training datasets (21 trajectories) and testing datasets (9 trajectories) from the real sampled data (Yakult_empty.mat) with a matlab .m program (Data_for_NeuralODE.m). | ---                                      |
| Folder: **png**                  | Storage training and testing results.                        | ---                                      |
| Folder: **trained_model**        | Storage trained models of Neural ODE.                        | ---                                      |
| **training.py**                  | Train the neural ODE with training datasets.                 | <img src="png\trained_results0925.png"/> |
| **testing.py**                   | Test different methods with testing datasets.                | <img src="png\testing_errors0925.png"/>  |

