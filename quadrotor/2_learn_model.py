import numpy as np

""" MLP learning using mini-batching """

from learning_MLP.trainer import Regression_Trainer
from model.model_drag_MLP import Drag_MLP

trainer = Regression_Trainer(model_res=Drag_MLP, param_reg=1e-4)

params = trainer.optimize_minibatch(learning_rate=1e-3, batch_size=100, epochs=500)

trainer.pth2params_savenpy(
    model_class=Drag_MLP,
    pth_path="learning_MLP/temp/model_param.pth",
    npy_path="learning_MLP/temp/model_param.npy",
)

""" Neural ODE learning using mini-batching """

from learning_NODE.trainer import E2E_Trainer
from model.model_NODE import Quadrotor_NODE

model = Quadrotor_NODE(discrete_h_learn=0.02)
trainer = E2E_Trainer(model_wrapper=model, param_reg=1e-4)

params_init = (np.random.rand(model.params_dim, 1) - 0.5) * 0.1

params = trainer.optimize_minibatch(
    params_init,
    step_init=1e-3,
    minibatch_size=100,
    epochs=10,
)
