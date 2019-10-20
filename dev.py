# from qenv import QEnv
#
# qcircuit = None
# env = QEnv(qcircuit, max_iter=3)
#
# done = False
# action = -1
# total_iterations = 0
#
# while not done:
#     state, reward, done, _ = env.step(action)
#     print(f"{total_iterations} - observed state: {state}, reward: {reward}")
#     action = -1
#     total_iterations += 1


# from qenv import Denoiser
# import json
# import numpy as np
#
# key_path = "res/qiskit_apikey.json"
# with open(key_path, 'r') as f:
#     APIkey = json.load(f)['key']
#
# n_qubits = 1
# denoiser = Denoiser(n_qubits, APIkey)
#
# psi = [1, 0]
# theta = [(np.pi / 4, 0, 0)]
# print(denoiser.get_dist(psi, theta))


from qenv import Compare

cmp = Compare()
cmp.run(plot=True)
