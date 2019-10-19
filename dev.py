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

from qenv import Denoiser
import json

key_path = "res/qiskit_apikey.json"
with open(key_path, 'r') as f:
    APIkey = json.load(f)['key']

n_qubits = 2
denoiser = Denoiser(2, APIkey)
print(denoiser)