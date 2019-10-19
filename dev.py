from qenv import QEnv

qcircuit = None
env = QEnv(qcircuit, max_iter=3)

done = False
action = -1
total_iterations = 0

while not done:
    state, reward, done, _ = env.step(action)
    print(f"{total_iterations} - observed state: {state}, reward: {reward}")
    action = -1
    total_iterations += 1
