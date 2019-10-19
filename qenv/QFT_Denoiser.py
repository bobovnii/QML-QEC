from qiskit import Aer, IBMQ, execute
from qiskit.providers.aer import noise
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.tools.visualization import plot_histogram
from qiskit.providers.ibmq.exceptions import IBMQAccountError
import math


class Denoiser():
    def __init__(self, n, APIkey):
        self.n = n

        # Load IBMQ backend used for noise modeling
        try:
            provider = IBMQ.enable_account(APIkey)
        except IBMQAccountError as e:
            provider = IBMQ.get_provider(hub='ibm-q')
            print(e)
        device = provider.get_backend('ibmq_ourense')
        properties = device.properties()

        # Generate noise model from device
        self.noise_model = noise.device.basic_device_noise_model(properties)

        # Select the QasmSimulator from the Aer provider
        self.simulator = Aer.get_backend('qasm_simulator')

    def get_dist(self, psi, theta, shots=10000, noise=True, init=True):
        # reset to new circuit
        self.q = QuantumRegister(self.n)
        self.c = ClassicalRegister(self.n)
        self.qc = QuantumCircuit(self.q, self.c)

        self.qc.initialize(psi, self.q)
        self._qft_inv()

        # Select the statevector simulator for noise-free simulation
        simulator = Aer.get_backend('statevector_simulator')
        self.psi = execute(self.qc, simulator).result().get_statevector()

        # ------------------------------------------------------------- #

        self.q = QuantumRegister(self.n)
        self.c = ClassicalRegister(self.n)
        self.qc = QuantumCircuit(self.q, self.c)
        
        # run noisy part of circuit
        if init:
            self.qc.initialize(self.psi, self.q)
        self._denoise(theta)
        self._qft()
        self.qc.measure(self.q, self.c)

        if noise:
            result = execute(self.qc, self.simulator, noise_model=self.noise_model, shots=shots).result()
        else:
            result = execute(self.qc, self.simulator, shots=shots).result()

        counts = result.get_counts()
        return [value/shots for (key, value) in sorted(counts.items())]

    def _qft(self):
        """n-qubit QFT on q in circ."""
        for j in range(self.n):
            self.qc.h(self.q[j])
            for k in range(j+1, self.n):
                self.qc.cu1(math.pi/float(2**(k-j)), self.q[k], self.q[j])
            self.qc.barrier()

    def _qft_inv(self):
        """n-qubit QFT on q in circ."""
        for j in range(self.n-1, -1, -1):
            for k in range(self.n-1, j, -1):
                self.qc.cu1(-math.pi/float(2**(k-j)), self.q[k], self.q[j])
            self.qc.h(self.q[j])
            self.qc.barrier()

    def _denoise(self, theta):
        for j in range(self.n):
            x, y, z = theta[j]
            self.qc.u3(x, y, z, self.q[j])
