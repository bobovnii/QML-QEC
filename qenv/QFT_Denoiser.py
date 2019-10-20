from qiskit import Aer, IBMQ, execute
from qiskit.providers.aer import noise
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.tools.visualization import plot_histogram
from qiskit.tools.monitor import job_monitor
from qiskit.providers.ibmq.exceptions import IBMQAccountError
from qiskit.extensions.standard import IdGate
from qiskit.transpiler.passes import Unroller
from qiskit.transpiler import PassManager
import qiskit.extensions.simulator.snapshot
import math
import random as rand


class Denoiser:
    """
        Initialize class as:
            denoiser = Denoiser(n, APIKEY)
                where n is the number of qubits, APIKEY is your key from your ibmq experience account
        Run the circuit using:
            denoiser.get_dist(psi, theta)
                which returns the probability distribution of the final state as a list of probabilities
                    for each basis state, ordered numerically according to its binary representation, and
                where psi is the input state in the same format as above, e.g. [1,0,0,0] for two qubits,
                    and theta are the parameters to the denoiser circuit, as a list of tuples (x, y, z)
                    specifiying the unitary correction, one tuple for each qubit
    """

    def __init__(self, n, APIkey):
        self.n = n
        self.q = QuantumRegister(self.n)
        self.c = ClassicalRegister(self.n)

        # Load IBMQ backend used for noise modeling
        try:
            provider = IBMQ.enable_account(APIkey)
        except IBMQAccountError as e:
            provider = IBMQ.get_provider(hub="ibm-q")
            print(e)

        device = provider.get_backend('ibmq_16_melbourne')
        properties = device.properties()

        # Generate noise model from device, without measurement error
        self.noise_model = noise.device.basic_device_noise_model(properties, readout_error=False, temperature=100)

        # Select the QasmSimulator from the Aer provider
        self.simulator = Aer.get_backend("qasm_simulator")

        # creatre QFT_inv instruction
        qc = QuantumCircuit(self.q, self.c, name='QFTdg')
        for j in range(self.n-1, -1, -1):
            for k in range(self.n-1, j, -1):
                qc.cu1(-math.pi/float(2**(k-j)), self.q[k], self.q[j])
            qc.h(self.q[j])
        self.qft_inv = qc.to_instruction()
        
        # creatre QFT instruction
        qc = QuantumCircuit(self.q, self.c, name='QFT')
        for j in range(self.n):
            qc.h(self.q[j])
            for k in range(j+1, self.n):
                qc.cu1(math.pi/float(2**(k-j)), self.q[k], self.q[j])
        self.qft = qc.to_instruction()
        
        self.pm = PassManager(Unroller(['u3', 'cx', 'QFT', 'QFTdg', 'initialize']))
                
    def get_dist(self, psi, theta, shots=1, force_noise=False, noise=True, init=True):
        # run ideal part of circuit
        self.qc = QuantumCircuit(self.q, self.c)

        if init:
            self.qc.initialize(psi, self.q)
            
        self.qc.append(self.qft_inv, self.q, [])
        
        # run noisy part of circuit
        self._denoise(theta)
        
        if force_noise:
            self._apply_noise()

        self._qft()
        #self.qc.append(self.qft, self.q, [])

        self.qc.snapshot('state')
        self.qc.measure(self.q, self.c)
        
        self.qc = self.pm.run(self.qc)

        if noise:
            self.result = execute(self.qc, self.simulator, noise_model=self.noise_model, backend_options={"fusion_enable":True}, shots=shots).result()
        else:
            self.result = execute(self.qc, self.simulator, backend_options={"fusion_enable":True}, shots=shots).result()

        return self.result.data()['snapshots']['statevector']['state']

    def _qft(self):
        """n-qubit QFT on q in circ."""
        for j in range(self.n):
            self.qc.h(self.q[j])
            for k in range(j + 1, self.n):
                self.qc.cu1(math.pi / float(2 ** (k - j)), self.q[k], self.q[j])
            self.qc.barrier()

    def _qft_inv(self):
        """n-qubit QFT on q in circ."""
        for j in range(self.n - 1, -1, -1):
            for k in range(self.n - 1, j, -1):
                self.qc.cu1(-math.pi / float(2 ** (k - j)), self.q[k], self.q[j])
            self.qc.h(self.q[j])
            self.qc.barrier()

    def _denoise(self, theta):
        for j in range(self.n):
            x, y, z = theta[j]
            self.qc.u3(x, y, z, self.q[j])

    def _apply_noise(self):
        for i in range(self.n):
            if rand.random() < 0.5:
                self.qc.z(i)
