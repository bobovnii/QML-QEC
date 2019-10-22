# QML-QEC

<img src="img/QML-QEC.png" width='20%' align='right'>

### Quantum Futures Hackathon at CERN

We developed an alternative approach for quantum error mitigation of noisy quantum hardware, inspired by variational algorithms such as QVECTOR. 

We used a noise model based on the hardware of the system and applied it selectively through the circuit. This was key to lower the complexity of the number of gates required in order to reduce noise in the system. We classically optimized the parameters for a single denoising unitary. Now we can place the unitary before the quantum circuit in order to prepare the quantum data so the errors from noise are reduced. 

### Team
Illia Babounikau, Sadra Boreiri, David Ittah, Jake Malliaros, Dmitry Grinko, Anton Karazeev 

![](img/VQECSlide.png)

![](img/HistogramSlide.png)

### Installation

You need to pass your `API key` for IBM Q Experience service: put it in [res/qiskit_apikey.json](res/qiskit_apikey.json) file.
