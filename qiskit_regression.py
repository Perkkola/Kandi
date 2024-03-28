import numpy as np
import pickle
from qiskit import QuantumCircuit
from qiskit.algorithms.gradients import *
from qiskit.algorithms.optimizers import *
from qiskit.circuit import Parameter
from sklearn.decomposition import PCA
from qiskit.circuit.library import RealAmplitudes
from qiskit_machine_learning.algorithms.regressors import NeuralNetworkRegressor
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import Estimator as LocalEstimator
from qiskit_ibm_runtime import QiskitRuntimeService, Estimator, Session, Options
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from typing import List, Literal
from sklearn.model_selection import train_test_split
import pandas as pd
import os

class MultiTargetQuantumNNRegressor:
    r"""MultiTargetQuantumNNRegressor
    This class creates a multivariate multiple regression model by instantiating a separate quantum
    neural network for each target output. This class automatically constructs the necessary
    quantum circuits for the feature map and an ansatz for the trainable weights to encode
    classical data into a quantum state.
    """
    def __init__(
    self,
    n_features: int,
    n_targets: int,
    init,
    optimizer: Literal['cobyla', 'adam', 'lbfgsb', 'slsqp'],
    maxiter: int = 50,
    use_quantum_cloud: bool = False,
    backend: str = None,
    ) -> None:
        r"""
        Parameters
        ----------
        n_features: int
        The number of features in the regression model.
        n_targets: int
        The number of regression targets i.e. the dimension of the y output vector.
        optimizer: {'cobyla', 'adam', 'lbfgsb','slsqp'}, default = 'slsqp'
        The solver for weight optimization.
        maxiter: int, default = 50
        Maximum number of iterations. The solver iterates until convergence.
        use_quantum_cloud: bool
        Determines whether to run on CPU or IBM Quantum Cloud.
        backend: str, default = None
        If running on quantum cloud, name of quantum hardware or simulator instance.
        """

        self._n_features = n_features
        self._n_targets = n_targets
        self._use_quantum_cloud = use_quantum_cloud
        self._backend = backend
        self.objective_func_vals = []
        self.weights = []
        self.scores = []
        self.init = init
        if self._use_quantum_cloud:
            if self._backend is None:
                self._backend = 'ibmq_qasm_simulator'
        if optimizer is None or optimizer == 'slsqp':
            self._optimizer = SLSQP(maxiter=maxiter)
        elif optimizer == 'cobyla':
            self._optimizer = COBYLA(maxiter=maxiter)
        elif optimizer == 'adam':
            self._optimizer = ADAM(maxiter=maxiter)
        elif optimizer == 'lbfgsb':
            self._optimizer = L_BFGS_B(maxiter=maxiter)
        else:
            raise Exception(f'The selected optimizer is invalid: {optimizer}')
        self._neural_networks: List[NeuralNetworkRegressor] = []

        for _ in range(n_targets):
            feature_map = self._create_feature_map()
            ansatz = self._create_ansatz()
            qc = self._create_quantum_circuit(feature_map=feature_map, ansatz=ansatz)
            estimator_qnn = self._create_estimator_qnn(qc=qc, feature_map=feature_map, ansatz=ansatz)
            regressor = self._create_neural_network_regressor(qnn=estimator_qnn)
            self._neural_networks.append(regressor)

    def _create_feature_map(self):
        """
        Creates a parameterized quantum circuit based on the number of features in the regression model.
        """
        feature_map = QuantumCircuit(self._n_features)
        parameters = []
        for i in range(self._n_features):
            parameter = Parameter(f"x{i}")
            parameters.append(parameter)
            feature_map.ry(parameter, i)
        return feature_map
    
    def _create_ansatz(self):
        """
        Creates an ansatz for the trainable weights.
        """
        ansatz = RealAmplitudes(self._n_features, reps=self._n_features, entanglement='circular')
        return ansatz
    def _create_quantum_circuit(self, feature_map: QuantumCircuit, ansatz: RealAmplitudes):
        """
        Builds the quantum circuit that will be used to encode the classical data.
        """
        qc = QuantumCircuit(self._n_features)
        qc.compose(feature_map, inplace=True)
        qc.compose(ansatz, inplace=True)
        return qc
    def _get_estimator(self):
        """
        Creates an Estimator primitive for use on a classical machine or IBM Quantum Cloud.
        """
        if self._use_quantum_cloud:
            service = QiskitRuntimeService(channel="ibm_quantum")
            backend = service.backend(self._backend)
            session = Session(service=service, backend=backend,max_time=18000)
            options = Options(max_execution_time=18000)
            options.execution.shots = 1024
            estimator = Estimator(session=session, options=options)
            return estimator
        else:
            return LocalEstimator()
    def _create_estimator_qnn(self, qc:QuantumCircuit, feature_map: QuantumCircuit, ansatz:
    RealAmplitudes):
        """
        Creates the Estimator Quantum Neural Network using the constructed quantum circuit and ansatz.
        """
        estimator = self._get_estimator()
        qnn = EstimatorQNN(
        circuit=qc,
        estimator=estimator,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters
        )
        return qnn
    def _create_neural_network_regressor(self, qnn: EstimatorQNN):
        """
        Creates the trainable regression model using an underlying Estimator Quantum Neural Network.
        """
        regressor = NeuralNetworkRegressor(
        neural_network=qnn,
        loss="squared_error",
        optimizer=self._optimizer,
        callback=self.callback_graph,
        initial_point=self.init
        )
        return regressor

    def fit(self, X: np.ndarray, y: np.ndarray, scale_data: bool = False):
        """
        Begins training the neural network corresponding to each output variable.
        Parameters
        ----------
        X: numpy array of shape (n_samples, n_features)
        The input data.
        y: numpy array of shape (n_samples, n_targets)
        The target values.
        scale_data: bool, default = False
        Set this value to true if the dataset is not already scaled down to values between 0 and 1.
        Note: Model training convergence is better when the transformation is applied.
        """
        if X.shape[1] != self._n_features:
            raise ValueError(f"Shapes don't match, X features: {X.shape[1]}, n_features: {self._n_features}!")
        if y.shape[1] != self._n_targets:
            raise ValueError(f"Shapes don't match, y targets: {y.shape[1]}, n_targets: {self._n_targets}!")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"Shapes don't match, X samples: {X.shape[0]}, y samples: {y.shape[0]}!")
        if scale_data:
            X, y = self.get_scaled_data(X, y)
            self._fit(X,y)
    def _fit(self, X: np.ndarray, y: np.ndarray):
        """
        Internally called from self.fit() to begin training each model.
        """
        for i in range(self._n_targets):
            self._neural_networks[i].fit(X, y[:, i])
    @staticmethod
    def get_scaled_data(X: np.ndarray, y: np.ndarray):
        """
        Scales the dataset down to values between 0 and 1.
        Note: Model training convergence is better when the transformation is applied.
        Parameters
        ----------
        X: numpy array of shape (n_samples, n_features)
        The input data.
        y: numpy array of shape (n_samples, n_targets)
        The target values.
        """
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
        if y is not None:
            y = scaler.fit_transform(y)
        return X, y

    @staticmethod
    def get_unscaled_data(X: np.ndarray, y: np.ndarray):
        """
        Reverses the scaling of the X and y values to original values.
        Parameters
        ----------
        X: numpy array of shape (n_samples, n_features)
        The input data.
        y: numpy array of shape (n_samples, n_targets)
        The target values.
        """
        scaler = MinMaxScaler()
        X = scaler.inverse_transform(X)
        if y is not None:
            y = scaler.inverse_transform(y)
        return X, y
    def predict(self, X: np.ndarray, scale_data: bool = False):
        """
        Predict the target values given the input vector.
        Parameters
        ----------
        X: numpy array of shape (n_samples, n_features)
        The input data.
        Returns
        ----------
        y: numpy array of shape (n_samples, n_targets)
        The target values.
        """
        if scale_data:
            X, _ = self.get_scaled_data(X, None)
        if X.shape[1] != self._n_features:
            raise ValueError(f"Shapes don't match, X features: {X.shape[1]}, n_features: {self._n_features}!")
        return self._predict(X)
    def _predict(self, X: np.ndarray):
        """
        Internally called from self.predict().
        """
        predictions = []
        for i in range(self._n_targets):
            prediction = self._neural_networks[i].predict(X)
            predictions.append(prediction)
            final_result = np.reshape(np.stack((predictions), axis= 1), (X.shape[0], len(predictions)))
        return final_result

    def score(self, X: np.ndarray, y: np.ndarray, scale_data: bool = False):
        """
        Compute the coefficient of determination i.e. the R-squared (R2) score.
        Parameters
        ----------
        X: numpy array of shape (n_samples, n_features)
        The input test data.
        y: numpy array of shape (n_samples, n_targets)
        The true target values.
        Returns
        ----------
        r2: float
        The r-squared (R2) score of the predictions.
        """
        if X.shape[1] != self._n_features:
            raise ValueError(f"Shapes don't match, X features: {X.shape[1]}, n_features: {self._n_features}!")
        if y.shape[1] != self._n_targets:
            raise ValueError(f"Shapes don't match, y targets: {y.shape[1]}, n_targets: {self._n_targets}!")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"Shapes don't match, X samples: {X.shape[0]}, y samples: {y.shape[0]}!")
        if scale_data:
            X, y = self.get_scaled_data(X, y)
        return self._score(X, y)
    
    def _score(self, X: np.ndarray, y: np.ndarray):
        """
        Internally called from self.score().
        """
        scores = []
        for i in range(self._n_targets):
            performance_score = self._neural_networks[i].score(X, y[:,i])
            scores.append(performance_score)
        return np.mean(scores)

    def mae(self, y_true, y_pred):
        """
        Computes the Mean Absolute Error.
        Parameters
        ----------
        y_true: numpy array of shape (n_samples, n_targets)
        The true target values.
        y_pred: numpy array of shape (n_samples, n_targets)
        The predicted target values.
        Returns
        ----------
        mae: float
        The mean absolute error regression loss.
        """
        return mean_absolute_error(y_true=y_true, y_pred=y_pred)
    def mse(self, y_true, y_pred):
        """
        Computes the Mean Squared Error.
        Parameters
        ----------
        y_true: numpy array of shape (n_samples, n_targets)
        The true target values.
        y_pred: numpy array of shape (n_samples, n_targets)
        The predicted target values.
        Returns
        ----------
        mae: float
        The mean squared error regression loss.
        """
        return mean_squared_error(y_true=y_true, y_pred=y_pred, squared=True)
    def rmse(self, y_true, y_pred):
        """
        Computes the Root Mean Squared Error.
        Parameters
        ----------
        y_true: numpy array of shape (n_samples, n_targets)
        The true target values.
        y_pred: numpy array of shape (n_samples, n_targets)
        The predicted target values.
        Returns
        ----------
        mae: float
        The root mean squared error regression loss.
        """
        return mean_squared_error(y_true=y_true, y_pred=y_pred, squared=False)

    def save_model(self, model_name: str = 'model'):
        """
        Saves the model to disk.
        Parameters
        ----------
        model_name: str, default = 'model'
        """
        if os.path.exists(model_name):
            os.remove(model_name)
        with open(f'{model_name}', 'wb') as file:
            pickle.dump(self, file)
    @staticmethod
    def load_model(model_name: str = 'model'):
        """
        Loads the saved model from disk.
        Parameters
        ----------
        model_name: str, default = 'model'
        Returns
        ----------
        model: MultiTargetQuantumNNRegressor
        The saved instance of the MultiTargetQuantumNNRegressor.
        """
        with open(model_name, 'rb') as file:
            model = pickle.load(file)
            class_type = MultiTargetQuantumNNRegressor
            if isinstance(model, class_type):
                return model
            else:
                raise Exception(f"The model loaded is not an instance of the {class_type.__name__} class.")

    def callback_graph(self, weights, obj_func_eval):
        self.objective_func_vals.append(obj_func_eval)
        self.weights.append(weights)
        # self.scores.append(self.score(X, y, True))
        print(f"Cost: {obj_func_eval}")
        # print(f"Score: {self.score(X, y, True)}")
        # print(f"Weights: {weights}")



df = pd.read_csv("./Admission_Predict.csv")

X = np.array(df.iloc[:,:-1])
y = np.array(df.iloc[:,-1])
X = PCA(n_components=7).fit_transform(X)
X, X_test, y, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
y.shape = (320, 1)
init = [0] * (7 * 7 + 7)

vqr = MultiTargetQuantumNNRegressor(7, 1, init, 'slsqp', maxiter=300)
vqr.fit(X, y, True)
print(f"Score: {vqr.score(X, y, True)}")
print(f"Objs func vals: {vqr.objective_func_vals}")

# X = np.array([[-0.32741112, -0.11288069,  0.49650164],
#        [-0.94268847, -0.78149813, -0.49440176],
#        [ 0.68523899,  0.61829019, -1.32935529],
#        [-1.25647971, -0.14910498, -0.25044557],
#        [ 1.66252391, -0.78480779,  1.79644309],
#        [ 0.42989295,  0.45376306,  0.21658276],
#        [-0.61965493, -0.39914738, -0.33494265],
#        [-0.54552144,  1.85889336,  0.67628493]])

# y = np.array([ -8.02307406, -23.10019118,  16.79149797, -30.78951577,
#         40.73946101,  10.53434892, -15.18438779, -13.3677773 ])
# y.shape = (8, 1)
    # print(vqr.weights)
# print(y)
# print(y.shape)
# print(f"Weights: {vqr.weights}")
# print(f"Predicted: {vqr.predict(X, True)}")

# print(f"True: {vqr.get_scaled_data(X, y)[1]}")