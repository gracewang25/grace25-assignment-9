import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for animations
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from matplotlib.animation import FuncAnimation
import os
from functools import partial
from matplotlib.patches import Circle

# Define result directory
RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)

# Define the MLP class
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, activation='tanh'):
        np.random.seed(0)
        self.lr = lr
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.activation_fn = activation

        # Initialize weights and biases
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2 / (input_dim + hidden_dim))
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2 / (hidden_dim + output_dim))
        self.b2 = np.zeros((1, output_dim))

    def _activation(self, Z):
        if self.activation_fn == 'tanh':
            return np.tanh(Z)
        if self.activation_fn == 'relu':
            return np.maximum(0, Z)
        if self.activation_fn == 'sigmoid':
            return 1 / (1 + np.exp(-Z))
        raise ValueError(f"Unsupported activation function: {self.activation_fn}")

    def _activation_derivative(self, Z):
        if self.activation_fn == 'tanh':
            return 1 - np.tanh(Z) ** 2
        if self.activation_fn == 'relu':
            return np.where(Z > 0, 1, 0)
        if self.activation_fn == 'sigmoid':
            sig = 1 / (1 + np.exp(-Z))
            return sig * (1 - sig)
        raise ValueError(f"Unsupported activation function: {self.activation_fn}")

    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self._activation(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = np.tanh(self.Z2)
        return self.A2

    def compute_loss(self, y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2) / 2

    def backward(self, X, y):
        m = y.shape[0]

        dA2 = (self.A2 - y) / m
        dZ2 = dA2 * (1 - self.A2 ** 2)
        dW2 = np.dot(self.A1.T, dZ2)
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self._activation_derivative(self.Z1)
        dW1 = np.dot(X.T, dZ1)
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

        self.dW1, self.dW2 = dW1, dW2

def generate_data(samples=100):
    np.random.seed(0)
    X = np.random.randn(samples, 2)
    y = ((X[:, 0] ** 2 + X[:, 1] ** 2) > 1).astype(int) * 2 - 1
    return X, y.reshape(-1, 1)

def update(frame, mlp, ax_input, ax_hidden, ax_gradient, X, y):
    ax_input.clear()
    ax_hidden.clear()
    ax_gradient.clear()

    for _ in range(10):
        y_pred = mlp.forward(X)
        mlp.backward(X, y)

    # Hidden layer scatter plot
    hidden_features = mlp.A1
    ax_hidden.scatter(
        hidden_features[:, 0],
        hidden_features[:, 1],
        hidden_features[:, 2],
        c=y.ravel(),
        cmap='coolwarm',
        alpha=0.7
    )

    # Add decision hyperplane in the hidden space
    xx, yy = np.meshgrid(
        np.linspace(hidden_features[:, 0].min(), hidden_features[:, 0].max(), 30),
        np.linspace(hidden_features[:, 1].min(), hidden_features[:, 1].max(), 30)
    )
    zz = (-mlp.W2[0, 0] * xx - mlp.W2[1, 0] * yy - mlp.b2[0, 0]) / (mlp.W2[2, 0] + 1e-6)
    ax_hidden.plot_surface(
        xx,
        yy,
        zz,
        color='cyan',
        alpha=0.3
    )

    # Add boundary (convex hull) around the hidden features
    if hidden_features.shape[0] > 3:
        hull = ConvexHull(hidden_features[:, :2])
        for simplex in hull.simplices:
            ax_hidden.plot3D(
                hidden_features[simplex, 0],
                hidden_features[simplex, 1],
                [hidden_features[:, 2].min()] * len(simplex),
                'r-'
            )
    ax_hidden.set_title(f'Hidden Layer Feature Space at Step {frame * 10}', fontsize=14)
    ax_hidden.set_xlabel('Hidden Unit 1')
    ax_hidden.set_ylabel('Hidden Unit 2')
    ax_hidden.set_zlabel('Hidden Unit 3')

    # Input layer decision boundary
    xx, yy = np.meshgrid(
        np.linspace(X[:, 0].min(), X[:, 0].max(), 100),
        np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
    )
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = mlp.forward(grid_points).reshape(xx.shape)

    ax_input.contourf(
        xx,
        yy,
        Z,
        levels=[-1, 0, 1],
        alpha=0.3,
        colors=['#89CFF0', '#FFB6C1']
    )
    ax_input.contour(xx, yy, Z, levels=[0], colors='black')
    ax_input.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='coolwarm', edgecolors='k')
    ax_input.set_title(f'Input Space Decision Boundary at Step {frame * 10}', fontsize=14)
    ax_input.set_xlabel('X1')
    ax_input.set_ylabel('X2')

    # Visualization of gradients with thicker lines and node labels
    node_positions = visualize_network(mlp, ax_gradient)
    draw_network_gradients(mlp, node_positions, ax_gradient, scaling_factor=200)
    ax_gradient.set_title(f'Network Gradients Visualization at Step {frame * 10}', fontsize=14)


def visualize_network(mlp, ax):
    node_positions = {}
    layers = [2, 3, 1]
    x_coords = [0.1, 0.5, 0.9]

    for i, size in enumerate(layers):
        y_coords = np.linspace(0.1, 0.9, size)
        for j, y in enumerate(y_coords):
            node_positions[(i, j)] = (x_coords[i], y)
            label = f"$x_{j+1}$" if i == 0 else f"$h_{j+1}$" if i == 1 else "$y$"
            circle = Circle((x_coords[i], y), 0.035, color='#2A9D8F', zorder=10)
            ax.add_patch(circle)
            ax.text(x_coords[i], y, label, fontsize=10, ha='center', va='center', zorder=15)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    return node_positions

def draw_network_gradients(mlp, positions, ax, scaling_factor=100):
    for i in range(2):  # Input to hidden layer connections
        for j in range(3):  # Hidden neurons
            start = positions[(0, i)]
            end = positions[(1, j)]
            linewidth = np.abs(mlp.dW1[i, j]) * scaling_factor
            ax.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                lw=linewidth,
                color='#E76F51'
            )

    for i in range(3):  # Hidden to output layer connections
        start = positions[(1, i)]
        end = positions[(2, 0)]  # Single output neuron
        linewidth = np.abs(mlp.dW2[i, 0]) * scaling_factor
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            lw=linewidth,
            color='#F4A261'
        )

def visualize(activation, lr, steps):
    X, y = generate_data()
    mlp = MLP(2, 3, 1, lr, activation)

    fig = plt.figure(figsize=(21, 7))
    ax_hidden = fig.add_subplot(131, projection='3d')
    ax_input = fig.add_subplot(132)
    ax_gradient = fig.add_subplot(133)

    ani = FuncAnimation(fig, partial(update, mlp=mlp, ax_input=ax_input, ax_hidden=ax_hidden, ax_gradient=ax_gradient, X=X, y=y), frames=steps // 10, repeat=False)
    ani.save(os.path.join(RESULT_DIR, "visualize.gif"), writer='pillow', fps=10)
    plt.close()

if __name__ == "__main__":
    visualize("tanh", 0.1, 1000)