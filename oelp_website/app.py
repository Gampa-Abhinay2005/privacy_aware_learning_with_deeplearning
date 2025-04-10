import matplotlib
matplotlib.use('Agg')  # Set Matplotlib backend to Agg
from flask import Flask, render_template, request
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam, SGD
import matplotlib.pyplot as plt
import io
import base64
import numpy as np

app = Flask(__name__)

# Dummy data generation function (replace with actual dataset)
def generate_dummy_data():
    # Generate dummy dataset for testing
    X_train = np.random.rand(100, 8)  # 100 samples, 8 features
    X_test = np.random.rand(20, 8)  # 20 test samples
    y_train = np.random.randint(2, size=(100, 2))  # Binary classification (2 classes)
    y_test = np.random.randint(2, size=(20, 2))  # Binary classification (2 classes)
    return X_train, y_train, X_test, y_test

# Function to train the model
def train_model(optimizer_name, optimizer, input_shape, num_hidden_layers, neurons_per_layer, output_shape, x_train, y_train, x_test, y_test):
    model = Sequential()
    model.add(Dense(neurons_per_layer[0], input_dim=input_shape, activation='relu'))
    
    for i in range(1, num_hidden_layers):
        model.add(Dense(neurons_per_layer[i], activation='relu'))
    
    model.add(Dense(output_shape, activation='softmax'))
    
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=32)
    return {'model': model, 'history': history}

# Function to plot accuracy graph
def plot_accuracy(history, optimizer_name):
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'Accuracy with {optimizer_name}')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Save plot to a BytesIO object and encode it as base64
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf-8')
    plt.close()
    
    return f"data:image/png;base64,{plot_url}"

# Function to plot error with noise
def plot_error_with_noise(model, x_test, y_test, noise_type):
    mse_values = []
    noise_levels = np.linspace(0, 1, 10)  # Range from 0 to 1
    
    for level in noise_levels:
        # Add noise (Gaussian or Laplace) to the test data
        if noise_type == 'gaussian':
            noise = np.random.normal(0, level, x_test.shape)
        elif noise_type == 'laplace':
            noise = np.random.laplace(0, level, x_test.shape)
        else:
            noise = np.zeros_like(x_test)

        # Apply the noise to the test set
        x_test_noisy = x_test + noise
        
        # Evaluate model on noisy data
        predictions = model.predict(x_test_noisy)
        mse = mean_squared_error(y_test, predictions)
        mse_values.append(mse)
    
    # Plot MSE vs. Noise Level
    plt.plot(noise_levels, mse_values, label='MSE vs. Noise Level')
    plt.title(f'Mean Squared Error with {noise_type} Noise')
    plt.xlabel('Noise Level')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    
    # Save plot to a BytesIO object and encode it as base64
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf-8')
    plt.close()
    
    return f"data:image/png;base64,{plot_url}"
# Function to plot validation error comparison between Adam and SGD optimizers
def plot_optimizer_comparison_error(history_adam, history_sgd):
    val_loss_adam = history_adam.history['val_loss']
    val_loss_sgd = history_sgd.history['val_loss']
    epochs = range(1, len(val_loss_adam) + 1)
    
    plt.plot(epochs, val_loss_adam, label='Adam Validation Loss')
    plt.plot(epochs, val_loss_sgd, label='SGD Validation Loss')
    plt.title('Validation Loss Comparison: Adam vs SGD')
    plt.xlabel('Epochs')
    plt.ylabel('Validation Loss')
    plt.legend()
    
    # Save plot to a BytesIO object and encode it as base64
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf-8')
    plt.close()
    
    return f"data:image/png;base64,{plot_url}"

@app.route('/', methods=['GET', 'POST'])
def index():
    try:
        # Get form data
        optimizer_choice = request.form.get('optimizer', default=None)
        noise_type = request.form.get('noise_type', default=None)
        
        # Get number of hidden layers and neurons per layer from the form
        num_hidden_layers = request.form.get('num_hidden_layers', default=1, type=int)
        neurons_per_layer = []
        for i in range(num_hidden_layers):
            neurons_per_layer.append(request.form.get(f'neurons_layer_{i}', default=32, type=int))
        
        # Dummy training data
        input_shape = 8  # Example input shape
        output_shape = 2  # Example output shape
        x_train, y_train, x_test, y_test = generate_dummy_data()
        
        # Initialize plot URLs
        plot_url_acc_adam = None
        plot_url_acc_sgd = None
        plot_url_acc_both = None
        plot_url_noise = None
        plot_url_optimizer_comparison = None  # For the optimizer comparison graph

        # Train model with Adam optimizer
        if optimizer_choice == '1':  # Adam selected
            result_adam = train_model('Adam', Adam(), input_shape, num_hidden_layers, neurons_per_layer, output_shape, x_train, y_train, x_test, y_test)
            plot_url_acc_adam = plot_accuracy(result_adam['history'], 'Adam Optimizer')

        # Train model with SGD optimizer
        elif optimizer_choice == '2':  # SGD selected
            result_sgd = train_model('SGD', SGD(), input_shape, num_hidden_layers, neurons_per_layer, output_shape, x_train, y_train, x_test, y_test)
            plot_url_acc_sgd = plot_accuracy(result_sgd['history'], 'SGD Optimizer')

        # Train both models with Adam and SGD optimizers
        elif optimizer_choice == '3':  # Both selected
            result_adam = train_model('Adam', Adam(), input_shape, num_hidden_layers, neurons_per_layer, output_shape, x_train, y_train, x_test, y_test)
            result_sgd = train_model('SGD', SGD(), input_shape, num_hidden_layers, neurons_per_layer, output_shape, x_train, y_train, x_test, y_test)
             # Generate accuracy plots for both Adam and SGD
            plot_url_acc_adam = plot_accuracy(result_adam['history'], 'Adam Optimizer')
            plot_url_acc_sgd = plot_accuracy(result_sgd['history'], 'SGD Optimizer')
            
            # Plot the validation error comparison for both optimizers
            plot_url_optimizer_comparison = plot_optimizer_comparison_error(result_adam['history'], result_sgd['history'])
            plot_url_acc_both = plot_accuracy(result_adam['history'], 'Adam Optimizer') + " and " + plot_accuracy(result_sgd['history'], 'SGD Optimizer')
            # Plot the validation error comparison for both optimizers
            plot_url_optimizer_comparison = plot_optimizer_comparison_error(result_adam['history'], result_sgd['history'])

        # Error plot if noise is selected
        if noise_type and optimizer_choice != '3':
            if optimizer_choice == '1':
                plot_url_noise = plot_error_with_noise(result_adam['model'], x_test, y_test, noise_type)
            elif optimizer_choice == '2':
                plot_url_noise = plot_error_with_noise(result_sgd['model'], x_test, y_test, noise_type)

        return render_template("index.html", 
                               num_hidden_layers=num_hidden_layers, 
                               neurons_per_layer=neurons_per_layer,
                               optimizer_choice=optimizer_choice, 
                               noise_type=noise_type,
                               plot_url_acc_adam=plot_url_acc_adam,
                               plot_url_acc_sgd=plot_url_acc_sgd,
                               plot_url_acc_both=plot_url_acc_both,
                               plot_url_noise=plot_url_noise,
                               plot_url_optimizer_comparison=plot_url_optimizer_comparison)

    except Exception as e:
        # Handle the exception by showing the error message
        return render_template("index.html", error_message=str(e), num_hidden_layers=1, neurons_per_layer=[], optimizer_choice=None, noise_type=None)

if __name__ == '__main__':
    app.run(debug=True)
