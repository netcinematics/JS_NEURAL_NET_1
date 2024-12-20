class NeuralNetwork {
    constructor(inputSize,learnRate) {
        // Making square matrices (equal dimensions)
        this.inputSize = inputSize;
        this.hiddenSize = inputSize;
        this.outputSize = 2;  // Two outputs: starts with 'a' and contains uppercase

        // Initialize weights with improved initial values for better convergence
        this.weightsIH = this.initializeWeights(this.inputSize, this.hiddenSize, 0.3);
        this.weightsHO = this.initializeWeights(this.hiddenSize, this.outputSize, 0.3);
        
        // Initialize biases with small positive values
        this.biasH = new Array(this.hiddenSize).fill(0.1);
        this.biasO = new Array(this.outputSize).fill(0.1);
        
        console.log('LEARN RATE',learnRate)
        // Learning parameters
        this.learningRate = (learnRate)?learnRate:0.15;  // Tuned for better convergence
    }

    // Initialize weights with Xavier initialization
    initializeWeights(rows, cols, scale) {
        const weights = [];
        for (let i = 0; i < rows; i++) {
            weights[i] = [];
            for (let j = 0; j < cols; j++) {
                // Xavier initialization: variance of weights based on layer size
                weights[i][j] = (Math.random() * 2 - 1) * Math.sqrt(scale / rows);
            }
        }
        return weights;
    }

    // Sigmoid activation function
    sigmoid(x) {
        return 1 / (1 + Math.exp(-x));
    }

    // Derivative of sigmoid for backpropagation
    sigmoidDerivative(x) {
        return x * (1 - x);
    }

    // Forward pass through the network
    forward(input) {
        // Hidden layer computations
        this.hiddenLayer = new Array(this.hiddenSize).fill(0);
        for (let i = 0; i < this.hiddenSize; i++) {
            let sum = this.biasH[i];
            for (let j = 0; j < this.inputSize; j++) {
                sum += input[j] * this.weightsIH[j][i];
            }
            this.hiddenLayer[i] = this.sigmoid(sum);
        }

        // Output layer computations
        this.outputLayer = new Array(this.outputSize).fill(0);
        for (let i = 0; i < this.outputSize; i++) {
            let sum = this.biasO[i];
            for (let j = 0; j < this.hiddenSize; j++) {
                sum += this.hiddenLayer[j] * this.weightsHO[j][i];
            }
            this.outputLayer[i] = this.sigmoid(sum);
        }

        return this.outputLayer;
    }

    // Backward pass for training
    backward(input, target, output) {
        // Output layer error
        const outputErrors = new Array(this.outputSize);
        for (let i = 0; i < this.outputSize; i++) {
            outputErrors[i] = (target[i] - output[i]) * this.sigmoidDerivative(output[i]);
        }

        // Hidden layer error
        const hiddenErrors = new Array(this.hiddenSize).fill(0);
        for (let i = 0; i < this.hiddenSize; i++) {
            for (let j = 0; j < this.outputSize; j++) {
                hiddenErrors[i] += outputErrors[j] * this.weightsHO[i][j];
            }
            hiddenErrors[i] *= this.sigmoidDerivative(this.hiddenLayer[i]);
        }

        // Update weights and biases
        // Hidden to output weights
        for (let i = 0; i < this.hiddenSize; i++) {
            for (let j = 0; j < this.outputSize; j++) {
                this.weightsHO[i][j] += this.learningRate * outputErrors[j] * this.hiddenLayer[i];
            }
        }

        // Input to hidden weights
        for (let i = 0; i < this.inputSize; i++) {
            for (let j = 0; j < this.hiddenSize; j++) {
                this.weightsIH[i][j] += this.learningRate * hiddenErrors[j] * input[i];
            }
        }

        // Update biases
        for (let i = 0; i < this.outputSize; i++) {
            this.biasO[i] += this.learningRate * outputErrors[i];
        }
        for (let i = 0; i < this.hiddenSize; i++) {
            this.biasH[i] += this.learningRate * hiddenErrors[i];
        }
    }

    // Training function
    train(inputs, targets, epochs) {
        const errors = [];
        for (let epoch = 0; epoch < epochs; epoch++) {
            let epochError = 0;
            for (let i = 0; i < inputs.length; i++) {
                const output = this.forward(inputs[i]);
                this.backward(inputs[i], targets[i], output);
                
                // Calculate mean squared error
                for (let j = 0; j < this.outputSize; j++) {
                    epochError += Math.pow(targets[i][j] - output[j], 2);
                }
            }
            epochError /= inputs.length;
            errors.push(epochError);
            
            // Early stopping if error is small enough
            if (epochError < 0.001) break;
        }
        return errors; //TODO: visualize 1000 returns, to show TRAINING MOVIE.
        //TODO: then use BACKTRACE to
    }

    // Visualize network state
    visualize() {
        console.log("\nNeural Network State:");
        console.log("Input -> Hidden Weights:");
        console.log(this.weightsIH.map(row => row.map(w => w.toFixed(3))));
        console.log("\nHidden -> Output Weights:");
        console.log(this.weightsHO.map(row => row.map(w => w.toFixed(3))));
        console.log("\nHidden Biases:", this.biasH.map(b => b.toFixed(3)));
        console.log("Output Biases:", this.biasO.map(b => b.toFixed(3)));
    }
}

// Test the neural network
function testNetwork() {
    debugger;
    // Convert word to input vector (using ASCII values normalized)
    function wordToInput(word, size) {
        const input = new Array(size).fill(0);
        for (let i = 0; i < Math.min(word.length, size); i++) {
            input[i] = word.charCodeAt(i) / 255;  // Normalize ASCII values
        }
        return input;
    }

    // Create training data
    const trainingWords = [
        "apple", "Banana", "cat", "Dog", "elephant",
        "art", "Book", "car", "Door", "egg",
        "another", "CAPS", "simple", "Test", "animal"
    ];

    const learnRate = 0.1544;
    const epoch_NUM = 10000;
    const inputSize = 10;  // Fixed input size
    const nn = new NeuralNetwork(inputSize, learnRate);

    // Prepare training data
    const inputs = trainingWords.map(word => wordToInput(word, inputSize));
    const targets = trainingWords.map(word => [
        word.startsWith('a') ? 1 : 0,
        /[A-Z]/.test(word) ? 1 : 0
    ]); //TODO extend with count>3 and ends with little a.

    // Train the network
    console.log("Training network...");
    const errors = nn.train(inputs, targets, epoch_NUM);
    console.log("Final error:", errors[errors.length - 1]);

    // Test the network
    const testWords = ["amazing", "and", "zebra", "Apple", "test"];
    console.log("\nTesting network:");
    for (const word of testWords) {
        const input = wordToInput(word, inputSize);
        const output = nn.forward(input);
        console.log(`\nWord: ${word}`);
        console.log(`Starts with 'a': ${output[0].toFixed(3)} (Expected: ${word.startsWith('a') ? 1 : 0})`);
        console.log(`Contains uppercase: ${output[1].toFixed(3)} (Expected: ${/[A-Z]/.test(word) ? 1 : 0})`);
    }

    // Visualize the final state
    // nn.visualize();
}

// Run the test
testNetwork();