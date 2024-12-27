class NeuralNetwork {
    constructor(inputSize, activationType = 'relu') {
        // Making square matrices (equal dimensions)
        this.inputSize = inputSize;
        this.hiddenSize = inputSize;
        this.outputSize = 2;  // Two outputs: starts with 'a' and contains uppercase
        this.activationType = activationType;

        // Initialize weights - Modified for ReLU (He initialization)
        const initScale = activationType === 'relu' ? 2.0 : 0.3;
        this.weightsIH = this.initializeWeights(this.inputSize, this.hiddenSize, initScale);
        this.weightsHO = this.initializeWeights(this.hiddenSize, this.outputSize, initScale);
        
        // Initialize biases - Slightly larger for ReLU to prevent dead neurons
        const biasInit = activationType === 'relu' ? 0.2 : 0.1;
        this.biasH = new Array(this.hiddenSize).fill(biasInit);
        this.biasO = new Array(this.outputSize).fill(biasInit);
        
        // Learning parameters - Adjusted for ReLU
        this.learningRate = activationType === 'relu' ? 0.01 : 0.15;  // Lower learning rate for ReLU
    }

    // Initialize weights with He initialization for ReLU
    initializeWeights(rows, cols, scale) {
        const weights = [];
        for (let i = 0; i < rows; i++) {
            weights[i] = [];
            for (let j = 0; j < cols; j++) {
                // He initialization for ReLU, Xavier for sigmoid
                const stddev = Math.sqrt(scale / rows);
                weights[i][j] = (Math.random() * 2 - 1) * stddev;
            }
        }
        return weights;
    }

    // Activation functions //TODO
    activate(x) {
        if (this.activationType === 'relu') {
            return Math.max(0, x); //TODO
        } else {
            return 1 / (1 + Math.exp(-x));  // sigmoid
        }
    }

    // Derivatives of activation functions
    activateDerivative(x) {
        if (this.activationType === 'relu') {
            return x > 0 ? 1 : 0;
        } else {
            return x * (1 - x);  // sigmoid derivative
        }
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
            this.hiddenLayer[i] = this.activate(sum);
        }

        // Output layer computations - Always use sigmoid for final layer
        this.outputLayer = new Array(this.outputSize).fill(0);
        for (let i = 0; i < this.outputSize; i++) {
            let sum = this.biasO[i];
            for (let j = 0; j < this.hiddenSize; j++) {
                sum += this.hiddenLayer[j] * this.weightsHO[j][i];
            }
            // Use sigmoid for output layer regardless of hidden layer activation
            this.outputLayer[i] = 1 / (1 + Math.exp(-sum));
        }

        return this.outputLayer;
    }

    // Backward pass for training
    backward(input, target, output) {
        // Output layer error (always use sigmoid derivative for output layer)
        const outputErrors = new Array(this.outputSize);
        for (let i = 0; i < this.outputSize; i++) {
            outputErrors[i] = (target[i] - output[i]) * (output[i] * (1 - output[i]));
        }

        // Hidden layer error
        const hiddenErrors = new Array(this.hiddenSize).fill(0);
        for (let i = 0; i < this.hiddenSize; i++) {
            for (let j = 0; j < this.outputSize; j++) {
                hiddenErrors[i] += outputErrors[j] * this.weightsHO[i][j];
            }
            hiddenErrors[i] *= this.activateDerivative(this.hiddenLayer[i]);
        }

        // Update weights and biases with gradient clipping
        const clipValue = 1.0;  // Prevent exploding gradients
        
        // Hidden to output weights
        for (let i = 0; i < this.hiddenSize; i++) {
            for (let j = 0; j < this.outputSize; j++) {
                let gradient = this.learningRate * outputErrors[j] * this.hiddenLayer[i];
                gradient = Math.max(Math.min(gradient, clipValue), -clipValue);
                this.weightsHO[i][j] += gradient;
            }
        }

        // Input to hidden weights
        for (let i = 0; i < this.inputSize; i++) {
            for (let j = 0; j < this.hiddenSize; j++) {
                let gradient = this.learningRate * hiddenErrors[j] * input[i];
                gradient = Math.max(Math.min(gradient, clipValue), -clipValue);
                this.weightsIH[i][j] += gradient;
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

    // Training function with improved error tracking
    train(inputs, targets, epochs) {
        const errors = [];
        let bestError = Infinity;
        let epochsWithoutImprovement = 0;
        
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
            
            // Early stopping with patience
            if (epochError < bestError) {
                bestError = epochError;
                epochsWithoutImprovement = 0;
            } else {
                epochsWithoutImprovement++;
            }
            
            // Stop if error is very small or no improvement for many epochs
            if (epochError < 0.001 || epochsWithoutImprovement > 50) {
                console.log(`Stopped at epoch ${epoch} with error ${epochError}`);
                break;
            }
        }
        return errors;
    }

    // Calculate accuracy on test set
    calculateAccuracy(testInputs, testTargets) {
        let correct = 0;
        let total = testInputs.length * this.outputSize;
        
        for (let i = 0; i < testInputs.length; i++) {
            const output = this.forward(testInputs[i]);
            for (let j = 0; j < this.outputSize; j++) {
                // Consider prediction correct if it's within 0.2 of target
                if (Math.abs(Math.round(output[j]) - testTargets[i][j]) < 0.2) {
                    correct++;
                }
            }
        }
        return (correct / total) * 100;
    }

    // Visualize network state
    visualize() {
        console.log(`\nNeural Network State (${this.activationType} activation):`);
        console.log("Input -> Hidden Weights (sample):");
        console.log(this.weightsIH.slice(0, 3).map(row => row.slice(0, 3).map(w => w.toFixed(3))));
        console.log("\nHidden -> Output Weights (sample):");
        console.log(this.weightsHO.slice(0, 3).map(row => row.map(w => w.toFixed(3))));
        console.log("\nHidden Biases (sample):", this.biasH.slice(0, 3).map(b => b.toFixed(3)));
        console.log("Output Biases:", this.biasO.map(b => b.toFixed(3)));
    }
}

// Test both networks
function compareNetworks() {

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
        "another", "CAPS", "simple", "Test", "animal",
        "architect", "BLUE", "yellow", "Green", "purple"
    ];

    const testWords = [
        "amazing", "and", "zebra", "Apple", "test",
        "alphabet", "MONKEY", "orange", "Bird", "fish"
    ];

    const inputSize = 10;
    
    // Create both networks
    const reluNet = new NeuralNetwork(inputSize, 'relu');
    const sigmoidNet = new NeuralNetwork(inputSize, 'sigmoid');

    // Prepare data
    const trainInputs = trainingWords.map(word => wordToInput(word, inputSize));
    const trainTargets = trainingWords.map(word => [
        word.startsWith('a') ? 1 : 0,
        /[A-Z]/.test(word) ? 1 : 0
    ]);

    const testInputs = testWords.map(word => wordToInput(word, inputSize));
    const testTargets = testWords.map(word => [
        word.startsWith('a') ? 1 : 0,
        /[A-Z]/.test(word) ? 1 : 0
    ]);

    // Train both networks
    console.log("\nTraining ReLU network...");
    const reluErrors = reluNet.train(trainInputs, trainTargets, 1000);
    console.log("Final ReLU error:", reluErrors[reluErrors.length - 1]);
    
    console.log("\nTraining Sigmoid network...");
    const sigmoidErrors = sigmoidNet.train(trainInputs, trainTargets, 1000);
    console.log("Final Sigmoid error:", sigmoidErrors[sigmoidErrors.length - 1]);

    // Calculate and compare accuracies
    const reluAccuracy = reluNet.calculateAccuracy(testInputs, testTargets);
    const sigmoidAccuracy = sigmoidNet.calculateAccuracy(testInputs, testTargets);

    console.log("\nAccuracy Comparison:");
    console.log(`ReLU Network: ${reluAccuracy.toFixed(2)}%`);
    console.log(`Sigmoid Network: ${sigmoidAccuracy.toFixed(2)}%`);

    // Test both networks on a few examples
    console.log("\nDetailed comparison on test words:");
    for (const word of testWords.slice(0, 5)) {  // Test first 5 words
        const input = wordToInput(word, inputSize);
        const reluOutput = reluNet.forward(input);
        const sigmoidOutput = sigmoidNet.forward(input);
        
        console.log(`\nWord: ${word}`);
        console.log("ReLU Network:");
        console.log(`  Starts with 'a': ${reluOutput[0].toFixed(3)} (Expected: ${word.startsWith('a') ? 1 : 0})`);
        console.log(`  Contains uppercase: ${reluOutput[1].toFixed(3)} (Expected: ${/[A-Z]/.test(word) ? 1 : 0})`);
        console.log("Sigmoid Network:");
        console.log(`  Starts with 'a': ${sigmoidOutput[0].toFixed(3)} (Expected: ${word.startsWith('a') ? 1 : 0})`);
        console.log(`  Contains uppercase: ${sigmoidOutput[1].toFixed(3)} (Expected: ${/[A-Z]/.test(word) ? 1 : 0})`);
    }
}

// Run the comparison
compareNetworks();