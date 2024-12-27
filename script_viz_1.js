class NeuralNetwork {
    constructor(inputSize, hiddenSize, outputSize) {
        // Network architecture parameters
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;

        // 1. Initialize input-to-hidden layer weights randomly
        this.inputHiddenWeights = this.initializeWeights(inputSize, hiddenSize);
        
        // 2. Initialize hidden-to-output layer weights randomly
        this.hiddenOutputWeights = this.initializeWeights(hiddenSize, outputSize);
    }

    // Utility method to initialize weights with small random values
    initializeWeights(rows, cols) {
        return Array.from({ length: rows }, () => 
            Array.from({ length: cols }, () => Math.random() * 2 - 1)
        );
    }

    // Convert string to one-hot encoded vector
    stringToOneHot(string, maxLength) {
        // Create a zero-filled array of fixed length
        const oneHot = new Array(maxLength).fill(0);
        
        // Fill one-hot vector with character encodings
        for (let i = 0; i < Math.min(string.length, maxLength); i++) {
            // Simple encoding: use character code
            oneHot[i] = string.charCodeAt(i) / 255;
        }
        
        return oneHot;
    }

    // Activation function: Sigmoid for non-linearity
    sigmoid(x) {
        let sigma =  1 / (1 + Math.exp(-x));
        if(isNaN(sigma)){ //true for any non number including NaN
          // debugger;
          sigma = 0; /// TODO: ZERO NaNs?
        }
        return sigma;
        // return 1 / (1 + Math.exp(-x));
    }

    // Derivative of sigmoid for backpropagation
    sigmoidDerivative(x) {
        return x * (1 - x);
    }

    // Forward propagation
    forward(input) {
        // 3. Compute hidden layer activations
        const hiddenLayer = this.inputHiddenWeights.map(row => {
            const sigma = row.reduce((sum, weight, j) => {
                // console.log('x',sum,weight,input[j])
                return sum + weight * input[j]
            } , 0);
            const sigmo = this.sigmoid(sigma);
            if(sigmo===undefined){
                // console.log('xxx')
                return sum + weight * 0.5;
            }
            return sigmo;
            // this.sigmoid(row.reduce((sum, weight, j) => sum + weight * input[j], 0))
        });
// debugger;
        // 4. Compute output layer activations 8 X 9
        const outputLayer = this.hiddenOutputWeights.map(row => {
            // console.log('row',row)
            const sigma = row.reduce((sum, weight, j) => {
                // console.log('x',sum,weight,hiddenLayer[j])
                if(hiddenLayer[j]===undefined){
                    // console.log('xxx')
                    return 0;//sum + weight * 0.5;
                }
                return sum + weight * hiddenLayer[j]
            }, 0);
            return this.sigmoid(sigma)
            // this.sigmoid(row.reduce((sum, weight, j) => sum + weight * hiddenLayer[j], 0))
        });

        return { hiddenLayer, outputLayer };
    }

    // Training method using backpropagation
    train(input, target, learningRate = 0.1) {
        // 5. Forward propagation
        const { hiddenLayer, outputLayer } = this.forward(input);

        // 6. Compute output layer error
        const outputError = outputLayer.map((output, i) => target[i] - output);
        const outputDelta = outputError.map((error, i) => 
            error * this.sigmoidDerivative(outputLayer[i])
        );

        // 7. Compute hidden layer error
        const hiddenError = this.hiddenOutputWeights.map((_, j) => 
            outputDelta.reduce((sum, delta, k) => sum + delta * this.hiddenOutputWeights[k][j], 0)
        );
        const hiddenDelta = hiddenError.map((error, i) => 
            error * this.sigmoidDerivative(hiddenLayer[i])
        );

        // 8. Update weights
        // Update hidden to output weights
        for (let k = 0; k < this.outputSize; k++) {
            for (let j = 0; j < this.hiddenSize; j++) {
                this.hiddenOutputWeights[k][j] += learningRate * outputDelta[k] * hiddenLayer[j];
            }
        }

        // Update input to hidden weights
        for (let j = 0; j < this.hiddenSize; j++) {
            for (let i = 0; i < this.inputSize; i++) {
                this.inputHiddenWeights[j][i] += learningRate * hiddenDelta[j] * input[i];
            }
        }
    }

    // Embed a string into a vector representation
    embedString(string, maxLength) {
        const oneHot = this.stringToOneHot(string, maxLength);
        return this.forward(oneHot).hiddenLayer;
    }

    // New method to find most similar token
    findMostSimilar(inputToken, tokens, maxLength) {
        const inputEmbedding = this.embedString(inputToken, maxLength);
        
        // Compute cosine similarity
        const similarities = tokens.map(token => {
            const tokenEmbedding = this.embedString(token, maxLength);
            return this.cosineSimilarity(inputEmbedding, tokenEmbedding);
        });

        // Find index of most similar token
        const mostSimilarIndex = similarities.reduce(
            (maxIndex, current, index, arr) => 
                current > arr[maxIndex] ? index : maxIndex, 0
        );

        return {
            token: tokens[mostSimilarIndex],
            similarity: similarities[mostSimilarIndex]
        };
    }

    // Compute cosine similarity between two vectors
    cosineSimilarity(vec1, vec2) {
        const dotProduct = vec1.reduce((sum, val, i) => sum + val * vec2[i], 0);
        const magnitude1 = Math.sqrt(vec1.reduce((sum, val) => sum + val * val, 0));
        const magnitude2 = Math.sqrt(vec2.reduce((sum, val) => sum + val * val, 0));
        return dotProduct / (magnitude1 * magnitude2);
    }
}


// Neural Network Visualization Techniques
// Understand network internals
// Debug learning process
// Gain insights into weight transformations


class NeuralNetVisualizer {
    constructor(neuralNet) {
        this.neuralNet = neuralNet;
    }

    // 1. Weight Heatmap Visualization
    visualizeWeights() {
        console.log("\n🔍 Weight Visualization 🔍");
        
        // Input to Hidden Layer Weights
        console.log("Input to Hidden Layer Weights:");
        this.printHeatmap(this.neuralNet.inputHiddenWeights, 
            "Input-Hidden", 
            (val) => this.colorMap(val, -1, 1)
        );

        console.log("\nHidden to Output Layer Weights:");
        // debugger;
        this.printHeatmap(this.neuralNet.hiddenOutputWeights, 
            "Hidden-Output", 
            (val) => this.colorMap(val, -1, 1)
        );
    }

    // Helper method to create color-coded console output
    colorMap(value, min, max) {
        // Map value to color intensity
        const normalized = (value - min) / (max - min);
        let r = Math.floor(255 * (1 - normalized));
        let b = Math.floor(255 * normalized);
        //TODO: cannot be negative?
        r = (r<0)?0:r;
        b = (b<0)?0:b;
        return `\x1b[48;2;${r};0;${b}m  \x1b[0m`;
    }

    // 2. Activation Potential Visualization
    visualizeActivations(input) {
        console.log("\n⚡ Activation Potential Visualization ⚡");
        
        // Perform forward propagation
debugger;
        const { hiddenLayer, outputLayer } = this.neuralNet.forward(input);

        console.log("Input Vector:", 
            input.map(val => val.toFixed(4))
        );
debugger;
        console.log("\nHidden Layer Activations:");
        hiddenLayer.forEach((activation, index) => {
            const barLength = Math.floor(activation * 20);
            const bar = "█".repeat(barLength);
            console.log(`Neuron ${index + 1}: ${activation.toFixed(4)} ${bar}`);
        });

        console.log("\nOutput Layer Activations:");
        outputLayer.forEach((activation, index) => {
            const barLength = Math.floor(activation * 20);
            const bar = "█".repeat(barLength);
            console.log(`Neuron ${index + 1}: ${activation.toFixed(4)} ${bar}`);
        });
    }

    // 3. Learning Trajectory Visualization
    visualizeLearningTrajectory(tokens, maxLength) {
        console.log("\n📈 Learning Trajectory Visualization 📈");
        
        // Track weight changes during training
        const weightTrajectory = {
            inputHidden: [],
            hiddenOutput: []
        };

        // Capture initial weights
        weightTrajectory.inputHidden.push(
            this.neuralNet.inputHiddenWeights.map(row => row.map(val => val))
        );
        weightTrajectory.hiddenOutput.push(
            this.neuralNet.hiddenOutputWeights.map(row => row.map(val => val))
        );

        // Training process with weight capture
        tokens.forEach((token, index) => {
            const input = this.neuralNet.stringToOneHot(token, maxLength);
            const target = new Array(tokens.length).fill(0);
            target[index] = 1;

            // Train and capture weights periodically
            for (let epoch = 0; epoch < 100; epoch++) {
                if (epoch % 20 === 0) {
                    this.neuralNet.train(input, target, 0.1);
                    
                    // Capture current weights
                    weightTrajectory.inputHidden.push(
                        this.neuralNet.inputHiddenWeights.map(row => row.map(val => val))
                    );
                    weightTrajectory.hiddenOutput.push(
                        this.neuralNet.hiddenOutputWeights.map(row => row.map(val => val))
                    );
                }
            }
        });

        // Summarize weight changes
        console.log("Weight Change Summary:");
        console.log("Input-Hidden Layer Initial vs Final Weights Range:");
        this.summarizeWeightChanges(weightTrajectory.inputHidden);
        
        console.log("\nHidden-Output Layer Initial vs Final Weights Range:");
        // debugger;
        this.summarizeWeightChanges(weightTrajectory.hiddenOutput);
    }

    // Helper to summarize weight changes
    summarizeWeightChanges(weightHistory) {
        const initialWeights = weightHistory[0];
        const finalWeights = weightHistory[weightHistory.length - 1];

        const weightChanges = initialWeights.map((row, i) => 
            row.map((val, j) => Math.abs(val - finalWeights[i][j]))
        );

        const maxChange = Math.max(...weightChanges.flat());
        const avgChange = weightChanges.flat()
            .reduce((sum, val) => sum + val, 0) / weightChanges.flat().length;

        console.log(`Maximum Weight Change: ${maxChange.toFixed(4)}`);
        console.log(`Average Weight Change: ${avgChange.toFixed(4)}`);
    }

    // 4. Detailed Weight Heatmap with Intensity
    printHeatmap(weights, label, colorFunc) {
        console.log(`${label} Weight Heatmap:`);
        weights.forEach((row, i) => {
            const rowVisualization = row.map((val, j) => 
                colorFunc(val)
            ).join('');
            console.log(`Neuron ${i + 1}: ${rowVisualization}`);
        });
    }
}

// Demonstration Function
function demonstrateNeuralNetVisualization() {
    // Tokens for demonstration
    const tokens = ["hello", "help", "world", "code", "coding","aaa"];
    const maxLength = 6;

    // Create neural network
    const nn = new NeuralNetwork(
        maxLength,    // input size
        6,            // hidden layer size
        tokens.length // output layer size
    );

    // Create visualizer
    const visualizer = new NeuralNetVisualizer(nn);
// debugger;
    // 1. Initial Weight Visualization (Before Training)
    console.log("🔬 VIZ: Neural Network State 🔬");
    visualizer.visualizeWeights();

    // 2. Training and Visualization
    tokens.forEach((token, index) => {
        const input = nn.stringToOneHot(token, maxLength);
        const target = new Array(tokens.length).fill(0);
        target[index] = 1;

        // Train network
        for (let epoch = 0; epoch < 200; epoch++) {
            nn.train(input, target, 0.05);
        }
    });

    // 3. Visualization after Training
    console.log("\n🦾 VIZ:WEIGHTS 🦾");
    visualizer.visualizeWeights();

    // 4. Activation Visualization
    console.log("\n⚡ VIZ:ACTIVIATION ⚡");
    // const sampleInput = nn.stringToOneHot("hello", maxLength);
    // const sampleInput = nn.stringToOneHot("aaa", maxLength);
    // const sampleInput = nn.stringToOneHot("xxx", maxLength);
    // const sampleInput = nn.stringToOneHot("abc", maxLength);
    // const sampleInput = nn.stringToOneHot("def", maxLength);
    const sampleInput = nn.stringToOneHot("world", maxLength);
    visualizer.visualizeActivations(sampleInput);

    // 5. Learning Trajectory
    console.log("\n🚀 VIZ:TRAJECTORY 🚀");
    visualizer.visualizeLearningTrajectory(tokens, maxLength);
}
// debugger;
// Run the visualization demonstration
demonstrateNeuralNetVisualization();