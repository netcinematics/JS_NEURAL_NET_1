// Minimal Neural Network for String Token Embedding and Processing

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
        return 1 / (1 + Math.exp(-x));
    }

    // Derivative of sigmoid for backpropagation
    sigmoidDerivative(x) {
        return x * (1 - x);
    }

    // Forward propagation
    forward(input) {
        // 3. Compute hidden layer activations
        // debugger;
        const hiddenLayer = this.inputHiddenWeights.map(row => {
            const sigma = row.reduce((sum, weight, j) => {
                // console.log('x',sum,weight,input[j])
                return sum + weight * input[j]
            } , 0);
            const sigmo = this.sigmoid(sigma);
            if(sigmo===undefined){
                // console.log('xxx')
                return sum + weight * 0.5; //TODO training 
            }
            return sigmo;
            // this.sigmoid(row.reduce((sum, weight, j) => sum + weight * input[j], 0))
        });
// debugger;
        // 4. Compute output layer activations 8 X 9
        const outputLayer = this.hiddenOutputWeights.map(row => {
            const sigma = row.reduce((sum, weight, j) => {
                // console.log('x',sum,weight,hiddenLayer[j])
                if(hiddenLayer[j]===undefined){
                    return 0;//sum + weight * 0.5;
                }
                return sum + weight * hiddenLayer[j]
            }, 0);
            return this.sigmoid(sigma)
            // this.sigmoid(row.reduce((sum, weight, j) => sum + weight * hiddenLayer[j], 0))
        });
// debugger;
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

// Comprehensive Demonstration Function
function demonstrateNeuralNetCapabilities() {
    console.log("🧠 Neural Network String Token Processing Demonstration 🧠");
    
    // Diverse set of tokens with semantic relationships
    const tokens = [
        "hello", "help", "hi", 
        "world", "word", "words",
        "code", "coding", //"coder" //get 8 by 8
    ];
    const maxLength = 8;  // Maximum token length

    // Create neural network
    const nn = new NeuralNetwork(
        maxLength,        // input size (max string length)
        8,                // hidden layer size (more neurons for richer representation)
        tokens.length     // output layer size (number of tokens)
    );

    // Training phase
    console.log("\n--- Training Phase ---");
    // debugger;//1
    tokens.forEach((token, index) => {
        const input = nn.stringToOneHot(token, maxLength);
        const target = new Array(tokens.length).fill(0);
        target[index] = 1;  // One-hot target for each token

        // Train for multiple epochs to ensure learning
        for (let epoch = 0; epoch < 2000; epoch++) { //TODO EPOCH 2000
            nn.train(input, target, 0.005);  //TODO LEARN RATE 0.05// Slightly reduced learning rate
        }
    });
    console.log("\n--- Training: DONE ---");
// debugger;
    // Similarity and Embedding Demonstration
    console.log("\n--- Token Embedding Similarities ---");
    const testTokens = ["hello", "help", "world", "xxx"];
    
    testTokens.forEach(token => {
        console.log(`\nAnalyzing token: "${token}"`);
        
        // Find embedding
        const embedding = nn.embedString(token, maxLength);
        console.log("Embedding vector:", 
            embedding.map(val => val.toFixed(4))
        );

        // Find most similar token
        const similar = nn.findMostSimilar(token, tokens, maxLength);
        console.log(`Most similar token: "${similar.token}"`, 
            `(Similarity: ${(similar.similarity * 100).toFixed(2)}%)`
        );
    });

    // Bonus: Visualization of learned representations
    console.log("\n--- Full Token Embeddings ---");
    tokens.forEach(token => {
        const fullEmbedding = nn.embedString(token, maxLength);
        console.log(`${token}: [${fullEmbedding.map(e => e.toFixed(4))}]`);
    });
}
// Run the comprehensive demonstration
demonstrateNeuralNetCapabilities();