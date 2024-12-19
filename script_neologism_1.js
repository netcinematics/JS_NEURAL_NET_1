class NeuralNetwork { //from cosinesimilarity
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
    // sigmoid(x) {
    //     return 1 / (1 + Math.exp(-x));
    // }
    // Sigmoid activation function
    sigmoid(x) {
        let sigma =  1 / (1 + Math.exp(-x));
        if(isNaN(sigma)){ //true for any non number including NaN
          // debugger;
          sigma = 0; /// TODO: ZERO NaNs?
        }
        return sigma;
        // return 1 / (1 + Math.exp(-x)); //FIX if NaN return 0
      }    
    //sigmoid Derivative : ACTIVATION SENSITIVITY
//     Gradient Calculation: In backpropagation, we need to know how much each neuron contributes to the error
// Learning Rate: The derivative helps determine how much to adjust weights
// Avoiding Vanishing Gradient: This derivative helps mitigate some issues with very deep networks
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
    embedString(string, maxLength) { //GetOneHot(); forward hidden layer
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
} //END NEURAL_NETWORK


class NeologismLearner {
    constructor(neuralNet, tokens, maxLength) {
        this.neuralNet = neuralNet;
        this.tokens = tokens;
        this.maxLength = maxLength;
        this.neologisms = new Map();
    }

    // Define a new word with semantic context
    defineNeologism(word, definition, contextTokens = []) {
        // 1. Generate initial embedding
        const initialEmbedding = this.generateNeologismEmbedding(word, contextTokens);
        
        // 2. Create semantic representation
        const semanticRepresentation = {
            word: word,
            definition: definition,
            embedding: initialEmbedding,
            contextTokens: contextTokens,
            usage: [], // Potential usage examples
            semanticNeighbors: []
        };

        // 3. Store neologism
        this.neologisms.set(word, semanticRepresentation);

        // 4. Integrate into neural network
        this.integrateNeologism(word, initialEmbedding);

        return semanticRepresentation;
    }

    // Generate initial embedding for new word
    generateNeologismEmbedding(word, contextTokens) {
        // If context tokens provided, average their embeddings
        if (contextTokens.length > 0) {
            const contextEmbeddings = contextTokens.map(token => 
                this.neuralNet.embedString(token, this.maxLength)
            );

            // Compute average embedding with some randomness
            return contextEmbeddings[0].map((val, index) => 
                contextEmbeddings.reduce((sum, emb) => sum + emb[index], 0) / contextEmbeddings.length 
                + (Math.random() - 0.5) * 0.2  // Add slight randomness
            );
        }

        // Fallback: use word's character-level encoding
        const wordEmbedding = this.neuralNet.stringToOneHot(word, this.maxLength);
        return this.neuralNet.forward(wordEmbedding).hiddenLayer;
    }

    // Integrate neologism into neural network
    integrateNeologism(word, embedding) {
        // Prepare training data
        const input = embedding;
        const target = new Array(this.tokens.length + 1).fill(0);
        target[this.tokens.length] = 1;  // Add new token to end of target vector

        // Extend neural network weights
        this.extendNeuralNetworkWeights();

        // Train network with new word
        for (let epoch = 0; epoch < 500; epoch++) {
            this.neuralNet.train(input, target, 0.05);
        }

        // Update tokens list
        this.tokens.push(word);
    }

    // Extend neural network weights to accommodate new token
    extendNeuralNetworkWeights() {
        // Extend input-hidden weights
        this.neuralNet.inputHiddenWeights.forEach(row => 
            row.push(Math.random() * 2 - 1)  // Add new random weight
        );

        // Add new row to input-hidden weights
        this.neuralNet.inputHiddenWeights.push(
            new Array(this.neuralNet.inputHiddenWeights[0].length).fill(0)
                .map(() => Math.random() * 2 - 1)
        );

        // Extend hidden-output weights
        this.neuralNet.hiddenOutputWeights.forEach(row => 
            row.push(Math.random() * 2 - 1)  // Add new random weight
        );

        // Add new row to hidden-output weights
        this.neuralNet.hiddenOutputWeights.push(
            new Array(this.neuralNet.hiddenOutputWeights[0].length).fill(0)
                .map(() => Math.random() * 2 - 1)
        );
    }

    // Find semantic neighbors for neologism
    findSemanticNeighbors(word, threshold = 0.6) {
        const neologism = this.neologisms.get(word);
        if (!neologism) return [];

        // Compare with all tokens and other neologisms
        const allTokens = [...this.tokens, ...Array.from(this.neologisms.keys())];
        
        return allTokens
            .map(token => ({
                token: token,
                similarity: this.neuralNet.cosineSimilarity(
                    neologism.embedding,
                    this.neuralNet.embedString(token, this.maxLength)
                )
            }))
            .filter(result => result.similarity > threshold)
            .sort((a, b) => b.similarity - a.similarity);
    }

    // Enrich neologism with additional context
    enrichNeologism(word, usageExample, additionalContextTokens = []) {
        const neologism = this.neologisms.get(word);
        if (!neologism) return null;

        // Add usage example
        if (usageExample) {
            neologism.usage.push(usageExample);
        }

        // Update context tokens
        neologism.contextTokens.push(...additionalContextTokens);

        // Recompute embedding with new context
        neologism.embedding = this.generateNeologismEmbedding(
            word, 
            neologism.contextTokens
        );

        // Retrain network with updated embedding
        this.integrateNeologism(word, neologism.embedding);

        // Update semantic neighbors
        neologism.semanticNeighbors = this.findSemanticNeighbors(word);

        return neologism;
    }
}

// Demonstration Function
function demonstrateNeologismLearning() {
    // Initial set of tokens
    const tokens = [
        "myth", "beast", "creature", "legend", "magical", 
        "fantasy", "monster", "imagination", "folklore"
    ];
    const maxLength = 15;  // Increased for longer words
debugger;
    // Create neural network
    const nn = new NeuralNetwork(
        maxLength,        // input size (max string length)
        12,               // hidden layer size 
        tokens.length     // output layer size
    );

    // Training phase for initial tokens
    console.log("🧠 Training Initial Neural Network 🧠");
    tokens.forEach((token, index) => {
        const input = nn.stringToOneHot(token, maxLength);
        const target = new Array(tokens.length).fill(0);
        target[index] = 1;

        // Train for multiple epochs
        for (let epoch = 0; epoch < 2000; epoch++) {
            nn.train(input, target, 0.03);
        }
    });

    // Create Neologism Learner
    const neologismLearner = new NeologismLearner(nn, tokens, maxLength);

    // Define new word "aDistructopuss"
    console.log("\n🆕 Defining New Neologism 🆕");
    const neologism = neologismLearner.defineNeologism(
        "aDistructopuss", 
        "A mythological beast of immense complexity and transformative power",
        ["myth", "beast", "creature", "magical"]
    );

    // Enrich with usage example
    console.log("\n📖 Enriching Neologism 📖");
    const enrichedNeologism = neologismLearner.enrichNeologism(
        "aDistructopuss",
        "In the ancient scrolls, the aDistructopuss was said to guard an invisible boundary, between a projection of illuzion and actual reality.",
        ["legend", "folklore", "imagination"]
    );

    // Find semantic neighbors
    console.log("\n🔍 Semantic Neighbors 🔍");
    const semanticNeighbors = neologismLearner.findSemanticNeighbors("aDistructopuss");

    // Prepare and print results
    const results = {
        neologism: enrichedNeologism,
        semanticNeighbors: semanticNeighbors
    };

    console.log(JSON.stringify(results, null, 2));

    return results;
}

// Run the neologism learning demonstration
const neologismResults = demonstrateNeologismLearning();