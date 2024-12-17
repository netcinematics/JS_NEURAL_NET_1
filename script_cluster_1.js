// Advanced Semantic Clustering for Neural Network

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
                if(isNaN(hiddenLayer[j])){debugger;}
                if(hiddenLayer[j]===undefined){
                    debugger;
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
// debugger;
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

class SemanticAnalyzer {
    constructor(neuralNet, tokens, maxLength) {
        this.neuralNet = neuralNet;
        this.tokens = tokens;
        this.maxLength = maxLength;
        this.embeddings = this.generateEmbeddings();
    }

    // Generate embeddings for all tokens
    generateEmbeddings() {
        return this.tokens.map(token => ({
            token: token,
            embedding: this.neuralNet.embedString(token, this.maxLength)
        }));
    }

    // Compute cosine similarity between two embeddings
    cosineSimilarity(vec1, vec2) {
        const dotProduct = vec1.reduce((sum, val, i) => sum + val * vec2[i], 0);
        const magnitude1 = Math.sqrt(vec1.reduce((sum, val) => sum + val * val, 0));
        const magnitude2 = Math.sqrt(vec2.reduce((sum, val) => sum + val * val, 0));
        return dotProduct / (magnitude1 * magnitude2);
    }

    // Cluster tokens based on embedding similarity
    clusterTokens(similarityThreshold = 0.7) {
        const clusters = [];

        this.embeddings.forEach((currentToken, index) => {
            // Check if token is already in a cluster
            const existingClusterIndex = clusters.findIndex(cluster => 
                cluster.some(member => 
                    this.cosineSimilarity(
                        this.embeddings.find(e => e.token === member).embedding, 
                        currentToken.embedding
                    ) > similarityThreshold
                )
            );

            if (existingClusterIndex !== -1) {
                // Add to existing cluster
                clusters[existingClusterIndex].push(currentToken.token);
            } else {
                // Create new cluster
                clusters.push([currentToken.token]);
            }
        });

        return clusters;
    }

    // Generate semantic analysis JSON
    generateSemanticAnalysisJSON() {
        const clusters = this.clusterTokens();
        
        // Compute cluster characteristics
        const clusterAnalysis = clusters.map((cluster, index) => {
            // Representative embedding (average of cluster embeddings)
            // const representativeEmbedding = this.embeddings
            //     .filter(e => cluster.includes(e.token))
            //     .reduce((acc, curr) => 
            //         acc.map((val, i) => val + curr.embedding[i]), 
            //         new Array(curr.embedding.length).fill(0)
            //     )
            //     .map(val => val / cluster.length);
            const representativeEmbedding = cluster
                .reduce((acc, tokenIndex) => {
                    const embedding = this.embeddings.find(e => e.token === tokenIndex).embedding;
                    return acc.map((val, i) => val + embedding[i]);
                }, new Array(this.embeddings[0].embedding.length).fill(0))
                .map(val => val / cluster.length);            

            // Find most representative token
            const mostRepresentativeToken = cluster.reduce((mostRep, token) => {
                const tokenEmbedding = this.embeddings.find(e => e.token === token).embedding;
                const repSimilarity = this.cosineSimilarity(tokenEmbedding, representativeEmbedding);
                
                const currentRepSimilarity = mostRep 
                    ? this.cosineSimilarity(
                        this.embeddings.find(e => e.token === mostRep).embedding, 
                        representativeEmbedding
                    )
                    : -1;

                return repSimilarity > currentRepSimilarity ? token : mostRep;
            }, null);

            return {
                clusterId: index + 1,
                tokens: cluster,
                representativeToken: mostRepresentativeToken,
                representativeEmbedding: representativeEmbedding.map(val => val.toFixed(4)),
                clusterSize: cluster.length
            };
        });

        // Compute inter-cluster relationships
        const clusterRelationships = clusterAnalysis.map((cluster1, i) => 
            clusterAnalysis.slice(i + 1).map((cluster2, j) => ({
                cluster1Id: cluster1.clusterId,
                cluster2Id: cluster1.clusterId + j + 1,
                similarity: this.cosineSimilarity(
                    cluster1.representativeEmbedding.map(parseFloat), 
                    cluster2.representativeEmbedding.map(parseFloat)
                )
            }))
        ).flat().filter(rel => rel.similarity > 0.5);

        return {
            tokens: this.tokens,
            clusters: clusterAnalysis,
            clusterRelationships: clusterRelationships
        };
    }
}

// Demonstration Function
function demonstrateSemanticClustering() {
    // Diverse set of tokens with semantic relationships
    const tokens = [
        // Programming-related tokens
        // "code", "coding", "programmer", "develop", "software",
        
        // Language-related tokens
        "hello", "hi", //"hey", "greet", "welcome",
        
        // World-related tokens
        "world", "planet", "earth", "globe", "universe",
        
        // Abstract thinking tokens
        "think", "thought", "idea", "concept", "mind"
    ];
    const maxLength = 12;  // Maximum token length
    //TODO 10 error in init random?

    // Create neural network
    const nn = new NeuralNetwork(
        maxLength,        // input size (max string length)
        12,               // hidden layer size (more neurons for richer representation)
        tokens.length     // output layer size (number of tokens)
    );

    // Training phase
    console.log("🧠 Training Neural Network for Semantic Analysis 🧠");
    tokens.forEach((token, index) => {
        const input = nn.stringToOneHot(token, maxLength);
        const target = new Array(tokens.length).fill(0);
        target[index] = 1;  // One-hot target for each token

        // Train for multiple epochs to ensure learning
        for (let epoch = 0; epoch < 3000; epoch++) {
            nn.train(input, target, 0.03);  // Slightly reduced learning rate
        }
    });

    // Create Semantic Analyzer
    const semanticAnalyzer = new SemanticAnalyzer(nn, tokens, maxLength);

    // Generate and output semantic analysis
    console.log("\n📊 Semantic Analysis Results 📊");
    const semanticAnalysis = semanticAnalyzer.generateSemanticAnalysisJSON();

    // Pretty print the JSON output
    console.log(JSON.stringify(semanticAnalysis, null, 2));

    // Optional: Write to file (in a real application)
    // fs.writeFileSync('semantic_analysis.json', JSON.stringify(semanticAnalysis, null, 2));

    return semanticAnalysis;
}
debugger;
// Run the semantic clustering demonstration
const result = demonstrateSemanticClustering();