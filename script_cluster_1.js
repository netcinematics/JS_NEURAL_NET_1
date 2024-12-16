// Advanced Semantic Clustering for Neural Network

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
            const representativeEmbedding = this.embeddings
                .filter(e => cluster.includes(e.token))
                .reduce((acc, curr) => 
                    acc.map((val, i) => val + curr.embedding[i]), 
                    new Array(curr.embedding.length).fill(0)
                )
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
        "code", "coding", "programmer", "develop", "software",
        
        // Language-related tokens
        "hello", "hi", "hey", "greet", "welcome",
        
        // World-related tokens
        "world", "planet", "earth", "globe", "universe",
        
        // Abstract thinking tokens
        "think", "thought", "idea", "concept", "mind"
    ];
    const maxLength = 10;  // Maximum token length

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

// Run the semantic clustering demonstration
const result = demonstrateSemanticClustering();