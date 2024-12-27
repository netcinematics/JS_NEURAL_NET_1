class LSTM {
    constructor(inputSize, hiddenSize) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = 2;  // Two binary classifications
        
        // Initialize LSTM gates (input, forget, output, cell state)
        this.initializeGateWeights();
        
        // Learning rate
        this.learningRate = 0.01;
    }

    initializeGateWeights() {
        // Helper function for weight initialization
        const initMatrix = (rows, cols) => {
            const matrix = [];
            for (let i = 0; i < rows; i++) {
                matrix[i] = [];
                for (let j = 0; j < cols; j++) {
                    // Xavier initialization
                    matrix[i][j] = (Math.random() * 2 - 1) * Math.sqrt(2.0 / (rows + cols));
                }
            }
            return matrix;
        };

        // Input gate
        this.Wi = initMatrix(this.hiddenSize, this.inputSize);   // Input weights
        this.Ui = initMatrix(this.hiddenSize, this.hiddenSize);  // Hidden weights
        this.bi = new Array(this.hiddenSize).fill(0);           // Bias

        // Forget gate
        this.Wf = initMatrix(this.hiddenSize, this.inputSize);
        this.Uf = initMatrix(this.hiddenSize, this.hiddenSize);
        this.bf = new Array(this.hiddenSize).fill(1);  // Bias initialized to 1 for better gradient flow

        // Output gate
        this.Wo = initMatrix(this.hiddenSize, this.inputSize);
        this.Uo = initMatrix(this.hiddenSize, this.hiddenSize);
        this.bo = new Array(this.hiddenSize).fill(0);

        // Cell state
        this.Wc = initMatrix(this.hiddenSize, this.inputSize);
        this.Uc = initMatrix(this.hiddenSize, this.hiddenSize);
        this.bc = new Array(this.hiddenSize).fill(0);

        // Output projection
        this.Wy = initMatrix(this.outputSize, this.hiddenSize);
        this.by = new Array(this.outputSize).fill(0);
    }

    sigmoid(x) {
        return 1 / (1 + Math.exp(-x));
    }

    tanh(x) {
        return Math.tanh(x);
    }

    forward(sequence) {
        const T = sequence.length;
        const states = {
            h: [new Array(this.hiddenSize).fill(0)],  // Hidden states
            c: [new Array(this.hiddenSize).fill(0)],  // Cell states
            i: [], f: [], o: [], g: [],                // Gates
            y: []                                      // Outputs
        };

        // Process each character in sequence
        for (let t = 0; t < T; t++) {
            const xt = sequence[t];
            const ht_1 = states.h[t];
            const ct_1 = states.c[t];

            // Input gate
            const it = new Array(this.hiddenSize).fill(0);
            for (let j = 0; j < this.hiddenSize; j++) {
                let sum = this.bi[j];
                for (let k = 0; k < this.inputSize; k++) {
                    sum += this.Wi[j][k] * xt[k];
                }
                for (let k = 0; k < this.hiddenSize; k++) {
                    sum += this.Ui[j][k] * ht_1[k];
                }
                it[j] = this.sigmoid(sum);
            }

            // Forget gate
            const ft = new Array(this.hiddenSize).fill(0);
            for (let j = 0; j < this.hiddenSize; j++) {
                let sum = this.bf[j];
                for (let k = 0; k < this.inputSize; k++) {
                    sum += this.Wf[j][k] * xt[k];
                }
                for (let k = 0; k < this.hiddenSize; k++) {
                    sum += this.Uf[j][k] * ht_1[k];
                }
                ft[j] = this.sigmoid(sum);
            }

            // Output gate
            const ot = new Array(this.hiddenSize).fill(0);
            for (let j = 0; j < this.hiddenSize; j++) {
                let sum = this.bo[j];
                for (let k = 0; k < this.inputSize; k++) {
                    sum += this.Wo[j][k] * xt[k];
                }
                for (let k = 0; k < this.hiddenSize; k++) {
                    sum += this.Uo[j][k] * ht_1[k];
                }
                ot[j] = this.sigmoid(sum);
            }

            // Cell state candidate
            const gt = new Array(this.hiddenSize).fill(0);
            for (let j = 0; j < this.hiddenSize; j++) {
                let sum = this.bc[j];
                for (let k = 0; k < this.inputSize; k++) {
                    sum += this.Wc[j][k] * xt[k];
                }
                for (let k = 0; k < this.hiddenSize; k++) {
                    sum += this.Uc[j][k] * ht_1[k];
                }
                gt[j] = this.tanh(sum);
            }

            // New cell state
            const ct = new Array(this.hiddenSize).fill(0);
            for (let j = 0; j < this.hiddenSize; j++) {
                ct[j] = ft[j] * ct_1[j] + it[j] * gt[j];
            }

            // New hidden state
            const ht = new Array(this.hiddenSize).fill(0);
            for (let j = 0; j < this.hiddenSize; j++) {
                ht[j] = ot[j] * this.tanh(ct[j]);
            }

            // Store states
            states.i.push(it);
            states.f.push(ft);
            states.o.push(ot);
            states.g.push(gt);
            states.c.push(ct);
            states.h.push(ht);

            // Calculate output only for the last timestep
            if (t === T - 1) {
                const yt = new Array(this.outputSize).fill(0);
                for (let j = 0; j < this.outputSize; j++) {
                    let sum = this.by[j];
                    for (let k = 0; k < this.hiddenSize; k++) {
                        sum += this.Wy[j][k] * ht[k];
                    }
                    yt[j] = this.sigmoid(sum);
                }
                states.y.push(yt);
            }
        }

        return states;
    }

    train(sequences, targets, epochs) {
        const errors = [];
        
        for (let epoch = 0; epoch < epochs; epoch++) {
            let epochError = 0;
            
            for (let i = 0; i < sequences.length; i++) {
                // Forward pass
                const states = this.forward(sequences[i]);
                const prediction = states.y[states.y.length - 1];
                
                // Calculate error
                for (let j = 0; j < this.outputSize; j++) {
                    epochError += Math.pow(targets[i][j] - prediction[j], 2);
                }

                // Backward pass (simplified for demonstration)
                // In practice, you'd implement full BPTT here
                this.backpropagate(sequences[i], states, targets[i]);
            }
            
            epochError /= sequences.length;
            errors.push(epochError);
            
            // Early stopping
            if (epochError < 0.001) break;
        }
        
        return errors;
    }

    backpropagate(sequence, states, target) {
        // Simplified backpropagation for demonstration
        // In practice, you'd implement full BPTT
        const T = sequence.length;
        const lastH = states.h[T];
        
        // Output gradient
        const outputDelta = new Array(this.outputSize);
        const prediction = states.y[states.y.length - 1];
        
        for (let i = 0; i < this.outputSize; i++) {
            outputDelta[i] = (prediction[i] - target[i]) * prediction[i] * (1 - prediction[i]);
        }
        
        // Update output weights
        for (let i = 0; i < this.outputSize; i++) {
            this.by[i] -= this.learningRate * outputDelta[i];
            for (let j = 0; j < this.hiddenSize; j++) {
                this.Wy[i][j] -= this.learningRate * outputDelta[i] * lastH[j];
            }
        }
    }
}

// Test the LSTM
function testLSTM() {

    // Helper function to convert word to sequence of character vectors
    function wordToSequence(word, inputSize) {
        return Array.from(word).map(char => {
            const vector = new Array(inputSize).fill(0);
            vector[char.charCodeAt(0) % inputSize] = 1;  // One-hot encoding
            return vector;
        });
    }

    // Create training data
    const trainingWords = [
        "apple", "Banana", "cat", "Dog", "elephant",
        "art", "Book", "car", "Door", "egg",
        "another", "CAPS", "simple", "Test", "animal"
    ];
debugger;
    const inputSize = 26;  // Size of character embedding
    const hiddenSize = 32; // Hidden state size
    const lstm = new LSTM(inputSize, hiddenSize);

    // Prepare training data
    const sequences = trainingWords.map(word => wordToSequence(word, inputSize));
    const targets = trainingWords.map(word => [
        word.startsWith('a') ? 1 : 0,
        /[A-Z]/.test(word) ? 1 : 0
    ]);

    // Train the network
    console.log("Training LSTM...");
    const errors = lstm.train(sequences, targets, 100);
    console.log("Final error:", errors[errors.length - 1]);

    // Test the network
    const testWords = ["amazing", "and", "zebra", "Apple", "test"];
    console.log("\nTesting LSTM:");
    for (const word of testWords) {
        const sequence = wordToSequence(word, inputSize);
        const states = lstm.forward(sequence);
        const output = states.y[states.y.length - 1];
        
        console.log(`\nWord: ${word}`);
        console.log(`Starts with 'a': ${output[0].toFixed(3)} (Expected: ${word.startsWith('a') ? 1 : 0})`);
        console.log(`Contains uppercase: ${output[1].toFixed(3)} (Expected: ${/[A-Z]/.test(word) ? 1 : 0})`);
    }
}

// Run the test
testLSTM();