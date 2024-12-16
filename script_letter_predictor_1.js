class NeuralNetwork {
    constructor(inputSize, hiddenSize, outputSize) {
      // Initialize weights with small random values
      this.weightsInputHidden = this.initializeWeights(inputSize, hiddenSize);
      this.weightsHiddenOutput = this.initializeWeights(hiddenSize, outputSize);
      
      // Initialize biases
      this.biasHidden = new Array(hiddenSize).fill(0).map(() => Math.random() * 0.1);
      this.biasOutput = new Array(outputSize).fill(0).map(() => Math.random() * 0.1);
    }
  
    // Initialize weights with small random values
    initializeWeights(rows, cols) {
      return Array.from({ length: rows }, () => 
        Array.from({ length: cols }, () => (Math.random() * 0.2) - 0.1)
      );
    }
  
    // Sigmoid activation function
    sigmoid(x) {
      let sigma =  1 / (1 + Math.exp(-x));
      if(isNaN(sigma)){ //true for any non number including NaN
        // debugger;
        sigma = 0;
      }
      return sigma;
      // return 1 / (1 + Math.exp(-x)); //FIX if NaN return 0
    }
  
    // Forward propagation
    forward(input) {
  // debugger;
      // Calculate hidden layer
      const hiddenLayer = this.weightsInputHidden.map((weights, i) => {
        const sum = weights.reduce((acc, weight, j) => acc + weight * input[j], 0) + this.biasHidden[i];
        return this.sigmoid(sum); //FIX: if Nan : 0;
        // const sigma = this.sigmoid(sum);
        // if(sigma===NaN){ return 0;}
        // return sigma;        
      });
  // debugger;
      // Calculate output layer
      const output = this.weightsHiddenOutput.map((weights, i) => {
        const sum = weights.reduce((acc, weight, j) => acc + weight * hiddenLayer[j], 0) + this.biasOutput[i];
        return this.sigmoid(sum); //FIX if NaN : 0;
        // const sigmo = this.sigmoid(sum);
        // if(sigmo===NaN){ return 0;}
        // return sigmo;
      });
    // debugger;
  
      return { hiddenLayer, output };
    }
  
    // Train on a single token
    train(input, target, learningRate = 0.1) { //TODO LEARN RATE
      // debugger;
      // Forward propagation
      const { hiddenLayer, output } = this.forward(input);
  
      // Calculate output error
      const outputError = target.map((t, i) => { return t - output[i]});
  
      // Update output layer weights and biases
      this.weightsHiddenOutput.forEach((weights, i) => {
        weights.forEach((weight, j) => {
          const delta = outputError[i] * output[i] * (1 - output[i]) * hiddenLayer[j];
          this.weightsHiddenOutput[i][j] += learningRate * delta;
        });
      });
  
      // Update output layer biases
      this.biasOutput = this.biasOutput.map((bias, i) => 
        bias + learningRate * outputError[i] * output[i] * (1 - output[i])
      );
  
      return output;
    }
  }
  
  // Encode 'aZZZa' as one-hot encoding
  function encodeToken(token) {
    const alphabet = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ';
    const oneHot = new Array(alphabet.length).fill(0);
    const index = alphabet.indexOf(token);
    if (index !== -1) oneHot[index] = 1;
    return oneHot;
  }
  
  // Main training script
  const inputSize = 52;  // Size of alphabet one-hot encoding
  const hiddenSize = 10;
  const outputSize = 52;
  
  const nn = new NeuralNetwork(inputSize, hiddenSize, outputSize);
  const token = 'c'; //TOKEN TO PREDICT?
  const targetToken = 'z'; //Z ?
  
  // Encode input and target
  const input = encodeToken(token);  
  const target = encodeToken(targetToken);
    
  // Training loop
  for (let epoch = 0; epoch < 20000; epoch++) { //TODO EPOCH 1000
    const output = nn.train(input, target, 0.5); //TODO LEARN RATE
    
    // Decode output
    const outputIndex = output.indexOf(Math.max(...output));
    const predictedToken = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'[outputIndex];
    
    if (epoch % 100 === 0) {
      console.log(`Epoch ${epoch}: Predicted token: ${predictedToken}`);
    }
  }
  
  // Final prediction
  const finalOutput = nn.forward(input);
  const finalOutputIndex = finalOutput.output.indexOf(Math.max(...finalOutput.output));
  const finalPredictedToken = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'[finalOutputIndex];
  console.log('Final predicted token:', finalPredictedToken);