// ************* AI - NEOLOGISM EXPERIMENT ****************
// MINIMAL Neural Network example, for wordcrafting study.
//   - generated by Claude, revised by Gemini.
//   - trains an NN then AFFIRMS presence in nn.
//------------------------------------------------------
//----------------------UI-VARIABLES--------------
const NNTXT_ELEM = document.getElementById("NNTXT_2_ELEM");
const INPUT_ELEM = document.getElementById("INPUT_2_ELEM");
const OUTPUT_ELEM = document.getElementById("OUTPUT_2_ELEM");
const TEST_BTN_2_ = document.getElementById("TEST_BTN_2_");
const TXT_INPUT_2_ = document.getElementById("TXT_INPUT_2_");
// onclick="queryNeuralNetwork_2_(event);">TES"testBTN_2_"T
import { neologizms } from './aWORDZa_1.js';
  //-----------------------------------------------
class NeuralNetwork_1 {
    constructor(inputSize, hiddenSize, outputSize) {
        // Initialize weights with random values between -1 and 1
        this.weightsIH = Array(inputSize).fill().map(() => 
            Array(hiddenSize).fill().map(() => Math.random() * 2 - 1)
        );
        this.weightsHO = Array(hiddenSize).fill().map(() => 
            Array(outputSize).fill().map(() => Math.random() * 2 - 1)
        );
        
        // Initialize biases
        this.biasH = Array(hiddenSize).fill().map(() => Math.random() * 2 - 1);
        this.biasO = Array(outputSize).fill().map(() => Math.random() * 2 - 1);
        
        this.learningRate = 0.1;
    }

    // Sigmoid activation function
    sigmoid(x) {
        return 1 / (1 + Math.exp(-x));
    }

    // Convert letter sequence to input vector
    // Each position represents presence (1) or absence (0) of a letter
    tokenToInput(token) {
        const input = Array(26 * 2).fill(0); // Space for 2 letters
        for (let i = 0; i < Math.min(token.length, 2); i++) {
            const charCode = token.charCodeAt(i) - 97; // 'a' starts at 97
            if (charCode >= 0 && charCode < 26) {
                input[i * 26 + charCode] = 1;
            }
        }
        return input;
    }

    // Forward pass through the network
    forward(input) {  // Hidden layer
        this.hidden = this.weightsIH[0].map((_, j) => {
            let sum = this.biasH[j];
            for (let i = 0; i < input.length; i++) {
                sum += input[i] * this.weightsIH[i][j];
            }
            return this.sigmoid(sum); //activation fn
        });

        // Output layer
        this.output = this.weightsHO[0].map((_, j) => {
            let sum = this.biasO[j];
            for (let i = 0; i < this.hidden.length; i++) {
                sum += this.hidden[i] * this.weightsHO[i][j];
            }
            return this.sigmoid(sum); //activation fn
        });

        return this.output;
    }

    // Train the network using backpropagation
    train(input, target) {
        // Forward pass
        this.forward(input);

        // Output layer error
        const outputErrors = target.map((t, i) => (t - this.output[i]));
        
        // Hidden layer error
        const hiddenErrors = this.hidden.map((_, i) => {
            let error = 0;
            for (let j = 0; j < outputErrors.length; j++) {
                error += outputErrors[j] * this.weightsHO[i][j];
            }
            return error;
        });

        // Update weights and biases
        // Output layer
        for (let i = 0; i < this.hidden.length; i++) {
            for (let j = 0; j < this.output.length; j++) {
                this.weightsHO[i][j] += this.learningRate * outputErrors[j] * 
                    this.output[j] * (1 - this.output[j]) * this.hidden[i];
            }
        }

        // Hidden layer
        for (let i = 0; i < input.length; i++) {
            for (let j = 0; j < this.hidden.length; j++) {
                this.weightsIH[i][j] += this.learningRate * hiddenErrors[j] * 
                    this.hidden[j] * (1 - this.hidden[j]) * input[i];
            }
        }

        // Update biases
        for (let i = 0; i < this.biasO.length; i++) {
            this.biasO[i] += this.learningRate * outputErrors[i] * 
                this.output[i] * (1 - this.output[i]);
        }
        for (let i = 0; i < this.biasH.length; i++) {
            this.biasH[i] += this.learningRate * hiddenErrors[i] * 
                this.hidden[i] * (1 - this.hidden[i]);
        }
    }
} //END NEURAL NETWORK class
// function queryNeuralNetwork_1(e){
TEST_BTN_2_.onclick = (e) => {
    if(!TXT_INPUT_2_){console.log('err: missing input'); return}
    const val = TXT_INPUT_2_.value;
    if(!val){console.log('needs input'); return;}
    tokens = val.split(' ')
    OUTPUT_ELEM.innerHTML = ''; //clear out put
    tokens.forEach(token => {
        const input = nn_1.tokenToInput(token);
        const output = nn_1.forward(input);
        console.log(`Token: ${token}, Output: ${output.map(v => v.toFixed(3))}`);
        // debugger;
        const txtPCTS = output.map( (v) => { 
            // return v.toFixed(3); 
            return v.toFixed(2)*100;
            // let decimal = v.toFixed(2)*100;
            // return decimal.toFixed(0)+'%'; //human readable %
        });
        // const txtPCTS = txtTOKENS.split(',')
        if(txtPCTS[0]<txtPCTS[1]){//YES
            OUTPUT_ELEM.innerHTML += `${token} || NO: ${txtPCTS[1]} || yes: ${txtPCTS[0]}<br>`;
        } else { //NO
            OUTPUT_ELEM.innerHTML += `${token} || YES: ${txtPCTS[0]} || no: ${txtPCTS[1]}<br>`;
        }
        // OUTPUT_ELEM.innerHTML = `${token}||${txtROW}||${1234}`;
    });
}

// Example usage:
let tokens = ["aa", "ab", "abc", "aaaa", "bb", "ba", "ab", "abb", "bbbb"];
// let tokens = neologizms.taxonomy;
const nn_1 = new NeuralNetwork_1(52, 10, 2); // 52 inputs (26 letters * 2 positions), 10 hidden neurons, 2 outputs
NNTXT_ELEM.innerHTML = tokens.join(' '); //set tokens in UI.
NNTXT_ELEM.innerHTML += tokens.join(' '); //set tokens in UI.
// INPUT_ELEM.innerHTML = tokens; //set tokens in UI.

// Training example
function trainExample() {
    // Example: Train to recognize if a token contains double letters
    tokens.forEach(token => {
        const input = nn_1.tokenToInput(token);
        const target = [
            token[0] === token[1] ? 1 : 0,  // First output: has double letters
            token.includes('b') ? 1 : 0      // Second output: contains 'a'
        ];
        nn_1.train(input, target);
    });
}

function RUN_NETWORK(){
    // Train the network
    // debugger;
    let epoch_num = 1000;                      //TODO EPOCH NUM
    for (let i = 0; i < epoch_num; i++) {
    // for (let i = 0; i < 1000; i++) {
    // for (let i = 0; i < 10; i++) {
        trainExample();
    }

    OUTPUT_ELEM.innerHTML = ''; //clear out put
    // Test the network 
    tokens.forEach(token => {

        const input = nn_1.tokenToInput(token);
        const output = nn_1.forward(input);

        console.log(`Token: ${token}, Output: ${output.map(v => v.toFixed(3))}`);
        // debugger;
        const txtPCTS = output.map( (v) => { 
            // return v.toFixed(3); 
            return v.toFixed(2)*100;
            // let decimal = v.toFixed(2)*100;
            // return decimal.toFixed(0)+'%'; //human readable %
        });
        // const txtPCTS = txtTOKENS.split(',')
        if(txtPCTS[0]<txtPCTS[1]){//YES
            OUTPUT_ELEM.innerHTML += `${token} || YES: ${txtPCTS[1]} || No: ${txtPCTS[0]}<br>`;
        } else { //NO
            OUTPUT_ELEM.innerHTML += `${token} || no: ${txtPCTS[0]} || Yes: ${txtPCTS[1]}<br>`;
        }
        // OUTPUT_ELEM.innerHTML = `${token}||${txtROW}||${1234}`;
    });
}; RUN_NETWORK();