
// ************* AI - NEOLOGISM EXPERIMENT ****************
// MINIMAL Neural Network example, for wordcrafting study.
//   - generated by Claude, revised by Gemini.
//   - trains an NN then AFFIRMS presence in nn.
//------------------------------------------------------
//----------------------UI-VARIABLES--------------
const NNTXT_1_ELEM = document.getElementById("NNTXT_1_ELEM");
const OUTPUT_1_ELEM = document.getElementById("OUTPUT_1_ELEM");
const TXT_INPUT_1_ELEM = document.getElementById('TXT_INPUT_1_')
const TEST_BTN_1_ELEM = document.getElementById('TEST_BTN_1_')
//-----------------------------------------------
//------------------------VISUALIZATION-----------
import { AI_VIZ_LOG } from './module_AI_VIZ_LOG_1.js';
import { AI_BRAIN_VIZ_1 } from './module_ai_BRAIN_VIZ_1.js';
const CANVAS_1_ELEM = document.getElementById('CANVAS_BRAIN_VIZ_1');
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
        AI_VIZ_LOG('--🎲 RANDOM BIAS/LR/WEIGHTS ⚖️')
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
        AI_VIZ_LOG('-🤖 TOKEN_VECTOR 🤖',token)
            //TODO VECTOR DETECTOR
            // AI_VIZ_LOG("\n--🔬 VIZ: VECTOR _DETECTOR 🔬");
            //AI_VIZ_BRAIN.VECTOR_DETECTOR(input,token)

        return input;
    }

    // Forward pass through the network
    forward(input) {  // Hidden layer : sum of vector across weights, with bias
        this.hidden = this.weightsIH[0].map((_, j) => {
            let sum = this.biasH[j];
            for (let i = 0; i < input.length; i++) {
                sum += input[i] * this.weightsIH[i][j]; //SUM MATRIX COLUMN
            }
            return this.sigmoid(sum); //activation fn, also //TODO: ReLU
        }); //hidden sigmoids.
        AI_VIZ_LOG('-⚡ HL:ActiveFn/Weights/Bias/sigma ⚡')
        // Output layer: sum of hidden * weight
        this.output = this.weightsHO[0].map((_, j) => {
            let sum = this.biasO[j];
            for (let i = 0; i < this.hidden.length; i++) {
                sum += this.hidden[i] * this.weightsHO[i][j]; //SUM MATRIX COLUMN
            }
            return this.sigmoid(sum); //activation fn also //TODO: ReLU
        });

        return this.output;
    }

    // Train the network using backpropagation
    train(input, target) { //target array of target types.
        // Forward pass
        AI_VIZ_LOG('--🏈 FORWARD_PASS 🏈')
        this.forward(input); //this.output

        // Output layer error
        const outputErrors = target.map((t, i) => (t - this.output[i]));
        AI_VIZ_LOG('--HL: OUTPUT ERROR')
        AI_VIZ_LOG('--OL: ERROR')
        
        // Hidden layer error
        const hiddenErrors = this.hidden.map((_, i) => {
            let error = 0;
            for (let j = 0; j < outputErrors.length; j++) {
                error += outputErrors[j] * this.weightsHO[i][j];
            }
            return error;
        });

        // Update weights and biases
        AI_VIZ_LOG('-Update WEIGHT/BIAS/Learn',this.learningRate)
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
function queryNeuralNetwork_1_(e){
    if(!TXT_INPUT_1_ELEM){console.log('err: missing input'); return}
    const val = TXT_INPUT_1_ELEM.value;
    if(!val){console.log('needs input'); return;}
    tokens = val.split(' '); // MULTIPLE INPUT (token test) BY SPACE
    OUTPUT_1_ELEM.innerHTML = ''; //clear out put
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
            OUTPUT_1_ELEM.innerHTML += `${token} || NO: ${txtPCTS[1]} || yes: ${txtPCTS[0]}<br>`;
        } else { //NO
            OUTPUT_1_ELEM.innerHTML += `${token} || YES: ${txtPCTS[0]} || no: ${txtPCTS[1]}<br>`;
        }
        // OUTPUT_1_ELEM.innerHTML = `${token}||${txtROW}||${1234}`;
    });
}
TEST_BTN_1_ELEM.onclick = queryNeuralNetwork_1_;
let nn_1;
// TOKENS STRONGER UP FRONT with MORE EPOCH, less with less epoch.
// let tokens = ["aa", "ab", "abc", "aaa", "bb", "ba", "ab", "abb", "bbb"];
let tokens = ["aa", "ab", "abc", "aaa", "bb", "ff", "xxx", "abb", "bbb"];
// let tokens = ["ape", "tree", "abc", "aaa", "bb", "ba", "ab", "abb", "bbb"];
function START_BRAIN(){
    AI_VIZ_LOG("----🧠 NN:INIT 🧠");
    nn_1 = new NeuralNetwork_1(52, 10, 2); // 52 inputs (26 letters * 2 positions), 10 hidden neurons, 2 outputs

    NNTXT_1_ELEM.innerHTML = tokens; //set tokens in UI.
    // INPUT_1_ELEM.innerHTML = tokens; //set tokens in UI.

    AI_VIZ_LOG("----🦾 NN:TRAIN_FRAME 🦾");
    function trainExample() {
        AI_VIZ_LOG("\n---💫 NN:TOKEN_LOOP 💫");
        // Example: Train to recognize if a token contains double letters
        tokens.forEach(token => {
            AI_VIZ_LOG("---🗃️ INPUT_VECTORS 🗃️");
            const input = nn_1.tokenToInput(token);//contains: VECTOR_DETECTOR.
            AI_VIZ_LOG("---🎯 TARGET_CASE 🎯");
            const target = [
                token[0] === token[1] ? 1 : 0,  // First NUM: has double letters
                token.includes('b') ? 1 : 0      // Second NUM: contains 'a'
            ];
            AI_VIZ_LOG("---🌪️ TRAIN_Fn 🌪️");
            nn_1.train(input, target);
            AI_VIZ_LOG("---🚧 END_TRAIN 🚧",token);
        });
    }

    // Train the network
    // debugger;
    let epoch_num = 1000;
    console.log('-----🌌 EPOCH_LOOP 🌌',epoch_num)
    for (let i = 0; i < epoch_num; i++) {
        trainExample();
        if(i%(epoch_num*0.1)===0){ //ten epoch counters
            AI_VIZ_LOG('-----🧭 EPOCH 🧭',i)
        }
    }

    OUTPUT_1_ELEM.innerHTML = ''; //clear out put
    // Test the network //optimize this TODO: move vars out to runFN() pattern.
    console.log('TOKEN_INTELLIGENCE:[','double letter,','contains(b)')
    tokens.forEach(token => {
        const input = nn_1.tokenToInput(token);
        const output = nn_1.forward(input);
        console.log(`Token: ${token}, Output: ${output.map(v => v.toFixed(3))}`);
        OUTPUT_1_ELEM.innerHTML = `Token: ${token}, Output: ${output.map(v => v.toFixed(3))}`
        // debugger;
        // const txtPCTS = output.map( (v) => { 
        //     // return v.toFixed(3); 
        //     return v.toFixed(2)*100;
        //     // let decimal = v.toFixed(2)*100;
        //     // return decimal.toFixed(0)+'%'; //human readable %
        // });
        // // const txtPCTS = txtTOKENS.split(',')
        // if(txtPCTS[0]<txtPCTS[1]){//YES
        //     OUTPUT_1_ELEM.innerHTML += `${token} || YES: ${txtPCTS[1]} || No: ${txtPCTS[0]}<br>`;
        // } else { //NO
        //     OUTPUT_1_ELEM.innerHTML += `${token} || no: ${txtPCTS[0]} || Yes: ${txtPCTS[1]}<br>`;
        // }
        // // OUTPUT_1_ELEM.innerHTML = `${token}||${txtROW}||${1234}`;
    });
    console.log('TOKEN_TEST:','EVAL: double letter and','contains(b)')
}; START_BRAIN();

function RENDER_BRAIN_VIZ(){

    const AI_BRAIN_VIZ = new AI_BRAIN_VIZ_1(nn_1,CANVAS_1_ELEM);
    
    // 1. Initial Weight Visualization (Before Training)
    AI_VIZ_LOG("---🔬 VIZ: AI_BRAIN 🔬");
    AI_BRAIN_VIZ.render_NEURON_WEIGHTS(); //visualizeWeights
    // AI_BRAIN_VIZ.render_BASELINE([{title:'a',title:'b'}])
    // AI_BRAIN_VIZ.render_BASELINE([{title:'a',title:'b'}])


}; RENDER_BRAIN_VIZ();

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

    // Create visualizerAI_BRAIN_VIZ
    const AI_BRAIN_VIZ = new NeuralNetVisualizer(nn);

    // 1. Initial Weight Visualization (Before Training)
    //console.log("---🔬 VIZ: Neural Network State 🔬");
    AI_BRAIN_VIZ.visualizeWeights();

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
    //console.log("\n---🦾 VIZ:WEIGHTS 🦾");
    AI_BRAIN_VIZ.visualizeWeights();

    // 4. Activation Visualization
    //console.log("\n⚡ VIZ:ACTIVIATION ⚡");
    const sampleInput = nn.stringToOneHot("hello", maxLength);
    AI_BRAIN_VIZ.visualizeActivations(sampleInput);

    // 5. Learning Trajectory
    //console.log("\n---🚀 VIZ:TRAJECTORY 🚀");
    AI_BRAIN_VIZ.visualizeLearningTrajectory(tokens, maxLength);
}
// debugger;
// Run the visualization demonstration
// demonstrateNeuralNetVisualization();

// Token: aa, Output: 0.955,0.030
// Token: ab, Output: 0.027,0.982
// Token: abc, Output: 0.027,0.982
// Token: aaa, Output: 0.955,0.030
// Token: bb, Output: 0.936,0.999
// Token: ba, Output: 0.101,0.951
// Token: ab, Output: 0.027,0.982
// Token: abb, Output: 0.027,0.982
// Token: bbb, Output: 0.936,0.999