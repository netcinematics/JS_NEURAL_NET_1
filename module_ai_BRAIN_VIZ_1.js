export class AI_BRAIN_VIZ_1 {
    constructor(neuralNet,canvasElem) {
        // Select the canvas and get its 2D rendering context
        if(!neuralNet){console.log('err: no network')}
        this.neuralNet = neuralNet;
        if(!canvasElem){console.log('err: no canvas')}
        this.canvas = canvasElem;//document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        // debugger;
        // this.canvas.width = 500; //default width
        this.canvas.width = canvasElem.parentElement.clientWidth * 0.8;
        this.canvas.height = 200; //default height

    }
    
    render_BASELINE(epochz = []) {
        // Clear the canvas
        // this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        // this.canvas.width = (epochz.length)?epochz.length*100:300;
        // Draw base timeline
        this.ctx.beginPath();
        this.ctx.strokeStyle = 'steelblue';
        this.ctx.lineWidth = 1;
        this.ctx.moveTo(25, this.canvas.height * 0.6 );
        this.ctx.lineTo(this.canvas.width - 25, this.canvas.height * 0.6 );
        this.ctx.stroke();

        let epoch = {};
        for(var i=0; i<epochz.length;i++){
            epoch = epochz[i];
            // console.log('EPOCH',epoch.title,(i+1)*10)
            this.DRAW_VERTICAL_LINE(((i+1)*100)+14);
            // this.DRAW_VERTICAL_LINE(epoch.x);
            // this.DRAW_POINT(epoch.x, epoch.y, epoch.emoji);
            
        }

    }

    DRAW_VERTICAL_LINE(x) {
        this.ctx.beginPath();
        this.ctx.strokeStyle = 'steelblue';
        this.ctx.moveTo(x, this.canvas.height * 0.6 - 10 );
        this.ctx.lineTo(x, this.canvas.height * 0.6 + 10);
        this.ctx.lineWidth = 1;
        this.ctx.stroke();
    }

    render_EPOCHZ(epochz) { 
        epochz.forEach(epoch => {
            // console.log('EPOCH',epoch.title)
            // this.DRAW_VERTICAL_LINE(epoch.x);
            this.DRAW_POINT(epoch.x, epoch.y, epoch.emoji);
        });
    }

    render_NEURON_WEIGHTS(){ //NEURON_WEIGHT VIZ.
        // this.neuralNet;
        // NN_MIN1:biasH, biasO, hidden, learningRate, output, weightsHO, weightsIH

        let CURSOR_POS = {x:30,y:0};

        let PAD_TOP = 30;
        let PAD_LEFT = 30;

        // for(var i=0; i< aiEpochz.length;i++){
        //     this.DRAW_GRID({
        //         rows : 20, cols : 20, 
        //         startX : 25 + (i*100), 
        //         startY : 30, spacing : 4}
        //     )
        // }
        // this.neuralNet.weightsIH 10x2
        // this.neuralNet.weightsHO 52x10
        let frame=[],vector=[],scalar=0;
        let rows=0,cols=0,spacing=4;
        let startX=0,startY=PAD_LEFT,colorVal='';;

        this.ctx.fillStyle = 'steelblue';
        this.ctx.font = "0.8em Arial italic bold "; //TITLE
        this.ctx.fillText("WEIGHTS IH", PAD_LEFT, PAD_TOP);

        rows = this.neuralNet.weightsIH.length;
        for (let frameIDX = 0; frameIDX < this.neuralNet.weightsIH.length; frameIDX++) {
            frame = this.neuralNet.weightsIH[frameIDX];
            cols = frame.length;
            startX = 25 + (frameIDX*spacing); 
            for (let vectorIDX = 0; vectorIDX < frame.length; vectorIDX++) {
                vector = frame[vectorIDX];
                // startX += vectorIDX * spacing;
                startY = 25 + (vectorIDX * spacing) + CURSOR_POS.y;
                colorVal = this.getColor_MAP_1(vector,-1,1)
                 // this.DRAW_POINT( startX + i * spacing, startY   );
                // console.log('point',startX,startY)
                this.ctx.beginPath();

                this.ctx.fillStyle = colorVal;
                // this.ctx.fillStyle = 'blue';
                this.ctx.arc(startX, startY, 1, 0, Math.PI * 2);
                this.ctx.fill();
            }
        }

        // let padding = this.canvas.width * 0.1;

        this.ctx.fillStyle = 'steelblue';
        this.ctx.font = "0.8em Arial italic bold "; //TITLE
        this.ctx.fillText("Weights HO", 20, 85);

        CURSOR_POS = {x:0,y:CURSOR_POS.y + 75}
        rows = this.neuralNet.weightsHO.length;
        for (let frameIDX = 0; frameIDX < this.neuralNet.weightsHO.length; frameIDX++) {
            frame = this.neuralNet.weightsHO[frameIDX];
            cols = frame.length;
            startX = PAD_LEFT + (frameIDX*spacing);// + CURSOR_POS.x; 
            for (let vectorIDX = 0; vectorIDX < frame.length; vectorIDX++) {
                vector = frame[vectorIDX];
                // startX += vectorIDX * spacing;
                startY = PAD_TOP + (vectorIDX * spacing) +CURSOR_POS.y;
                colorVal = this.getColor_MAP_1(vector,-1,1)
                 // this.DRAW_POINT( startX + i * spacing, startY   );
                // console.log('point',startX,startY)
                this.ctx.beginPath();

                this.ctx.fillStyle = colorVal;
                // this.ctx.fillStyle = 'blue';
                this.ctx.arc(startX, startY, 1, 0, Math.PI * 2);
                this.ctx.fill();
            }
        }        

        // debugger;
        this.ctx.fillStyle = 'steelblue';
        this.ctx.font = "0.8em Arial italic bold "; //TITLE
        this.ctx.fillText("BiasH (hidden)", PAD_LEFT, 130);

        CURSOR_POS = {x:0,y:CURSOR_POS.y + 40}
        rows = 1;
        for (let vectorIDX = 0; vectorIDX < this.neuralNet.biasH.length; vectorIDX++) {
            scalar = this.neuralNet.biasH[vectorIDX];
            startX = PAD_LEFT+ vectorIDX * spacing;
            startY =PAD_TOP+ CURSOR_POS.y;
            colorVal = this.getColor_MAP_1(scalar,-1,1)
            this.ctx.beginPath();
            this.ctx.fillStyle = colorVal;
            // console.log('point',startX,startY)
            this.ctx.arc(startX, startY, 1, 0, Math.PI * 2);
            this.ctx.fill();
        }


        this.ctx.fillStyle = 'steelblue';
        this.ctx.font = "0.8em Arial italic bold "; //TITLE
        this.ctx.fillText("Output", PAD_LEFT, 165);


        CURSOR_POS = {x:0,y:CURSOR_POS.y + 40}
        rows = 1;
        for (let vectorIDX = 0; vectorIDX < this.neuralNet.output.length; vectorIDX++) {
            scalar = this.neuralNet.output[vectorIDX];
            startX = PAD_LEFT+ (vectorIDX * 40);
            startY = PAD_TOP+ CURSOR_POS.y;
            colorVal = this.getColor_MAP_1(scalar,-1,1)
            this.ctx.fillStyle = colorVal;
            this.ctx.font = "0.6em Arial italic bold "; //TITLE
            this.ctx.fillText(scalar.toFixed(2), startX, startY);
        }


        // this.DRAW_GRID({rows : 14, cols : 14, startX : 30, startY : 30, spacing : 4} )
        // CURSOR_POS = {x:30,y:30 + 100};
        // this.DRAW_GRID({rows:14, cols:14, startX:CURSOR_POS.x, startY:CURSOR_POS.y, spacing:4} )
        // CURSOR_POS = {x:30+100,y:30 + 100};
        // this.DRAW_GRID({rows:14, cols:14, startX:CURSOR_POS.x, startY:CURSOR_POS.y, spacing:4} )
        // CURSOR_POS = {x:30+200,y:30 + 100};
        // this.DRAW_GRID({rows:14, cols:14, startX:CURSOR_POS.x, startY:CURSOR_POS.y, spacing:4} )
        // CURSOR_POS = {x:30+300,y:30 + 100};
        // this.DRAW_GRID({rows:14, cols:14, startX:CURSOR_POS.x, startY:CURSOR_POS.y, spacing:4} )

        
        // this.render_BASELINE(['a','b','c','d'])
        // 1. Weight Heatmap Visualization
        // console.log("\n🔍 Weight Visualization 🔍");
        
        // // Input to Hidden Layer Weights
        // console.log("Input to Hidden Layer Weights:");
        // this.printHeatmap(this.neuralNet.inputHiddenWeights, 
        // debugger;

        // this.printHeatmap(this.neuralNet.hidden,//inputHiddenWeights, 
        //     "Hidden", 
        //     (val) => this.colorMap(val, -1, 1)
        // );
  
        // console.log("\nHidden to Output Layer Weights:");
        // debugger;
        // this.printHeatmap(this.neuralNet.output,//hiddenOutputWeights, 
        //     "Output", 
        //     (val) => this.colorMap(val, -1, 1)
        // );



    }

    getColor_MAP_1 (value, min, max){ // Map value to color intensity
        // if(value<min){console.log('warning: color less than min',min,value)}
        // if(value>max){console.log('warning: color more than max',max,value)}
        const normalized = (value - min) / (max - min);
        let r = Math.floor(255 * (1 - normalized));
        let b = Math.floor(255 * normalized);
        //TODO: cannot be negative?
        r = (r<0)?0:r;
        b = (b<0)?0:b;
        return (value===0)?'black':(b>r)?'blue':'red';
    }
    // colorMap(value, min, max) { // Map value to color intensity
    //     const normalized = (value - min) / (max - min);
    //     let r = Math.floor(255 * (1 - normalized));
    //     let b = Math.floor(255 * normalized);
    //     //TODO: cannot be negative?
    //     r = (r<0)?0:r;
    //     b = (b<0)?0:b;
    //     return `\x1b[48;2;${r};0;${b}m  \x1b[0m`;
    // }

    // // 4. Detailed Weight Heatmap with Intensity
    // printHeatmap(weights, label, colorFunc) {
    //     console.log(`${label} Weight Heatmap:`);
    //     weights.forEach((row, i) => {
    //         debugger;
    //         const rowVisualization = row.map((val, j) => 
    //             colorFunc(val)
    //         ).join('');
    //         console.log(`Neuron ${i + 1}: ${rowVisualization}`);
    //     });
    // }


    DRAW_GRID(grid = {rows : 14, cols : 14, startX : 30, startY : 30, spacing : 4} ) {
        // console.log('grid', rows, cols, startX, startY, spacing);
        for (let row = 0; row < grid.rows; row++) {
            this.DRAW_VECTOR(
                grid.startX, 
                grid.startY + row * grid.spacing, 
                grid.cols, 
                grid.spacing
            );
        }
    }

    DRAW_VECTOR(startX, startY, count = 10, spacing = 10) {
        // console.log('vector', startX, startY, count, spacing);
        for (let i = 0; i < count; i++) {
            this.DRAW_POINT( startX + i * spacing, startY   );
        }
    }

    DRAW_POINT(x, y) {
        // console.log('point',x,y)
        this.ctx.beginPath();
        this.ctx.fillStyle = 'blue';
        this.ctx.arc(x, y, 1, 0, Math.PI * 2);
        this.ctx.fill();
    }
    
    DRAW_EMOJI(x, y, emoji) {
        console.log('emoji',x,y)
        // Draw emoji
        this.ctx.font = '12px Arial';
        this.ctx.fillStyle = 'white'; // Ensure visibility on black background
        // this.ctx.textAlign = 'center';
        this.ctx.fillText(emoji, x, y);
    }
}