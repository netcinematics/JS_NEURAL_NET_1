const CANVAS_ID = 'CANVAS_ELEM_1'
class AI_Visualizer_1 {
    constructor(canvasId) {
        // Select the canvas and get its 2D rendering context
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        
        this.canvas.width = 500; //default width
        this.canvas.height = 200; //default height

    }
    
    render_BASELINE(epochz = []) {
        // Clear the canvas
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.canvas.width = (epochz.length)?epochz.length*100:300;
        // Draw base timeline
        this.ctx.beginPath();
        this.ctx.strokeStyle = 'white';
        this.ctx.moveTo(25, this.canvas.height * 0.6 );
        this.ctx.lineTo(this.canvas.width - 25, this.canvas.height * 0.6 );
        this.ctx.stroke();

        let epoch = {};
        for(var i=0; i<epochz.length;i++){
            epoch = epochz[i];
            console.log('EPOCH',epoch.title,(i+1)*10)
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

// Example usage
document.addEventListener('DOMContentLoaded', () => {
    const timelineVisualizer = new AI_Visualizer_1(CANVAS_ID);
    // debugger;
    // AI timeline EPOCHZ
    const aiEpochz = [
        { x: 100, y: 200, emoji: '🤖' , title:'Early AI concept'},
        { x: 250, y: 180, emoji: '💡' , title:'Machine Learning breakthrough'},
        { x: 400, y: 220, emoji: '🧠' , title:'Neural Networks'},
        { x: 550, y: 190, emoji: '🚀' , title:'Deep Learning era'},
        { x: 700, y: 210, emoji: '🤯' , title:'Generative AI'},
        { x: 50, y: 210, emoji: '🤯' , title:'GenAI'},
        { x: 50, y: 210, emoji: '🤯' , title:'GenAI'},
    ];

    timelineVisualizer.render_BASELINE(aiEpochz);
    // timelineVisualizer.render_EPOCHZ(aiEpochz);

    // timelineVisualizer.DRAW_GRID(
    //     {rows : 20, cols : 20, startX : 15, startY : 30, spacing : 4});
    for(var i=0; i< aiEpochz.length;i++){
        timelineVisualizer.DRAW_GRID({
            rows : 20, cols : 20, 
            startX : 25 + (i*100), 
            startY : 30, spacing : 4}
        )
    }
    // timelineVisualizer.DRAW_GRID(
    //     {rows : 20, cols : 20, startX : 110, startY : 30, spacing : 4});
    // timelineVisualizer.DRAW_GRID(
    //     {rows : 20, cols : 20, startX : 210, startY : 30, spacing : 4});
    // timelineVisualizer.DRAW_GRID(
    //     {rows : 20, cols : 20, startX : 310, startY : 30, spacing : 4});

    // Test 1: Draw a point at x:10, y:10
    timelineVisualizer.DRAW_POINT(100, 100);

    // Test 2: Draw an emoji at x:20, y:20
    timelineVisualizer.DRAW_EMOJI(20, 15, '🤖');


});
