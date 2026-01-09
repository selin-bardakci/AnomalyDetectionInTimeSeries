// Homepage - Animated Time Series Visualization

const canvas = document.getElementById('timeSeriesCanvas');
const ctx = canvas.getContext('2d');

// Set canvas size
function resizeCanvas() {
    const parent = canvas.parentElement;
    canvas.width = parent.offsetWidth - 64;
    canvas.height = 400;
}

resizeCanvas();
window.addEventListener('resize', resizeCanvas);

// Generate multiple time series
const numSeries = 5;
const dataPoints = 200;
const series = [];

for (let s = 0; s < numSeries; s++) {
    const data = [];
    let value = Math.random() * 200 + 100;
    
    for (let i = 0; i < dataPoints; i++) {
        value += (Math.random() - 0.5) * 30;
        value = Math.max(50, Math.min(350, value));
        data.push(value);
    }
    
    series.push({
        data: data,
        color: `hsl(${180 + s * 30}, 70%, 60%)`,
        offset: 0
    });
}

// Animation
function animate() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw grid
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
    ctx.lineWidth = 1;
    
    for (let i = 0; i < 10; i++) {
        const y = (canvas.height / 10) * i;
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(canvas.width, y);
        ctx.stroke();
    }
    
    // Draw time series
    series.forEach(serie => {
        ctx.strokeStyle = serie.color;
        ctx.lineWidth = 2;
        ctx.globalAlpha = 0.6;
        
        ctx.beginPath();
        serie.data.forEach((value, i) => {
            const x = (canvas.width / dataPoints) * ((i + serie.offset) % dataPoints);
            const y = canvas.height - value;
            
            if (i === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        });
        ctx.stroke();
        
        // Update offset for animation
        serie.offset += 0.5;
        if (serie.offset > dataPoints) {
            serie.offset = 0;
        }
    });
    
    ctx.globalAlpha = 1;
    
    requestAnimationFrame(animate);
}

animate();
