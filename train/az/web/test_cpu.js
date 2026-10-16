const fs=require('fs'), NN=require('./nn.js');
const b=fs.readFileSync('net.bin'), W=NN.parse(b.buffer.slice(b.byteOffset,b.byteOffset+b.length));
const R=JSON.parse(fs.readFileSync('net.bin.json'));
const t=Date.now(), o=NN.forwardCPU(W,Uint8Array.from(R.x),2,R.H,R.W), dt=(Date.now()-t)/2;
let md=0; for(let i=0;i<R.lg.length;i++) md=Math.max(md,Math.abs(R.lg[i]-o.lg[i]));
console.log('v',Array.from(o.v),R.v,'e',Array.from(o.e),R.e,'max logit diff',md,'ms/pos',dt);
