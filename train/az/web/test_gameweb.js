// Checks gameweb.js against game.js: identical legal moves, encodings and action indices.
const path=require('path');
const E=require('../../engine')(path.join(__dirname,'..','..','..','hexwar.html'));
const G=require('../game')(E), GW=require('./gameweb.js')(E), {randomMap}=require('../maps');
let n=0,bad=0,t1=0,t2=0;
for(const mp of ['valley','plateau','canyon',...Array.from({length:8},randomMap)]){
  E.loadMap(mp);
  for(let g=0;g<10;g++){let s=G.newState(); while(!s.over&&s.round<30){
    let a=performance.now(); const A=G.legal(s), xa=G.encode(s); t1+=performance.now()-a;
    a=performance.now(); const B=GW.legal(s), xb=GW.encode(s); t2+=performance.now()-a; n++;
    let ok=A.length===B.length&&A.every((v,i)=>v===B[i])&&xa.H===xb.H&&xa.W===xb.W&&xa.x.every((v,i)=>v===xb.x[i]);
    if(ok) for(const act of A) if(G.actIndex(s,act)!==GW.actIndex(s,act)){ok=false;break;}
    if(!ok){bad++; if(bad<4){const d=[];for(let i=0;i<xa.x.length;i++)if(xa.x[i]!==xb.x[i])d.push([Math.floor(i/(xa.H*xa.W)),xa.x[i],xb.x[i]]);console.log('diff',d.slice(0,8),A.length,B.length);}}
    s=G.apply(s,Math.random()<0.3?G.END:A[Math.floor(Math.random()*A.length)]);
  }}
}
console.log('states',n,'mismatches',bad,'game.js us',(t1/n*1000).toFixed(0),'gameweb us',(t2/n*1000).toFixed(0));
