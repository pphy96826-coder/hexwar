const path=require('path');
const E=require('../../engine')(path.join(__dirname,'..','..','..','hexwar.html'));
const G=require('../web/gameweb.js')(E); E.loadMap('valley');
const st=[]; for(let g=0;g<20;g++){let s=G.newState(); while(!s.over&&s.round<20){const L=G.legal(s); s=G.apply(s,Math.random()<0.3?G.END:L[Math.floor(Math.random()*L.length)]); st.push(s);}}
let t=performance.now(); for(const s of st) E.evalBoard(s,G.toMove(s),true); console.log('evalBoard us', ((performance.now()-t)/st.length*1000).toFixed(0));
t=performance.now(); for(const s of st) G.encode(s); console.log('encode us', ((performance.now()-t)/st.length*1000).toFixed(0));
