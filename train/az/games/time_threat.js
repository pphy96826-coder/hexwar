const path=require('path');
const E=require('../../engine')(path.join(__dirname,'..','..','..','hexwar.html'));
const G=require('../web/gameweb.js')(E), T=require('../web/threat.js')(E);
const {randomMap}=require('../maps');
for(const mp of ['valley',randomMap()]){
  E.loadMap(mp);
  const st=[]; for(let g=0;g<10;g++){let s=G.newState(); while(!s.over&&s.round<16){const L=G.legal(s); s=G.apply(s,Math.random()<0.3?G.END:L[Math.floor(Math.random()*L.length)]); st.push(s);}}
  let t=performance.now(); for(const s of st) T.correction(s,G.toMove(s)); const a=(performance.now()-t)/st.length;
  t=performance.now(); for(const s of st) T.ramReach(s,0); const b=(performance.now()-t)/st.length;
  t=performance.now(); for(const s of st) E.evalBoard(s,G.toMove(s),true); const c=(performance.now()-t)/st.length;
  console.log(typeof mp==='string'?mp:'random', 'correction ms',a.toFixed(2),'ramReach ms',b.toFixed(2),'evalBoard ms',c.toFixed(2));
}
