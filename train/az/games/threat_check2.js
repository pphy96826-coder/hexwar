const path=require('path'), fs=require('fs');
const E=require('../../engine')(path.join(__dirname,'..','..','..','hexwar.html'));
const G=require('../web/gameweb.js')(E), T=require('../web/threat.js')(E);
const R=JSON.parse(fs.readFileSync(process.argv[2])); E.loadMap(R.mapDef||R.map);
const human=R.aiSides.findIndex(a=>!a), ai=1-human;
let s=G.newState();
for(let round=1;!s.over&&round<=R.round;round++) for(let half=0;half<2&&!s.over;half++){
  const p=G.toMove(s);
  for(const m of R.moves.filter(m=>m[0]===round&&m[1]===p)){ if(m[2]==='ram') console.log(`   >> round ${round}: ${p===human?'human':'AI'} places ram at ${m[3]},${m[4]}`); s=G.apply(s,E.TIDX[m[2]]*10000+m[3]*100+m[4]); }
  if(p===ai){ const res=E.resolve(s), post=Object.assign({},s,{units:s.units.filter(u=>!res.dead.has(u.id)),round:s.round+1});
    const rr=T.ramReach(post,human);
    console.log(`round ${round} after AI deploy: hp ${s.hp}, human chain needed ${rr.n}, human bank ${s.bank[human]} | correction for AI ${T.correction(s,ai).toFixed(1)}  hand eval ${E.evalBoard(s,ai,true).toFixed(1)}`); }
  s=G.apply(s,G.END);
}
