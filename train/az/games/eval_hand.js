const path=require('path'), fs=require('fs');
const E=require('../../engine')(path.join(__dirname,'..','..','..','hexwar.html'));
const G=require('../web/gameweb.js')(E);
const R=JSON.parse(fs.readFileSync(process.argv[2])); E.loadMap(R.mapDef||R.map);
let s=G.newState();
for(let round=1;!s.over&&round<=R.round;round++) for(let half=0;half<2&&!s.over;half++){
  const p=G.toMove(s);
  if(p===1) console.log(`round ${round} AI to move: hp ${s.hp}  hand eval for AI ${E.evalBoard(s,1,true).toFixed(1)}`);
  for(const m of R.moves.filter(m=>m[0]===round&&m[1]===p)) s=G.apply(s,E.TIDX[m[2]]*10000+m[3]*100+m[4]);
  if(p===1) console.log(`           after AI deploys: hand eval ${E.evalBoard(s,1,true).toFixed(1)}`);
  s=G.apply(s,G.END);
}
