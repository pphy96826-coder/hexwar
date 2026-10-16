// Positions from the player's winning games where the AI is to move, in the rounds before the player's ram landed.
const path=require('path'), fs=require('fs');
const E=require('../../engine')(path.join(__dirname,'..','..','..','hexwar.html'));
const G=require('../web/gameweb.js')(E);
const out=[];
for(const f of fs.readdirSync(__dirname).filter(f=>f.endsWith('.json'))){
  const R=JSON.parse(fs.readFileSync(path.join(__dirname,f))); if(!R.aiSides||!R.moves) continue;
  const human=R.aiSides.findIndex(a=>!a); if(human<0||R.winner!==human) continue;
  const rams=R.moves.filter(m=>m[1]===human&&m[2]==='ram').map(m=>m[0]); if(!rams.length) continue;
  const map=R.mapDef||R.map||'valley'; E.loadMap(map);
  let s=G.newState(), ok=true;
  for(let round=1;ok&&!s.over&&round<=R.round;round++) for(let half=0;half<2&&!s.over;half++){
    const p=G.toMove(s);
    if(p!==human&&rams.some(r=>r>=round&&r-round<=2)) out.push({id:`${f.slice(0,18)} r${round}`,map,s});
    for(const m of R.moves.filter(m=>m[0]===round&&m[1]===p)){ const a=E.TIDX[m[2]]*10000+m[3]*100+m[4]; if(!G.legal(s).includes(a)){ok=false;break;} s=G.apply(s,a); }
    if(!ok) break; s=G.apply(s,G.END);
  }
}
fs.writeFileSync(process.argv[2],JSON.stringify(out)); console.log(out.length,'puzzles:',out.map(p=>p.id).join(', '));
