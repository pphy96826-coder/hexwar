// For a game record: at each turn of the AI, how much would the human's units hit the AI base next round if nobody
// moved (persistent threat), and could the human deploy a ram next to the AI base next round?
const path=require('path'), fs=require('fs');
const E=require('../../engine')(path.join(__dirname,'..','..','..','hexwar.html'));
const G=require('../web/gameweb.js')(E);
const R=JSON.parse(fs.readFileSync(process.argv[2])); E.loadMap(R.mapDef||R.map);
const human=R.aiSides.findIndex(a=>!a), ai=1-human;
function threat(s,p){   // to p's base
  const e=1-p, res=E.resolve(s), alive=s.units.filter(u=>!res.dead.has(u.id));
  const post={round:s.round+1,units:alive,roads:s.roads,bank:s.bank,hp:s.hp,rage:s.rage,pending:[],nextId:s.nextId,placed:0,mover:0};
  const r2=E.resolve(post);
  const nb=Math.min(10,s.bank[e]+E.income(s.round+1));
  const adj=new Set(E.NB1.get(E.BASE[p]));
  const ramCells=nb>=5?[...E.deployable(post,e,'ram')].filter(k=>adj.has(k)):[];
  return {stat:r2.baseDmg[p],ram:ramCells.length};
}
let s=G.newState();
for(let round=1;!s.over&&round<=R.round;round++) for(let half=0;half<2&&!s.over;half++){
  const p=G.toMove(s);
  for(const m of R.moves.filter(m=>m[0]===round&&m[1]===p)){ if(m[2]==='ram') console.log(`   >> round ${round}: ${p===human?'human':'AI'} places ram at ${m[3]},${m[4]}`); s=G.apply(s,E.TIDX[m[2]]*10000+m[3]*100+m[4]); }
  if(p===ai){ const t=threat(s,ai); console.log(`round ${round} after AI's deploy: AI hp ${s.hp[ai]}, human bank ${s.bank[human]}: static dmg to AI base ${t.stat}, cells where human could drop a ram next to AI base: ${t.ram}`); }
  s=G.apply(s,G.END);
}
