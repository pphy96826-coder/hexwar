// Random point-symmetric maps, so the network learns to read terrain instead of memorising three boards.
// Same format as MAPS in hexwar.html: {id,name,cols,rows,base:[c,r],t:{terrain:[[c,r],...]}} (one side; mirrored on load).
const exists=(C,R,c,r)=>c>=0&&c<C&&(c%2===1?(r>=1&&r<=R&&r%2===1):(r>=2&&r<=R-1&&r%2===0));
const hd=(c1,r1,c2,r2)=>{const dc=Math.abs(c1-c2),dr=Math.abs(r1-r2);return dc+Math.max(0,(dr-dc)/2);};
const DIRS=[[0,-2],[1,-1],[1,1],[0,2],[-1,1],[-1,-1]];
const pick=a=>a[Math.floor(Math.random()*a.length)];

function randomMap(seedName){
  for(let attempt=0;attempt<200;attempt++){
    const C=pick([9,11,13,15]), R=pick([19,21,23,25]);
    const mir=(c,r)=>[C-1-c,R+1-r];
    const cc=(C-1)/2;
    let bc=cc+pick([0,0,0,-1,1]);
    let br=bc%2===1?R-2:R-3;            // one row of cells behind the base
    if(!exists(C,R,bc,br)) continue;
    const T=new Map(), key=(c,r)=>c*100+r;
    const set=(c,r,t)=>{if(!exists(C,R,c,r))return false;const m=mir(c,r);T.set(key(c,r),t);T.set(key(...m),t);return true;};
    const [b2c,b2r]=mir(bc,br);
    const nearBase=(c,r)=>hd(c,r,bc,br)<=2||hd(c,r,b2c,b2r)<=2;
    const mid=(R+1)/2;
    // river: a zig-zag line across the middle, mirrored, with 1-3 crossings
    if(Math.random()<0.5){
      const rc=[]; let r=mid;
      for(let c=0;c<=cc;c++){
        const want=c%2===1?1:0;                       // parity of rows in this column
        const opts=[r-1,r+1,r].filter(x=>Math.abs(x-mid)<=2&&(x%2===want)&&exists(C,R,c,x));
        if(c>0) r=pick(opts.filter(x=>Math.abs(x-rc[c-1])===1).length?opts.filter(x=>Math.abs(x-rc[c-1])===1):opts);
        else r=pick(opts.length?opts:[mid]);
        rc.push(r);
      }
      const river=[];
      for(let c=0;c<cc;c++) river.push([c,rc[c]]);
      // centre column: on the axis if the parity allows, otherwise the two cells around it
      if(cc%2===mid%2) river.push([cc,mid]); else river.push([cc,mid-1]);
      for(const [c,r] of river) set(c,r,'river');
      const nb=1+Math.floor(Math.random()*3);
      for(let i=0;i<nb;i++){const [c,r]=pick(river);set(c,r,Math.random()<0.45?'mine':'bridge');}
    }
    // rock walls, forests, hills, mines on one side (mirrored)
    const cells=[]; for(let c=0;c<C;c++) for(let r=mid+1;r<=R;r++) if(exists(C,R,c,r)) cells.push([c,r]);
    const free=([c,r])=>!T.has(key(c,r))&&!nearBase(c,r);
    const walls=Math.floor(Math.random()*4);
    for(let w=0;w<walls;w++){
      let [c,r]=pick(cells); const d=pick(DIRS), len=2+Math.floor(Math.random()*2);
      for(let i=0;i<len;i++){ if(free([c,r])) set(c,r,'rock'); c+=d[0]; r+=d[1]; }
    }
    for(const [t,n] of [['forest',1+Math.floor(Math.random()*3)],['hill',Math.floor(Math.random()*3)],['mine',Math.floor(Math.random()*2)]])
      for(let i=0;i<n;i++){const x=pick(cells);if(free(x))set(x[0],x[1],t);}
    // validate: both bases have 6 open neighbours and a land route between them
    const blocked=(c,r)=>{const t=T.get(key(c,r));return t==='rock'||t==='river';};
    const nbs=(c,r)=>DIRS.map(([a,b])=>[c+a,r+b]).filter(([x,y])=>exists(C,R,x,y));
    if(nbs(bc,br).length!==6||nbs(bc,br).some(([x,y])=>blocked(x,y))||T.has(key(bc,br))) continue;
    const seen=new Set([key(bc,br)]), q=[[bc,br]];
    while(q.length){const [c,r]=q.pop();for(const [x,y] of nbs(c,r)){const k=key(x,y);if(seen.has(k)||blocked(x,y))continue;seen.add(k);q.push([x,y]);}}
    if(!seen.has(key(b2c,b2r))) continue;
    // write one side only, in the MAPS format (loadMap mirrors it)
    const t={}, done=new Set();
    for(const [k,v] of T){ if(done.has(k)) continue; const c=Math.floor(k/100), r=k%100, m=key(...mir(c,r)); done.add(k); done.add(m); (t[v]=t[v]||[]).push([c,r]); }
    return {id:seedName||('rand'+Math.floor(Math.random()*1e9)),name:'随机',cols:C,rows:R,base:[bc,br],t};
  }
  throw new Error('could not generate a valid map');
}
module.exports={randomMap};
if(require.main===module){
  // print a few maps as text to eyeball them
  for(let i=0;i<3;i++){
    const m=randomMap(); const T=new Map(); const mir=(c,r)=>[m.cols-1-c,m.rows+1-r];
    for(const [t,l] of Object.entries(m.t)) for(const [c,r] of l){T.set(c*100+r,t);const [a,b]=mir(c,r);T.set(a*100+b,t);}
    const ch={river:'~',bridge:'=',mine:'$',rock:'#',forest:'f',hill:'^'};
    const [b2c,b2r]=mir(...m.base);
    console.log(`${m.cols}x${m.rows} base ${m.base}`);
    for(let r=1;r<=m.rows;r++){let line='';for(let c=0;c<m.cols;c++){const ok=exists(m.cols,m.rows,c,r);line+=!ok?' ':((c===m.base[0]&&r===m.base[1])||(c===b2c&&r===b2r))?'B':(ch[T.get(c*100+r)]||'.');}console.log('  '+line);}
  }
}
