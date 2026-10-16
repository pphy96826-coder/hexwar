// simEndRound must reproduce the live game's round transition exactly (deaths, revives, roads, rage, income, winner).
const path=require('path');
const E=require('../engine')(path.join(__dirname,'..','..','hexwar.html'));
let games=0, rounds=0, bad=0;
for(const map of ['valley','plateau','canyon']) for(let g=0;g<12;g++){
  E.newGame('hotseat',{map}); let st=E.st;
  while(st.phase!=='over'&&st.round<30){
    for(let i=0;i<2&&st.phase==='deploy';i++){
      const p=st.active, gen=E.aiGen(p,'easy',50); let r; do{r=gen.next();}while(!r.done);
      for(const [t,k] of r.value) if(st.bank[p]>=E.U[t].cost) E.place(p,t,k);
      if(i===1){
        // both sides deployed: predict the transition with simEndRound, then let the live game do it
        const sim={round:st.round,mover:1,units:st.units.slice(),roads:Object.assign({},st.roads),bank:st.bank.slice(),hp:st.hp.slice(),rage:st.rage.slice(),pending:st.pending.slice(),nextId:1e6};
        const n=E.simEndRound(sim);
        E.endTurn(); st=E.st; rounds++;
        const sig=x=>x.units.map(u=>`${u.type}${u.owner}@${u.c},${u.r}`).sort().join('|');
        const ok=st.phase==='over'
          ? (n.over&&n.winner===st.winner&&n.hp.join()===st.hp.join())
          : (!n.over&&n.round===st.round&&n.hp.join()===st.hp.join()&&n.bank.join()===st.bank.join()&&n.rage.join()===st.rage.join()&&sig(n)===sig(st)&&JSON.stringify(n.roads)===JSON.stringify(st.roads));
        if(!ok){bad++; if(bad<4) console.log('MISMATCH',map,'round',sim.round,{sim:{hp:n.hp,bank:n.bank,over:n.over,w:n.winner},live:{hp:st.hp,bank:st.bank,phase:st.phase,w:st.winner}});}
      } else E.endTurn();
      st=E.st;
    }
  }
  games++;
}
console.log(`games ${games}, rounds checked ${rounds}, mismatches ${bad}`);
