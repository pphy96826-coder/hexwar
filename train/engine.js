// Loads the game's own <script> from hexwar.html into a sandbox, so training uses exactly the rules you play.
const fs=require('fs'), vm=require('vm');
module.exports=function loadEngine(htmlPath){
  const html=fs.readFileSync(htmlPath,'utf8');
  const code=html.match(/<script>([\s\S]*)<\/script>/)[1];
  const els={};
  const fake=()=>({innerHTML:'',textContent:'',hidden:true,style:{},dataset:{},classList:{toggle(){}},
    setAttribute(){},addEventListener(){},closest(){return null},querySelectorAll(){return [];}});
  let api=null;
  const ctx={
    document:{querySelector:s=>els[s]||(els[s]=fake()),querySelectorAll:()=>[],addEventListener(){},body:fake(),createElement:fake},
    matchMedia:()=>({matches:true}),
    setTimeout:f=>{f();return 0;},clearTimeout(){},
    performance,console,navigator:{},
    __HEXWAR_EXPORT:a=>{api=a;},
  };
  ctx.globalThis=ctx; ctx.window=ctx;
  vm.createContext(ctx);
  vm.runInContext(code,ctx);
  if(!api) throw new Error('hexwar.html did not export its engine');
  return api;
};
