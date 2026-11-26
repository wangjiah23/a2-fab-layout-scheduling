'use strict';
const baseline=normalizeScene(DB.scene),changed=normalizeScene(DB.changed);
const baselineKey=sceneKey(baseline),externalRegistry=new Map();
const cases=new Map([
  ['baseline',{label:'LVHM · 全厂基准',scene:baseline}],
  ['local',{label:'A2 / A3 · 局部验证',local:true,scene:baseline}],
  ['changed',{label:'LVHM · 产品配比变化',scene:changed}]
]);
const methodsFor=c=>c.standalone?[]:c.local?['milp']:['heuristic_lp','rl'];
const methodLabel=id=>Object.hasOwn(names,id)?names[id]:'外部求解结果';
const grade=n=>n===undefined?'—':Number(n).toFixed(2);
const initialResult={
  kind:'e3',title:'全厂设备布局',method:'heuristic_lp',source:'第七周 · E3 全厂启发式实验',
  plant:true,scene:baseline,before:DB.plant.slots,after:DB.e3.layout.map(r=>({...r,area:slotByName.get(r.equipment_name).area})),metrics:DB.e3.metrics
};
const rlResult={kind:'rl',title:'强化学习布局回放',method:'rl',source:'期中汇报 · 第 19、20 页',scene:baseline};
function localRow(s,original){
  const group=original?s.original_group:s.optimized_group;
  return {equipment_name:original?s.old_equipment:s.new_equipment,tool_group:group,area:familyArea(group),
    bay:s.bay,x:s.position_x,y:s.position_y,width:s.width,height:s.height,slot_name:s.slot_name};
}
const ls=DB.local.summary;
const localResult={kind:'local',title:'A2 / A3 区域布局',method:'milp',local:true,plant:true,scene:baseline,
  source:'第六周 · A2 / A3 区域 MILP',before:DB.local.slots.map(s=>localRow(s,true)),after:DB.local.slots.map(s=>localRow(s,false)),
  metrics:{reference_cost:ls.reference_affected_cost,optimized_cost:ls.objective,cost_reduction_pct:ls.improvement_local*100,elapsed_sec:ls.solver_seconds}};
let currentCase='baseline',method='heuristic_lp',result=initialResult,tab='map',cleanup=null,focused='',counter=0,importSerial=0;
const chosen=()=>cases.get(currentCase);
function registryId(s,m){return m+'|'+sceneKey(s);}
function findResult(c,m){
  if(c.standalone)return c.result;
  if(c.local)return m==='milp'?localResult:null;
  const bound=externalRegistry.get(registryId(c.scene,m));if(bound)return bound;
  if(sceneKey(c.scene)===baselineKey)return m==='heuristic_lp'?initialResult:m==='rl'?rlResult:null;
  return null;
}
function message(text='',error=false){const box=$('message');box.textContent=text;box.classList.toggle('error',error);box.hidden=!text;}
function modal(title,content){
  $('dialogTitle').textContent=title;$('dialogBody').replaceChildren();
  if(typeof content==='function')content($('dialogBody'));else $('dialogBody').append(content);
  if(!$('dialog').open)$('dialog').showModal();
}
function section(body,title,text){body.append(el('h3','',title),el('p','',text));}
function fact(key,value){const row=el('div');row.append(el('span','',key),el('b','',value));return row;}
function firstTab(r){return !r?'pending':r.kind==='rl'?'replay':'map';}
function addCase(key,c){
  cases.set(key,c);$('cases').append(option(key,c.label));return key;
}
function caseForScene(s,label){
  const key=sceneKey(s);
  for(const [id,c] of cases)if(!c.local&&!c.standalone&&sceneKey(c.scene)===key)return id;
  return addCase('user'+(++counter),{label:'导入 · '+label,scene:s});
}
function chooseCase(id,requested=null){
  currentCase=id;$('cases').value=id;const c=chosen(),possible=methodsFor(c);
  method=possible.includes(requested)?requested:possible.includes(method)?method:possible[0]||null;
  result=c.standalone?c.result:null;tab=firstTab(result);focused='';message();render();
}
function showResult(){
  const r=findResult(chosen(),method);
  if(!r){message('此场景与方法尚无归档结果。可导出任务，在外部运行算法后导入结果包。');return;}
  result=r;focused='';tab=firstTab(r);message('已载入'+(r.kind==='external'?'外部导入':'项目归档')+'结果。');render();
}
function switches(){
  const c=chosen(),host=$('methods');host.replaceChildren();
  for(const m of methodsFor(c)){
    const available=!!findResult(c,m),row=el('label','method'+(method===m?' active':''));
    const radio=el('input');radio.type='radio';radio.name='method';radio.checked=method===m;radio.value=m;
    radio.onchange=()=>{method=m;result=null;tab='pending';focused='';message();render();};
    row.append(radio,el('span','',methodLabel(m)),el('small','',available?'已归档':'待求解'));host.append(row);
  }
  if(c.standalone)host.append(el('p','muted','独立布局文件'));
  const ready=!!findResult(c,method);
  $('loadResult').disabled=!ready||!!c.standalone;$('loadResult').textContent=ready?'查看离线结果  →':'暂无离线结果';
  $('rlRemote').hidden=method!=='rl'||c.local||c.standalone||sceneKey(c.scene)!==baselineKey;
  $('exportJob').hidden=!!(c.standalone||c.local);$('showInputs').disabled=!!c.standalone;
  $('inputFacts').replaceChildren();
  if(c.standalone){$('inputFacts').append(fact('文件类型','独立布局'),fact('设备数',fmt(c.result.after.length)));return;}
  if(c.local){$('inputFacts').append(fact('研究区域','A2 / A3'),fact('局部设备位','48'),fact('区域加工工序','59'));return;}
  const f=sceneFacts(c.scene);$('inputFacts').append(fact('计划周期','7 天'),fact('产品 / 路线数',c.scene.demands.length+' / '+f.routes),fact('需求合计',fmt(f.lots,3)+' lot'));
}
function metric(label,value,unit='',featured=false){
  const m=el('div','metric'+(featured?' featured':'')),b=el('b','',value);
  if(unit)b.append(el('small','',unit));m.append(el('label','',label),b);return m;
}
function metrics(){
  const box=$('kpis'),c=chosen();box.replaceChildren();
  if(!result){
    const f=sceneFacts(c.scene);box.append(metric(c.local?'局部设备位':'固定生产设备',c.local?'48':'1,068','台',true),metric('产品需求',fmt(f.lots,3),'lot'),metric(c.local?'区域加工工序':'路线加工工序',fmt(c.local?59:f.steps),'道'),metric('有效产能系数','85','%'));return;
  }
  const r=result,m=r.metrics||{};
  if(r.kind==='rl'){
    const get=e=>DB.rl.curve.find(p=>p[0]===e)[1].toFixed(2);
    box.append(metric('第 95 轮成本降幅',get(95),'%',true),metric('第 295 轮成本降幅',get(295),'%'),metric('评估记录',DB.rl.curve.length,'次'),metric('全厂设备','1,068','台'));return;
  }
  if(r.local){box.append(metric('区域相关成本降幅',m.cost_reduction_pct.toFixed(2),'%',true),metric('区域机位','48','个'),metric('设备组变化的机位','6','个'),metric('区域加工工序','59','道'));return;}
  if(m.cost_reduction_pct!==undefined){
    box.append(metric('运输成本降幅',grade(m.cost_reduction_pct),'%',true),metric('布局设备',fmt(r.after.length),'台'));
    if(m.moved!==undefined)box.append(metric('调整机位的设备',fmt(m.moved),'台'));
    if(m.elapsed_sec!==undefined)box.append(metric('离线求解耗时',grade(m.elapsed_sec),'秒'));
  }else{
    box.append(metric('导入设备',fmt(r.after.length),'台',true),metric('文件中的 Bay',new Set(r.after.map(e=>e.bay).filter(Boolean)).size,'个'),
      metric('文件中的设备组',new Set(r.after.map(e=>e.tool_group).filter(Boolean)).size,'组'));
  }
}
function resultTabs(){
  const c=chosen(),r=result;if(!r)return [['pending','场景状态'],['inputs','输入数据']];
  if(r.kind==='rl')return [['replay','布局回放'],['chart','训练评估'],['inputs','输入数据']];
  const list=[['map','布局视图'],['table','设备明细']];
  if(r.local)list.push(['flow','加工与物流']);
  if(r.metrics?.reference_cost!==undefined||r.metrics?.optimized_cost!==undefined)list.push(['metrics','指标对照']);
  if(!c.standalone)list.push(['inputs','输入数据']);return list;
}
function showMetrics(host,r){
  const m=r.metrics,local=!!r.local,top=el('div','viewtop');
  top.append(el('b','toolbar-title',local?'区域相关运输成本':'全厂运输成本'),el('span','preset-tag','模型目标值'));host.append(top);
  const bars=el('div','costbars');
  for(const [label,k] of [['参考布局','reference_cost'],['优化布局','optimized_cost']]){
    const row=el('div','barrow'),track=el('div','bartrack'),bar=el('i'),v=el('b','',m[k]===undefined?'未提供':fmt(m[k],2));
    bar.style.width=m[k]===undefined?'0':Math.max(0,m[k]/(Math.max(m.reference_cost||0,m.optimized_cost||0)||1)*100)+'%';
    if(k==='optimized_cost')bar.className='opt';track.append(bar);row.append(el('span','',label),track,v);bars.append(row);
  }host.append(bars);
  const rows=[];
  if(m.reference_cost!==undefined&&m.optimized_cost!==undefined)rows.push(['运输成本减少量',fmt(m.reference_cost-m.optimized_cost,2),'使用实验记录中的原始目标单位']);
  if(m.cost_reduction_pct!==undefined)rows.push(['相对参考布局降幅',grade(m.cost_reduction_pct)+'%',local?'仅统计受区域决策影响的成本':'相对该次实验的参考布局']);
  if(m.lp_count!==undefined)rows.push(['LP 调用次数',fmt(m.lp_count)+' 次','本次归档的运行统计']);
  if(m.elapsed_sec!==undefined)rows.push(['离线求解耗时',grade(m.elapsed_sec)+' 秒',local?'MILP 主求解阶段':'来自本次运行记录']);
  host.append(simpleTable(['指标','数值','口径'],rows));
  if(r.kind==='e3')host.append(el('p','details-caption','工厂轨道显示可用运输网络。本次 E3 归档含最终布局和汇总指标；逐工序加工与运输分配可在 A2 / A3 局部案例查看。'));
}
function showTab(id,focus=''){
  if(cleanup){cleanup();cleanup=null;}
  tab=id;focused=focus;
  const c=chosen(),r=result,tabHost=$('tabs'),host=$('view');tabHost.replaceChildren();host.replaceChildren();
  for(const [k,label] of resultTabs()){
    const item=btn(label,()=>showTab(k),k===id?'on':'');item.setAttribute('aria-current',k===id?'page':'false');tabHost.append(item);
  }
  const expand=btn('放大视图',toggleExpanded,'expand-view');expand.id='expandView';expand.title='展开以便投影，按 Esc 返回';tabHost.append(expand);syncExpanded();
  if(id==='map')cleanup=makeMap(host,r,focus);
  else if(id==='table')layoutTable(host,r,n=>showTab('map',n));
  else if(id==='replay')cleanup=replayView(host);
  else if(id==='chart')cleanup=chartView(host);
  else if(id==='flow')localFlow(host);
  else if(id==='metrics')showMetrics(host,r);
  else if(id==='inputs')inputView(host,c.scene,!!c.local);
  else{
    const ready=!!findResult(c,method);
    const box=callout(host,ready?'这组输入已有离线结果':'这组输入尚无求解记录',
      ready?'已匹配当前工厂、产品需求与工艺路线。载入后可查看此次实验的方案。':'可将该场景导出给线下求解程序，获得结果后通过“导入布局结果”查看。');
    box.append(btn(ready?'载入并展示结果':'导出本场景任务',ready?showResult:exportJob));
    inputView(host,c.scene,!!c.local);
  }
}
function syncExpanded(){
  const b=$('expandView');if(!b)return;
  const enlarged=document.body.classList.contains('enlarged');
  b.textContent=enlarged?'退出放大  ×':'放大视图  ⤢';b.setAttribute('aria-pressed',String(enlarged));
}
function toggleExpanded(){document.body.classList.toggle('enlarged');syncExpanded();}
document.addEventListener('keydown',e=>{if(e.key==='Escape'&&document.body.classList.contains('enlarged')&&!$('dialog').open)toggleExpanded();});
function render(){
  const c=chosen(),r=result;switches();metrics();
  $('context').textContent=c.standalone?'外部文件  /  布局查看':('SMAT2022 · LVHM_E  /  '+(c.local?'区域验证':'全厂规划'));
  $('title').textContent=r?r.title:c.local?'A2 / A3 区域场景':c.label.replace(/^LVHM · /,'').replace(/^导入 · /,'')+' · 场景输入';
  $('source').hidden=!r;
  $('footerContext').textContent=r?('来源 · '+r.source):('当前场景 · '+c.label+'  /  '+methodLabel(method));
  showTab(tab,focused);
}
function exportJob(){
  const c=chosen();if(!c.scene||c.local)return;
  const body={schema:'fab-job-v1',method,scenario:c.scene};
  download('离线任务_'+method+'.json',JSON.stringify(body,null,2),'application/json');
  message('已导出当前场景与方法。离线任务采用固定工厂的既有路线编号，新增路线随任务提供。');
}
function guide(){
  modal('使用指南',body=>{
    section(body,'现场展示','打开后为全厂启发式 + LP 归档。可先看“指标对照”，再切换到强化学习，点击“查看离线结果”，播放设备逐步放置的过程。');
    section(body,'运行全厂 PPO','选择全厂基准与强化学习，点“远程训练与推理”。在本机启动仓库中的 RL 控制台后，可连接服务器、部署环境并执行任务。推理完成后点“在工作台对照布局”。');
    section(body,'切换生产输入','通过“导入生产场景”选择 Excel、CSV 或 JSON。模板包含产品需求和新增路线。当前工厂固定为 SMAT2022 LVHM_E，计划期为 7 天。');
    body.append(btn('下载基准场景 XLSX',()=>downloadTemplate('baseline')),btn('下载配比变化 XLSX',()=>downloadTemplate('changed')));
    section(body,'查看外部布局','CSV 至少提供 equipment_name、x、y 三列。可补充 tool_group、bay、width、height。含 layout.csv 和同目录 summary.json 或 e3_summary.json 的 ZIP 也可导入。');
    section(body,'关联新的离线计算','JSON 结果包用 schema=fab-result-v1，附导出任务中的 scenario、method，以及本厂完整的 layout 和可选 metrics。导入后，本次打开应用期间可以通过同一场景与方法重新载入。');
    body.append(btn('下载示例布局 CSV',()=>download('03_启发式全厂布局.csv',b64Bytes(DB.rawCsv),'text/csv')),btn('下载示例结果包 JSON',()=>download('04_启发式离线结果包.json',JSON.stringify({schema:'fab-result-v1',scenario:baseline,method:'heuristic_lp',metrics:DB.e3.metrics,layout:DB.e3.layout}),'application/json')));
  });
}
function sourceInfo(){
  const r=result;if(!r)return;
  modal('结果来源',body=>{
    section(body,r.source,r.kind==='e3'?'week7_heuristic.zip 内的 e3_global/layout.csv 与 e3_summary.json。参考设备位及轨道来自 SMAT2022 LVHM_E。':
      r.kind==='rl'?'A2_期中汇报.pptx 第 19 页原始图表数据与第 20 页第 95 轮布局动画。记录按原顺序抽帧，支持逐段播放。':
      r.local?'week6/MILP交互展示.html 内嵌的机位、加工分配、边界物流及求解汇总。':(r.scene?'文件包含场景、求解方法与完整设备布局，已在本次会话中关联。':'按文件内的设备坐标独立绘制。'));
    section(body,'求解方式','归档案例的计算在软件外完成，工作台负责核对输入、加载已有输出并可视化。');
    if(r.kind==='e3')section(body,'指标口径','运输成本按原实验的 LP 目标计量，耗时为该次离线求解记录。布局与固定轨道可逐 Bay 查看。');
    if(r.kind==='rl')section(body,'记录范围','评估曲线覆盖第 0 至 295 轮，回放取自第 95 轮。此次归档提供过程图像与评估数据。');
    if(r.local)section(body,'区域边界','以 E1 全厂参考解固定区外边界流量。降幅以受区域调整影响的运输成本为分母。');
  });
}
function downloadTemplate(k){download(k==='changed'?'02_产品配比变化.xlsx':'01_基准生产场景.xlsx',b64Bytes(DB.templates[k]),'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet');}
async function importScene(file){
  if(!file)return;const seq=++importSerial;
  try{
    const s=await readSceneFile(file);if(seq!==importSerial)return;
    const id=caseForScene(s,file.name);chooseCase(id,method);
    message('已导入 '+file.name+'。'+(findResult(chosen(),method)?'场景内容与归档匹配，可查看离线结果。':'当前输入尚无该方法的结果，可导出离线任务。'));
  }catch(e){if(seq===importSerial)message('导入场景失败。'+e.message,true);}
}
async function importResult(file){
  if(!file)return;const seq=++importSerial;
  try{
    const r=await readResultFile(file);if(seq!==importSerial)return;
    if(r.scene){
      const id=caseForScene(r.scene,file.name);externalRegistry.set(registryId(r.scene,r.method),r);
      chooseCase(id,r.method);result=r;tab='map';render();message('已导入 '+file.name+'，完整布局已与该场景和方法关联。');
    }else{
      const id=addCase('file'+(++counter),{standalone:true,label:'布局 · '+file.name,result:r});
      chooseCase(id);message('已导入 '+file.name+'，展示文件内 '+fmt(r.after.length)+' 台设备的坐标。');
    }
  }catch(e){if(seq===importSerial)message('导入结果失败。'+e.message,true);}
}
const localConsole=['http:','https:'].includes(location.protocol)&&['127.0.0.1','localhost','[::1]'].includes(location.hostname);
$('rlRemote').onclick=()=>{
  if(localConsole){location.assign('/rl/');return;}
  message('请先在仓库 rl/console 目录运行 uv run --locked --python 3.11 python app.py，再打开本机控制台。');
  window.open('http://127.0.0.1:7860/rl/','_blank','noopener');
};
async function loadRemoteResult(){
  if(!localConsole)return;
  const token=new URLSearchParams(location.search).get('rl_result');
  if(!token)return;
  if(!/^[a-f0-9]{40}$/.test(token)){message('推理结果链接无效。',true);return;}
  try{
    const response=await fetch('/rl-results/'+token,{cache:'no-store'});
    if(!response.ok)throw Error('请在远程运行页重新打开这次推理，或导入已下载的结果包。');
    await importResult(new File([await response.blob()],'E3-PPO-fab-result.json',{type:'application/json'}));
  }catch(error){message('读取推理结果失败。'+error.message,true);}
}
$('cases').onchange=()=>{importSerial++;chooseCase($('cases').value);};
$('loadResult').onclick=showResult;$('exportJob').onclick=exportJob;$('template').onclick=()=>downloadTemplate('baseline');
$('guide').onclick=guide;$('source').onclick=sourceInfo;$('closeDialog').onclick=()=>$('dialog').close();
$('dialog').onclick=e=>{if(e.target===$('dialog')){const b=$('dialog').getBoundingClientRect();if(e.clientX<b.left||e.clientX>b.right||e.clientY<b.top||e.clientY>b.bottom)$('dialog').close();}};
$('showInputs').onclick=()=>showTab('inputs');
$('uploadScene').onclick=()=>$('sceneFile').click();$('uploadResult').onclick=()=>$('resultFile').click();
$('sceneFile').onchange=e=>{const f=e.target.files[0];e.target.value='';importScene(f);};
$('resultFile').onchange=e=>{const f=e.target.files[0];e.target.value='';importResult(f);};
const drop=$('uploadResult');drop.ondragover=e=>{e.preventDefault();drop.classList.add('drag');};drop.ondragleave=()=>drop.classList.remove('drag');
drop.ondrop=e=>{e.preventDefault();drop.classList.remove('drag');importResult(e.dataTransfer.files[0]);};
render();
loadRemoteResult();
