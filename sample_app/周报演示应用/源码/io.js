'use strict';
const DB = JSON.parse(document.getElementById('archive').textContent);
const $ = id => document.getElementById(id);
const names = {heuristic_lp:'启发式 + LP', rl:'强化学习（RL）', milp:'局部 MILP'};
const slotByName = new Map(DB.plant.slots.map(r=>[r.equipment_name,r]));
const familySet = new Set(DB.plant.slots.map(r=>r.tool_group));
const areaByFamily = new Map(DB.plant.slots.map(r=>[r.tool_group,r.area]));
const familyAreaFold = new Map(DB.plant.slots.map(r=>[r.tool_group.toLowerCase(),r.area]));
function familyArea(group){return areaByFamily.get(group)||familyAreaFold.get(str(group).toLowerCase())||'其他';}
const collator = new Intl.Collator('zh-CN',{numeric:true});
const cmp = (a,b) => collator.compare(String(a),String(b));
const fmt = (n,p=0) => Number(n).toLocaleString('zh-CN',{maximumFractionDigits:p});
function el(tag,cl,content){
  const e=document.createElement(tag);
  if(cl)e.className=cl;
  if(content!==undefined)e.textContent=String(content);
  return e;
}
function btn(label,fn,cl='toolbtn'){
  const b=el('button',cl,label);b.type='button';b.addEventListener('click',fn);return b;
}
function option(value,label){const o=el('option','',label);o.value=value;return o;}
function b64Bytes(s){return Uint8Array.from(atob(s),c=>c.charCodeAt(0));}
function download(name,body,type='application/octet-stream'){
  const url=URL.createObjectURL(new Blob([body],{type}));
  const a=el('a');a.href=url;a.download=name;document.body.append(a);a.click();a.remove();
  setTimeout(()=>URL.revokeObjectURL(url),3000);
}
function asCsv(rows,headers){
  const esc=v=>'"'+String(v??'').replaceAll('"','""')+'"';
  return '\ufeff'+[headers,...rows.map(r=>headers.map(k=>r[k]))].map(r=>r.map(esc).join(',')).join('\r\n');
}
function csvRows(raw){
  const text=raw.replace(/^\uFEFF/,'');
  const sep=text.split(/\r?\n/,1)[0].includes('\t')?'\t':',';
  const rows=[];let row=[],field='',quoted=false;
  for(let i=0;i<text.length;i++){
    const c=text[i];
    if(quoted){
      if(c==='"' && text[i+1]==='"'){field+='"';i++;}
      else if(c==='"')quoted=false;
      else field+=c;
    }else if(c==='"' && !field){quoted=true;}
    else if(c===sep){row.push(field);field='';}
    else if(c==='\n'||c==='\r'){
      if(c==='\r' && text[i+1]==='\n')i++;
      row.push(field);rows.push(row);row=[];field='';
    }else field+=c;
  }
  if(quoted)throw Error('CSV 中有未闭合的引号。');
  row.push(field);rows.push(row);return rows;
}
function objects(rows){
  const present=rows.filter(r=>r.some(v=>String(v??'').trim()!==''));
  if(!present.length)return [];
  const headers=present[0].map(v=>String(v??'').replace(/^\uFEFF/,'').trim());
  return present.slice(1).map(row=>Object.fromEntries(headers.map((h,i)=>[h,row[i]??''])));
}
function str(v){return String(v??'').trim();}
function val(o,keys){for(const k of keys)if(Object.hasOwn(o,k)&&str(o[k])!=='')return o[k];return '';}
function numberField(v,label){
  if(str(v)===''||typeof v==='boolean'||!Number.isFinite(Number(v)))throw Error(label+'需要填写有限数值。');
  return Number(v);
}
function idField(v,label){const t=str(v);if(!t||t.length>180)throw Error(label+'需要填写有效编号。');return t;}
function xml(raw){
  const doc=new DOMParser().parseFromString(raw,'application/xml');
  if(doc.getElementsByTagName('parsererror').length)throw Error('工作簿 XML 格式有误。');
  return doc;
}
const tags=(e,name)=>Array.from(e.getElementsByTagNameNS('*',name));
function checkGeometryFields(keys){
  const geometry=keys.find(k=>/轨道|端口|机位|设备清单|厂房几何|^((plant|rails?|ports?|slots?|machines?|equipment|geometry|topology|layout)([_ -].*)?)$/i.test(k));
  if(geometry)throw Error('“'+geometry+'”涉及厂房几何。既有设备重新摆放请使用“导入起始布局”，变更机位或轨道需要建立新的工厂实例。');
}
async function zipEntry(zip,path,mode='string'){
  const entry=zip.file(path);
  if(!entry)throw Error('压缩包中缺少 '+path);
  if(entry._data && entry._data.uncompressedSize>64*1024*1024)throw Error('单个表格或数据文件超过 64 MB。');
  return entry.async(mode);
}
async function readExcel(file){
  const z=await JSZip.loadAsync(file);
  const book=xml(await zipEntry(z,'xl/workbook.xml'));
  const rel=xml(await zipEntry(z,'xl/_rels/workbook.xml.rels'));
  const relmap=new Map(tags(rel,'Relationship').map(x=>[x.getAttribute('Id'),x.getAttribute('Target')]));
  const shared=z.file('xl/sharedStrings.xml')?tags(xml(await zipEntry(z,'xl/sharedStrings.xml')),'si').map(s=>tags(s,'t').map(t=>t.textContent).join('')):[];
  const data={};
  for(const sheet of tags(book,'sheet')){
    const name=sheet.getAttribute('name').trim();
    if(!['产品需求','新增路线','demands','new_routes'].includes(name)){checkGeometryFields([name]);continue;}
    const rid=sheet.getAttributeNS('http://schemas.openxmlformats.org/officeDocument/2006/relationships','id');
    const target=relmap.get(rid);
    if(!target)throw Error('工作簿缺少数据表。');
    const path=new URL(target,'https://workbook.invalid/xl/').pathname.slice(1);
    const doc=xml(await zipEntry(z,path));
    const matrix=tags(doc,'row').map(row=>{
      const cells=[];
      for(const c of tags(row,'c')){
        const ref=(c.getAttribute('r')||'').match(/^([A-Z]+)/);
        if(!ref)continue;
        const index=Array.from(ref[1]).reduce((n,s)=>n*26+s.charCodeAt(0)-64,0)-1;
        if(index>128)continue;
        const type=c.getAttribute('t'), v=tags(c,'v')[0]?.textContent??'';
        cells[index]=type==='s'?shared[Number(v)]??'':type==='inlineStr'?tags(c,'t').map(t=>t.textContent).join(''):v;
      }
      return Array.from(cells,x=>x??'');
    });
    data[name]=objects(matrix);
  }
  const demands=data['产品需求']??data.demands;
  if(!demands)throw Error('找不到“产品需求”工作表，请使用场景模板。');
  return {demands,new_routes:data['新增路线']??data.new_routes??[]};
}
function normalizeScene(obj){
  if(!obj||typeof obj!=='object')throw Error('场景内容不完整。');
  const allowed=new Set(['factory_id','scope','horizon_min','effective_time_fraction','demands','new_routes','initial_layout']);
  const unknown=Object.keys(obj).filter(k=>!allowed.has(k));
  if(unknown.length)throw Error('场景不支持字段 '+unknown.join('、')+'。新增厂房、机位、端口或轨道需建立新的工厂实例。');
  const factory_id=obj.factory_id??DB.scene.factory_id, scope=obj.scope??'full';
  if(factory_id!==DB.scene.factory_id||scope!=='full')throw Error('当前场景导入支持 SMAT2022-LVHM_E 全厂及其既有设备位。');
  const horizon_min=numberField(obj.horizon_min??DB.scene.horizon_min,'计划期');
  const effective_time_fraction=numberField(obj.effective_time_fraction??DB.scene.effective_time_fraction,'有效产能系数');
  if(horizon_min<=0||effective_time_fraction<=0||effective_time_fraction>1)throw Error('计划期必须为正，有效产能系数须在 (0, 1]。');
  if(!Array.isArray(obj.demands)||!obj.demands.length)throw Error('产品需求表为空。');
  const newRows=obj.new_routes??[];
  if(!Array.isArray(newRows))throw Error('新增路线应为逐工序记录的列表。');
  const steps=new Set(),routeIds=new Set();
  const new_routes=newRows.map((r,i)=>{
    const route=idField(val(r,['route','路线编号']),'第 '+(i+1)+' 行路线');
    if(Object.hasOwn(DB.routes,route))throw Error('新增路线编号与内置路线 '+route+' 重复，请另取编号。');
    const step=numberField(val(r,['step','工序序号']),'工序序号');
    if(!Number.isInteger(step)||step<1||steps.has(JSON.stringify([route,step])))throw Error('工序序号须为同一路线内唯一的正整数。');
    steps.add(JSON.stringify([route,step]));routeIds.add(route);
    const t=str(val(r,['step_type','工序类型'])).toLowerCase();
    const step_type=['process','加工','加工工序'].includes(t)?'process':['delay','等待','延迟'].includes(t)?'delay':null;
    if(!step_type)throw Error('工序类型请填写 process（加工）或 delay（等待）。');
    const stnfam=str(val(r,['stnfam','设备族编号','tool_group']));
    if(step_type==='process'&&!familyAreaFold.has(stnfam.toLowerCase())&&!Object.hasOwn(DB.aliases,stnfam))throw Error('设备族 '+(stnfam||'（空）')+' 无法映射到本厂。');
    const time_min_per_lot=numberField(val(r,['time_min_per_lot','单批用时（分钟）','单批用时(分钟)']),'单批用时');
    if(time_min_per_lot<0)throw Error('单批用时应大于或等于零。');
    return {route,step,step_type,stnfam,time_min_per_lot};
  }).sort((a,b)=>cmp(a.route,b.route)||a.step-b.step);
  for(const route of routeIds)if(!new_routes.some(r=>r.route===route&&r.step_type==='process'))throw Error('新增路线 '+route+' 至少需要一道加工工序。');
  const partIds=new Set();
  const demands=obj.demands.map((r,i)=>{
    const part=idField(val(r,['part','产品编号','product_id']),'第 '+(i+1)+' 行产品');
    if(partIds.has(part))throw Error('产品编号 '+part+' 重复。');
    partIds.add(part);
    const route=idField(val(r,['route','路线编号','route_id']),'路线');
    if(!Object.hasOwn(DB.routes,route)&&!routeIds.has(route))throw Error('找不到路线 '+route+'。请在“新增路线”中填写可映射的工序。');
    const demand_lots=numberField(val(r,['demand_lots','需求量（lot）','需求量(lot)','需求量','lots']),'产品 '+part+' 的需求量');
    if(demand_lots<0)throw Error('产品需求量应大于或等于零。');
    const pieces=val(r,['pieces_per_lot','每批片数','PIECES']);
    const pieces_per_lot=pieces===''?25:numberField(pieces,'每批片数');
    if(!Number.isInteger(pieces_per_lot)||pieces_per_lot<1)throw Error('每批片数须为正整数。');
    return {part,route,demand_lots,...(pieces_per_lot===25?{}:{pieces_per_lot})};
  }).sort((a,b)=>cmp(a.part,b.part));
  if(!demands.some(r=>r.demand_lots>0))throw Error('请至少为一个产品设置正需求。');
  const initial_layout=obj.initial_layout==null?null:initialRows(obj.initial_layout);
  return {factory_id,scope,horizon_min,effective_time_fraction,demands,new_routes,...(initial_layout?{initial_layout}:{})};
}
function sceneKey(s){
  const n=v=>Number(Number(v).toPrecision(12));
  return JSON.stringify([s.factory_id,s.scope,s.horizon_min,s.effective_time_fraction,
    s.demands.map(r=>[r.part,r.route,n(r.demand_lots),r.pieces_per_lot??25]),s.new_routes.map(r=>[r.route,r.step,r.step_type,r.stnfam,n(r.time_min_per_lot)]),s.initial_layout??null]);
}
function routeSteps(s,route){
  return Object.hasOwn(DB.routes,route)?DB.routes[route]:s.new_routes.filter(r=>r.route===route).map(r=>[r.step,r.step_type,r.stnfam,r.time_min_per_lot,'新增路线']);
}
function checkFile(file){
  if(!file||!file.size)throw Error('文件为空。');
  if(file.size>48*1024*1024)throw Error('此演示版支持 48 MB 以内的单个文件。');
  return file.name.toLowerCase().split('.').pop();
}
async function readSceneFile(file){
  const ext=checkFile(file);let obj;
  if(ext==='xlsx')obj=await readExcel(file);
  else if(ext==='csv')obj={demands:objects(csvRows(await file.text()))};
  else if(ext==='txt'){
    const data=await file.arrayBuffer();let raw;
    try{raw=new TextDecoder('utf-8',{fatal:true}).decode(data);}catch{raw=new TextDecoder('gbk').decode(data);}
    obj=orderScene(raw);
  }else if(ext==='json'){
    const doc=JSON.parse((await file.text()).replace(/^\uFEFF/,''));
    if(doc?.schema&&doc.schema!=='fab-job-v1')throw Error('请选择生产场景或 fab-job-v1 任务。布局结果可从“导入布局结果”打开。');
    if(doc?.scenario){
      const unknown=Object.keys(doc).filter(k=>!['schema','method','scenario'].includes(k));
      if(unknown.length){checkGeometryFields(unknown);throw Error('任务中存在未支持的字段 '+unknown.join('、')+'。');}
    }
    obj=doc?.scenario??doc;
  }
  else throw Error('场景支持 XLSX、CSV、JSON 和 SMT2020 order.txt。');
  return normalizeScene(obj);
}
function orderScene(raw){
  const rows=csvRows(raw),required=['LOT','PART','PRIOR','PIECES','START','RDIST','REPEAT','RUNITS','RPT#','LOTSPERRPT','DUE','ORDER','HOTLOT'];
  if(!raw.split(/\r?\n/,1)[0].includes('\t')||required.some(field=>!rows[0].includes(field)))throw Error('order.txt 须为含标准 SMT2020 表头的制表符订单文件。');
  const catalog=new Map(DB.scene.demands.map(d=>[d.part,d.route])),parts=new Map();
  for(const row of objects(rows)){
    const part=str(row.PART),pieces=numberField(row.PIECES,'PIECES'),repeat=numberField(row.REPEAT,'REPEAT'),lots=numberField(row.LOTSPERRPT,'LOTSPERRPT');
    if(!catalog.has(part))throw Error('订单中产品 '+part+' 未关联内置路线。新增产品可用场景模板指定路线。');
    if(str(row.RUNITS).toLowerCase()!=='min'||repeat<=0||lots<0||!Number.isInteger(pieces)||pieces<1)throw Error('订单到达间隔须为正分钟数，批量非负，每批片数为正整数。');
    const d=parts.get(part)||{part,route:catalog.get(part),demand_lots:0,pieces_per_lot:pieces};
    if(d.pieces_per_lot!==pieces)throw Error('同一产品的每批片数须一致。');
    d.demand_lots+=0.25*DB.scene.horizon_min*lots/repeat;parts.set(part,d);
  }
  return {demands:[...parts.values()]};
}
function initialRows(rows){
  const data=normalizeLayout(rows);
  if(data.length!==slotByName.size||data.some(d=>!slotByName.has(d.equipment_name)))throw Error('起始布局须包含本厂全部 1,068 台设备，每台一次。');
  return data.map(r=>{
    const row={equipment_name:r.equipment_name,x:r.x,y:r.y};
    for(const k of ['width','height','bay','tool_group'])if(r[k])row[k]=r[k];
    for(const k of ['original_slot','new_slot'])if(r[k]!=='')row[k]=numberField(r[k],k);
    return row;
  }).sort((a,b)=>cmp(a.equipment_name,b.equipment_name));
}
async function readInitial(file){
  const ext=checkFile(file);
  if(ext==='csv')return initialRows(objects(csvRows(await file.text())));
  if(ext==='json'){
    const v=JSON.parse((await file.text()).replace(/^\uFEFF/,''));
    if(v&&!Array.isArray(v))checkGeometryFields(Object.keys(v).filter(k=>k!=='layout'));
    return initialRows(Array.isArray(v)?v:v?.layout);
  }
  throw Error('起始布局支持 layout.csv 或含 layout 的 JSON。');
}
function normalizeLayout(rows){
  if(!Array.isArray(rows)||!rows.length||rows.length>50000)throw Error('布局表应包含 1 至 50,000 行设备。');
  const ids=new Set();
  return rows.map((r,i)=>{
    const prefix='布局第 '+(i+1)+' 行';
    const equipment_name=idField(val(r,['equipment_name','equipment_id','NAME','设备编号','设备名称']),prefix+'设备');
    if(ids.has(equipment_name))throw Error('布局设备编号 '+equipment_name+' 重复。');
    ids.add(equipment_name);
    const x=numberField(val(r,['x','POSITION_X','position_x','X','横坐标']),prefix+' x 坐标');
    const y=numberField(val(r,['y','POSITION_Y','position_y','Y','纵坐标']),prefix+' y 坐标');
    const dims={};
    for(const [k,keys] of [['width',['width','WIDTH','宽度']],['height',['height','HEIGHT','高度']]]){
      const v=val(r,keys);
      if(v!==''){dims[k]=numberField(v,prefix+' '+k);if(dims[k]<=0)throw Error(prefix+'尺寸必须大于零。');}
    }
    const tool_group=str(val(r,['tool_group','TOOL_GROUP','设备族','设备组'])),bay=str(val(r,['bay','BAY','Bay','Bay编号']));
    return {equipment_name,tool_group,bay,x,y,...dims,area:str(r.area)||familyArea(tool_group),
      original_slot:val(r,['original_slot','原机位']),new_slot:val(r,['new_slot','新机位'])};
  });
}
function readMetrics(raw){
  if(!raw||typeof raw!=='object')return {};
  const m={};
  for(const k of ['reference_cost','optimized_cost','policy_cost','elapsed_sec','moved','lp_count','global_relative_gap_bound','primal_infeasibility']){
    if(raw[k]===undefined||raw[k]===null)continue;
    m[k]=numberField(raw[k],k);
    if(m[k]<0)throw Error('结果指标 '+k+' 必须为非负数。');
  }
  if(m.reference_cost>0&&m.optimized_cost!==undefined)m.cost_reduction_pct=(1-m.optimized_cost/m.reference_cost)*100;
  return m;
}
async function readResultFile(file){
  const ext=checkFile(file);let layout,metrics={},scene=null,method=null,source=file.name,trace=null,policy=null,reference=null,selection=null;
  if(ext==='json'){
    const data=JSON.parse((await file.text()).replace(/^\uFEFF/,''));
    if(data.schema && data.schema!=='fab-result-v1')throw Error('请选择 fab-result-v1 结果包或布局 CSV。');
    layout=Array.isArray(data)?data:data.layout;metrics=readMetrics(data.metrics);trace=data.replay;
    if(data.policy_layout)policy=normalizeLayout(data.policy_layout);
    if(data.reference_layout)reference=normalizeLayout(data.reference_layout);
    selection=data.selection;
    if(data.scenario){scene=normalizeScene(data.scenario);method=data.method;if(!['heuristic_lp','rl'].includes(method))throw Error('结果包方法需填写 heuristic_lp 或 rl。');}
  }else if(ext==='csv')layout=objects(csvRows(await file.text()));
  else if(ext==='zip'){
    const z=await JSZip.loadAsync(file);
    const paths=Object.keys(z.files).filter(p=>!z.files[p].dir&&!p.startsWith('__MACOSX/')&&p.split('/').pop().toLowerCase()==='layout.csv');
    if(paths.length!==1)throw Error('ZIP 内须包含一份 layout.csv，请只打包一次求解的输出。');
    const p=paths[0], base=p.slice(0,p.lastIndexOf('/')+1);
    if(z.file(base+'result.json')){
      const raw=await zipEntry(z,base+'result.json');
      let bundled;try{bundled=JSON.parse(raw);}catch{}
      if(bundled?.schema==='fab-result-v1')return readResultFile(new File([raw],'result.json',{type:'application/json'}));
    }
    layout=objects(csvRows(await zipEntry(z,p)));
    const meta=['e3_summary.json','summary.json'].map(f=>base+f).find(f=>z.file(f));
    if(meta){metrics=readMetrics(JSON.parse((await zipEntry(z,meta)).replace(/^\uFEFF/,'')));}
    source+=' / '+p;
  }else throw Error('结果导入支持布局 CSV、结果 JSON 及含 layout.csv 的 ZIP。');
  const after=normalizeLayout(layout);
  if(scene&&(after.length!==slotByName.size||after.some(r=>!slotByName.has(r.equipment_name))))throw Error('带全厂场景的结果包应包含本厂全部 1,068 台生产设备，各出现一次。局部布局可以单独导入 CSV。');
  let replay=null;
  if(trace!=null){
    if(method!=='rl'||!scene||!Array.isArray(trace.slots)||trace.slots.length!==after.length)throw Error('推理回放与全厂设备数不一致。');
    const bySlot=new Map((policy||after).map(row=>[Number(row.new_slot),row])),n=after.length;
    if(policy&&(policy.length!==n||policy.some(r=>!slotByName.has(r.equipment_name))))throw Error('候选布局的设备清单不完整。');
    if(bySlot.size!==n||new Set(trace.slots).size!==n||trace.slots.some(i=>!Number.isInteger(i)||i<0||i>=n||!bySlot.has(i)))throw Error('回放必须按真实动作逐一填满每个槽位。');
    const automatic=trace.automatic_steps||[];
    if(!Array.isArray(automatic)||automatic.some(i=>!Number.isInteger(i)||i<0||i>=n))throw Error('回放的自动放置步号无效。');
    if(Number.isInteger(trace.cycle))source='PPO · 第 '+trace.cycle+' 轮检查点';
    replay={rows:trace.slots.map(slot=>bySlot.get(slot)),automatic:new Set(automatic),
      cycle:Number.isInteger(trace.cycle)?trace.cycle:null,checkpoint:typeof trace.checkpoint==='string'?trace.checkpoint:''};
  }
  if(reference&&(reference.length!==after.length||reference.some(r=>!slotByName.has(r.equipment_name))))throw Error('输入布局的设备清单不完整。');
  const before=reference||(scene?.initial_layout?scene.initial_layout.map(r=>({...slotByName.get(r.equipment_name),...r})):scene?DB.plant.slots:null);
  return {kind:'external',title:replay?(selection==='initial'?'本次推荐保留输入布局':'PPO 推理布局'):'导入的设备布局',source,replay,after,before,policy,selection,plant:!!scene,scene,method,metrics};
}
