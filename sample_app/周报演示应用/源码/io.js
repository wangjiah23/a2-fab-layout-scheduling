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
    if(!['产品需求','新增路线','demands','new_routes'].includes(name))continue;
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
  const factory_id=obj.factory_id??DB.scene.factory_id, scope=obj.scope??'full';
  if(factory_id!==DB.scene.factory_id||scope!=='full')throw Error('当前场景导入支持 SMAT2022-LVHM_E 全厂及其既有设备位。');
  const horizon_min=numberField(obj.horizon_min??DB.scene.horizon_min,'计划期');
  const effective_time_fraction=numberField(obj.effective_time_fraction??DB.scene.effective_time_fraction,'有效产能系数');
  if(horizon_min!==DB.scene.horizon_min||effective_time_fraction!==DB.scene.effective_time_fraction)throw Error('此版本的计划期为 10,080 分钟，有效产能系数为 0.85。');
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
    if(step_type==='process'&&!familySet.has(stnfam)&&!Object.hasOwn(DB.aliases,stnfam))throw Error('设备族 '+(stnfam||'（空）')+' 无法映射到本厂。');
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
    return {part,route,demand_lots};
  }).sort((a,b)=>cmp(a.part,b.part));
  if(!demands.some(r=>r.demand_lots>0))throw Error('请至少为一个产品设置正需求。');
  return {factory_id,scope,horizon_min,effective_time_fraction,demands,new_routes};
}
function sceneKey(s){
  const n=v=>Number(Number(v).toPrecision(12));
  return JSON.stringify([s.factory_id,s.scope,s.horizon_min,s.effective_time_fraction,
    s.demands.map(r=>[r.part,r.route,n(r.demand_lots)]),s.new_routes.map(r=>[r.route,r.step,r.step_type,r.stnfam,n(r.time_min_per_lot)])]);
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
  else if(ext==='json'){const doc=JSON.parse((await file.text()).replace(/^\uFEFF/,''));obj=doc.scenario??doc;}
  else throw Error('场景支持 XLSX、CSV 和 JSON 格式。');
  return normalizeScene(obj);
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
  for(const k of ['reference_cost','optimized_cost','elapsed_sec','moved','lp_count']){
    if(raw[k]===undefined||raw[k]===null)continue;
    m[k]=numberField(raw[k],k);
    if(m[k]<0)throw Error('结果指标 '+k+' 必须为非负数。');
  }
  if(m.reference_cost>0&&m.optimized_cost!==undefined)m.cost_reduction_pct=(1-m.optimized_cost/m.reference_cost)*100;
  return m;
}
async function readResultFile(file){
  const ext=checkFile(file);let layout,metrics={},scene=null,method=null,source=file.name;
  if(ext==='json'){
    const data=JSON.parse((await file.text()).replace(/^\uFEFF/,''));
    if(data.schema && data.schema!=='fab-result-v1')throw Error('请选择 fab-result-v1 结果包或布局 CSV。');
    layout=Array.isArray(data)?data:data.layout;metrics=readMetrics(data.metrics);
    if(data.scenario){scene=normalizeScene(data.scenario);method=data.method;if(!['heuristic_lp','rl'].includes(method))throw Error('结果包方法需填写 heuristic_lp 或 rl。');}
  }else if(ext==='csv')layout=objects(csvRows(await file.text()));
  else if(ext==='zip'){
    const z=await JSZip.loadAsync(file);
    const paths=Object.keys(z.files).filter(p=>!z.files[p].dir&&!p.startsWith('__MACOSX/')&&p.split('/').pop().toLowerCase()==='layout.csv');
    if(paths.length!==1)throw Error('ZIP 内须包含一份 layout.csv，请只打包一次求解的输出。');
    const p=paths[0], base=p.slice(0,p.lastIndexOf('/')+1);
    layout=objects(csvRows(await zipEntry(z,p)));
    const meta=['e3_summary.json','summary.json'].map(f=>base+f).find(f=>z.file(f));
    if(meta){metrics=readMetrics(JSON.parse((await zipEntry(z,meta)).replace(/^\uFEFF/,'')));}
    source+=' / '+p;
  }else throw Error('结果导入支持布局 CSV、结果 JSON 及含 layout.csv 的 ZIP。');
  const after=normalizeLayout(layout);
  if(scene&&(after.length!==slotByName.size||after.some(r=>!slotByName.has(r.equipment_name))))throw Error('带全厂场景的结果包应包含本厂全部 1,068 台生产设备，各出现一次。局部布局可以单独导入 CSV。');
  return {kind:'external',title:'导入的设备布局',source,after,before:scene?DB.plant.slots:null,plant:!!scene,scene,method,metrics};
}
