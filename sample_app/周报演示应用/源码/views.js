'use strict';
function simpleTable(headers,rows){
  const wrapper=el('div','tablewrap'),table=el('table'),thead=el('thead'),tr=el('tr'),body=el('tbody');
  for(const h of headers)tr.append(el('th','',h));
  thead.append(tr);
  for(const row of rows){const t=el('tr');for(const v of row)t.append(el('td','',v));body.append(t);}
  table.append(thead,body);wrapper.append(table);return wrapper;
}
function callout(host,title,text){
  const d=el('div','callout');d.append(el('h3','',title),el('p','',text));host.append(d);return d;
}
function sceneFacts(s){
  const routes=[...new Set(s.demands.filter(d=>d.demand_lots>0).map(d=>d.route))];
  const steps=routes.flatMap(r=>routeSteps(s,r));
  return {steps:steps.filter(r=>r[1]==='process').length,delays:steps.filter(r=>r[1]==='delay').length,routes:routes.length,lots:s.demands.reduce((a,r)=>a+r.demand_lots,0)};
}
function inputView(host,s,local=false){
  if(local)callout(host,'A2 / A3 的区域输入','固定区域边界供需及区外流量，选取原全厂基准输入中的 48 个机位、4 个设备组和 59 道区域加工工序。下表为其全厂母场景。');
  const t=el('div','table-tools'),title=el('b','table-title','生产需求与工艺路线'),chooser=el('select');
  chooser.setAttribute('aria-label','查看产品工艺路线');chooser.append(option('','全部产品 · 需求汇总'));
  for(const d of s.demands)chooser.append(option(d.part,d.part+' / '+d.route+' · 工序明细'));
  t.append(title,chooser);host.append(t);const area=el('div');host.append(area);
  function draw(){
    area.replaceChildren();
    const d=s.demands.find(d=>d.part===chooser.value);
    if(!d){
      area.append(simpleTable(['产品','工艺路线','计划需求（lot）','每批片数','加工工序','等待工序'],s.demands.map(d=>{
        const steps=routeSteps(s,d.route);return [d.part,d.route,fmt(d.demand_lots,6),d.pieces_per_lot??25,steps.filter(x=>x[1]==='process').length,steps.filter(x=>x[1]==='delay').length];
      })));
      area.append(el('p','details-caption','选择一个产品可查看路线顺序、设备族与单批加工时间。固定工厂采用 SMAT2022 LVHM_E 的设备位与运输轨道。'));
    }else{
      area.append(simpleTable(['工序序号','工序类型','原设备族','厂内设备组','分钟 / lot'],routeSteps(s,d.route).map(r=>[
        r[0],r[1]==='delay'?'等待':'加工',r[2]||'—',r[1]==='delay'?'—':(Object.hasOwn(DB.aliases,r[2])?DB.aliases[r[2]]:r[2]),fmt(r[3],5)])));
      if(d.pieces_per_lot&&Object.hasOwn(DB.routes,d.route))area.append(el('p','details-caption','内置路线表列出标准 25 片/批的工时。本次求解按 '+d.pieces_per_lot+' 片/批重算 per_piece 和 per_batch 工时。'));
    }
  }chooser.onchange=draw;draw();
}
function layoutTable(host,result,inspect){
  const header=el('div','table-tools'),label=el('b','table-title','设备明细'),tools=el('div','tools'),q=el('input');
  q.type='search';q.placeholder='搜索设备、设备组或 Bay';q.setAttribute('aria-label','搜索设备');
  tools.append(q,btn('导出布局 CSV',()=>{
    const cols=['equipment_name','tool_group','bay','x','y','width','height','original_slot','new_slot'];
    download('设备布局.csv',asCsv(result.after,cols),'text/csv;charset=utf-8');
  }));header.append(label,tools);host.append(header);
  const wrap=el('div','tablewrap'),table=el('table'),head=el('thead'),hrow=el('tr'),body=el('tbody');
  for(const title of ['设备编号','设备组','参考 Bay','结果 Bay','X','Y','定位'])hrow.append(el('th','',title));
  head.append(hrow);table.append(head,body);wrap.append(table);host.append(wrap);
  const foot=el('div','table-footer'),status=el('small'),prev=btn('上一页',()=>{page--;draw();}),next=btn('下一页',()=>{page++;draw();});
  foot.append(status,prev,next);host.append(foot);
  const before=new Map((result.before||[]).map(e=>[e.equipment_name,e])),sorted=[...result.after].sort((a,b)=>cmp(a.equipment_name,b.equipment_name));
  let page=0;
  function draw(){
    const query=q.value.trim().toLowerCase(),match=sorted.filter(r=>[r.equipment_name,r.tool_group,r.bay].some(v=>String(v||'').toLowerCase().includes(query)));
    const count=Math.max(1,Math.ceil(match.length/25));page=Math.min(Math.max(page,0),count-1);body.replaceChildren();
    for(const row of match.slice(page*25,(page+1)*25)){
      const tr=el('tr','clickrow');for(const v of [row.equipment_name,row.tool_group||'—',before.get(row.equipment_name)?.bay||'—',row.bay||'—',fmt(row.x,2),fmt(row.y,2)])tr.append(el('td','num',v));
      const td=el('td');td.append(btn('定位',e=>{e.stopPropagation();inspect(row.equipment_name);}));tr.append(td);tr.onclick=()=>inspect(row.equipment_name);body.append(tr);
    }
    if(!match.length){const tr=el('tr'),td=el('td','empty','未找到匹配设备');td.colSpan=7;tr.append(td);body.append(tr);}
    status.textContent='共 '+fmt(match.length)+' 台设备 · 第 '+(page+1)+' / '+count+' 页';prev.disabled=page===0;next.disabled=page===count-1;
  }q.oninput=()=>{page=0;draw();};draw();
}
function replayView(host){
  const rl=DB.rl,frames=rl.frames;let index=0,timer=null;
  const top=el('div','viewtop');top.append(el('b','toolbar-title','原布局与第 95 轮策略布局'),el('span','preset-tag','原始记录抽帧'));host.append(top);
  const imageWrap=el('div','rlimage'),image=el('img','rlframe');image.width=1600;image.height=497;image.alt='第 95 轮原始回放，左侧为参考布局，右侧为强化学习放置设备的过程。';imageWrap.append(image);host.append(imageWrap);
  const controls=el('div','rlcontrols'),play=btn('播放回放',toggle),slider=el('input'),out=el('output'),speed=el('select');
  slider.type='range';slider.min='0';slider.max=String(frames.length-1);slider.value='0';slider.setAttribute('aria-label','RL 布局回放进度');
  speed.setAttribute('aria-label','回放速度');speed.append(option('360','1×'),option('180','2×'),option('720','0.5×'));
  controls.append(play,slider,out,speed);host.append(controls,el('p','details-caption','左右两幅图来自第 95 轮策略的同一次回放。拖动进度条可查看中间步骤，设备放置数量以原图标注为准。'));
  function show(){image.src='data:image/webp;base64,'+frames[index].img;slider.value=String(index);out.textContent='原始帧 '+(frames[index].frame+1)+' / '+rl.frameCount;}
  function stop(){if(timer!==null){clearInterval(timer);timer=null;}play.textContent='播放回放';}
  function start(){stop();if(index>=frames.length-1)index=0;show();play.textContent='暂停';timer=setInterval(()=>{index++;show();if(index===frames.length-1)stop();},Number(speed.value));}
  function toggle(){if(timer===null)start();else stop();}
  slider.oninput=()=>{stop();index=Number(slider.value);show();};speed.onchange=()=>{if(timer!==null)start();};show();return stop;
}
function chartView(host){
  const data=DB.rl.curve;
  const top=el('div','viewtop'),tools=el('div','tools'),epoch=el('select');epoch.setAttribute('aria-label','选择评估轮次');
  for(const [n] of data)epoch.append(option(n,'第 '+n+' 轮'));epoch.value='95';
  tools.append(epoch,btn('导出曲线 CSV',()=>download('RL_训练评估.csv',asCsv(data.map(([epoch,reduction_pct])=>({epoch,reduction_pct})),['epoch','reduction_pct']),'text/csv;charset=utf-8')));
  top.append(el('b','toolbar-title','相对原布局的运输成本降幅'),tools);host.append(top);
  const cv=el('canvas','chartcanvas');cv.setAttribute('aria-label','每五轮评估的运输成本降幅');host.append(cv);
  const note=el('p','details-caption');host.append(note);let graph=null;
  function draw(){
    const W=cv.clientWidth,H=cv.clientHeight;if(!W||!H)return;const d=Math.min(devicePixelRatio||1,2);
    cv.width=Math.round(W*d);cv.height=Math.round(H*d);const c=cv.getContext('2d');c.scale(d,d);
    const L=54,R=W-23,T=24,B=H-38,last=data.at(-1)[0],min=Math.min(0,Math.floor(Math.min(...data.map(p=>p[1]))/5)*5),max=Math.max(20,Math.ceil(Math.max(...data.map(p=>p[1]))/5)*5);
    const x=n=>L+(R-L)*n/last,y=v=>B-(B-T)*(v-min)/(max-min);
    c.font='11px Consolas, monospace';c.textAlign='right';
    for(let v=min;v<=max;v+=5){c.strokeStyle='#dee7df';c.beginPath();c.moveTo(L,y(v));c.lineTo(R,y(v));c.stroke();c.fillStyle='#64776c';c.fillText(v+'%',L-8,y(v)+3);}
    c.textAlign='center';for(const n of [0,50,100,150,200,250,last]){c.fillStyle='#64776c';c.fillText(n,x(n),B+18);}
    c.beginPath();data.forEach(([n,v],i)=>{if(i===0)c.moveTo(x(n),y(v));else c.lineTo(x(n),y(v));});
    c.lineWidth=2.2;c.strokeStyle='#187a62';c.stroke();
    for(const [n,v] of data){c.fillStyle='#36876d';c.beginPath();c.arc(x(n),y(v),2.2,0,Math.PI*2);c.fill();}
    const [n,v]=data.find(p=>p[0]===Number(epoch.value))||data[0];
    c.strokeStyle='#b78439';c.lineWidth=1;c.setLineDash([4,4]);c.beginPath();c.moveTo(x(n),T);c.lineTo(x(n),B);c.stroke();c.setLineDash([]);
    c.beginPath();c.arc(x(n),y(v),5,0,Math.PI*2);c.fillStyle='#dc963e';c.fill();
    note.textContent='第 '+n+' 轮相对原布局下降 '+v.toFixed(2)+'%。曲线含 '+data.length+' 个无探索评估点，每 5 轮记录一次。';
    graph={L,R,last};
  }
  epoch.onchange=draw;cv.onclick=e=>{if(!graph)return;const n=Math.max(0,Math.min(graph.last,Math.round(((e.clientX-cv.getBoundingClientRect().left-graph.L)/(graph.R-graph.L))*graph.last/5)*5));epoch.value=String(n);draw();};
  const ob=new ResizeObserver(draw);ob.observe(cv);draw();return ()=>ob.disconnect();
}
function localFlow(host){
  const intro=el('div','viewtop');intro.append(el('b','toolbar-title','A2 / A3 加工分配与边界运输'),el('span','preset-tag','原始交互成果'));host.append(intro);
  host.append(el('p','details-caption','在下方原始展示中选择工序或机位，可查看关联设备、加工量及进出区域的运输关系。'));
  const iframe=el('iframe','local-iframe');iframe.title='A2 / A3 布局优化解释器';iframe.setAttribute('sandbox','allow-scripts');
  iframe.srcdoc=new TextDecoder().decode(b64Bytes(DB.local.html));host.append(iframe);
}
