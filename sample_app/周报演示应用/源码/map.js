'use strict';
const areaColors={Dry_Etch:'#358c87',Litho:'#b67555',Wet_Etch:'#5e9eb2',TF:'#7589b8',Diffusion:'#75a549',Dielectric:'#d39c53',Litho_Met:'#b481a5',Implant:'#8070a7',Planar:'#b2a647',Def_Met:'#688591',TF_Met:'#9c7883','其他':'#6d9391'};
const areaLabels={Dry_Etch:'干法刻蚀',Litho:'光刻',Wet_Etch:'湿法刻蚀',TF:'薄膜',Diffusion:'扩散',Dielectric:'介质',Litho_Met:'光刻量测',Implant:'注入',Planar:'平坦化',Def_Met:'缺陷量测',TF_Met:'薄膜量测'};
const colorFor=area=>Object.hasOwn(areaColors,area)?areaColors[area]:areaColors['其他'];
function makeMap(host,r,initialName='',replay=null){
  let mode=replay?'both':'after',bay='',zoom=1,panX=0,panY=0,selected=initialName,tracks=r.plant,pan=null,plot=[],raf=0,disposed=false;
  const before=r.before||[],after=(replay&&r.policy)||r.after;
  let step=0,timer=null;
  const maps=[new Map(before.map(e=>[e.equipment_name,e])),new Map(after.map(e=>[e.equipment_name,e]))];
  let example=null,maxMove=0;
  for(const a of after){
    const b=maps[0].get(a.equipment_name);
    if(b){const d=Math.hypot(a.x-b.x,a.y-b.y);if(d>maxMove){example=a;maxMove=d;}}
  }
  const areas=[...new Set(after.map(a=>a.area||'其他'))].sort((a,b)=>{const keys=Object.keys(areaColors);return keys.indexOf(a)-keys.indexOf(b);});
  const top=el('div','viewtop'),segments=el('div','segments'),tools=el('div','right-tools'),select=el('select');
  select.title='聚焦 Bay';select.setAttribute('aria-label','聚焦 Bay');select.append(option('','全厂 / 全部 Bay'));
  const bays=[...new Set([...before,...after].map(a=>a.bay).filter(Boolean))].sort(cmp);
  for(const b of bays)select.append(option(b,b));
  const switches=new Map();
  for(const [id,text] of (r.before?[['after','优化布局'],['before','参考布局'],['both','并排对照']]:[['after','导入布局']])){
    const b=btn(text,()=>{mode=id;for(const [key,item] of switches)item.classList.toggle('on',key===id);reset();});b.classList.toggle('on',id===mode);segments.append(b);switches.set(id,b);
  }
  select.onchange=()=>{bay=select.value;reset();};
  tools.append(select);
  if(r.plant){
    const rail=btn('轨道',()=>{tracks=!tracks;rail.setAttribute('aria-pressed',String(tracks));rail.style.background=tracks?'#e8f3e8':'';queue();});
    rail.title='显示或隐藏固定运输网络';rail.setAttribute('aria-pressed','true');rail.style.background='#e8f3e8';tools.append(rail);
  }
  tools.append(btn('−',()=>scaleAt(1/1.5)),btn('＋',()=>scaleAt(1.5)),btn('复位',()=>{reset();}));
  top.append(segments,tools);
  const wrap=el('div','mapwrap'),canvas=el('canvas','mapcanvas');
  canvas.setAttribute('aria-label','设备布局图，可缩放、拖动及点选设备');canvas.tabIndex=0;
  wrap.append(canvas,el('div','maphelp','滚轮缩放 · 拖动平移 · 单击设备查看明细'));
  const bottom=el('div','below'),legend=el('div','legend');
  for(const area of areas){
    const a=el('span'),sw=el('i');sw.style.background=colorFor(area);a.append(sw,document.createTextNode(areaLabels[area]||area));legend.append(a);
  }
  const detail=el('div','info');bottom.append(legend);host.append(top,wrap,bottom,detail);
  if(replay){
    const controls=el('div','rlcontrols placement-controls'),slider=el('input'),out=el('output'),action=el('p','placement-action'),speed=el('select');
    const play=btn('播放',()=>timer===null?start():pause()),prev=btn('上一步',()=>seek(step-1)),next=btn('下一步',()=>seek(step+1));
    slider.type='range';slider.min=0;slider.max=after.length;slider.step=1;slider.setAttribute('aria-label','推理放置步数');
    speed.setAttribute('aria-label','每秒放置步数');for(const rate of [5,15,30,60])speed.append(option(rate,rate+' 步/秒'));speed.value='15';
    function pause(){clearInterval(timer);timer=null;play.textContent='播放';}
    function update(n){
      step=Math.max(0,Math.min(after.length,n));const row=replay.rows[step-1];
      selected=row?.equipment_name||'';slider.value=String(step);out.textContent=step+' / '+after.length+' 台';
      action.textContent=row?('第 '+step+' 步　'+(replay.automatic.has(step-1)?'同组自动放置':'策略放置')+'　'+row.equipment_name+' → '+row.bay+' · 槽位 '+row.new_slot):'第 0 步　设备尚未放置，固定轨道和 AMHS 已就位。';
      prev.disabled=step===0;next.disabled=step===after.length;queue();
    }
    function seek(n){pause();update(n);}
    function start(){pause();if(step===after.length)update(0);play.textContent='暂停';timer=setInterval(()=>{update(step+1);if(step===after.length)pause();},1000/Number(speed.value));}
    speed.onchange=()=>{if(timer!==null)start();};slider.oninput=()=>seek(Number(slider.value));
    controls.append(play,prev,next,slider,out,speed,btn('终局',()=>seek(after.length)));
    host.insertBefore(controls,bottom);host.insertBefore(action,bottom);
    update(0);
    if(replay.autoplay&&!matchMedia('(prefers-reduced-motion: reduce)').matches)start();
  }
  const ctx=canvas.getContext('2d');
  function details(){
    detail.replaceChildren();
    if(!selected){
      detail.append(el('b','','设备点选'),el('span','',r.plant?'颜色表示设备工艺类别，灰绿色线条表示固定运输轨道。':'设备按导入文件的原始坐标绘制。'));
      if(example)detail.append(btn('查看移位示例',()=>{selected=example.equipment_name;bay='';select.value='';reset();}));
      return;
    }
    const a=maps[1].get(selected),b=maps[0].get(selected),current=(mode==='before'?b:a)||a||b;
    if(!current)return;
    detail.append(el('b','',selected),el('span','',(current.tool_group||'设备族未提供')+' · '+(current.bay||'Bay 未提供')),
      el('span','','坐标 ('+fmt(current.x,1)+', '+fmt(current.y,1)+')'));
    if(a&&b){
      detail.append(el('span','','所属 Bay '+(b.bay||'—')+' → '+(a.bay||'—')));
      if(a.original_slot!==undefined&&a.original_slot!==''&&a.new_slot!==undefined&&a.new_slot!=='')detail.append(el('span','','机位 '+a.original_slot+' → '+a.new_slot));
      if(mode!=='both'&&(!bay||a.bay===bay&&b.bay===bay)&&(a.x!==b.x||a.y!==b.y))detail.append(el('span','','橙色虚线表示设备移位'));
    }
  }
  function reset(){zoom=1;panX=0;panY=0;queue();}
  function scaleAt(factor,cx=0,cy=0){
    const next=Math.min(22,Math.max(.65,zoom*factor)),ratio=next/zoom;
    panX=cx-(cx-panX)*ratio;panY=cy-(cy-panY)*ratio;zoom=next;queue();
  }
  function queue(){if(!raf&&!disposed)raf=requestAnimationFrame(draw);}
  function bounds(rows){
    let x0=Infinity,y0=Infinity,x1=-Infinity,y1=-Infinity;
    for(const p of rows){
      const w=(p.width||0)/2,h=(p.height||0)/2;
      x0=Math.min(x0,p.x-w);x1=Math.max(x1,p.x+w);y0=Math.min(y0,p.y-h);y1=Math.max(y1,p.y+h);
    }
    return Number.isFinite(x0)?[x0,y0,x1,y1]:[0,0,1,1];
  }
  function draw(){
    raf=0;const W=canvas.clientWidth,H=canvas.clientHeight;if(!W||!H)return;
    const dpr=Math.min(window.devicePixelRatio||1,2);
    if(canvas.width!==Math.round(W*dpr)||canvas.height!==Math.round(H*dpr)){canvas.width=Math.round(W*dpr);canvas.height=Math.round(H*dpr);}
    ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,W,H);
    const placed=replay?replay.rows.slice(0,step):after;
    const panels=mode==='both'?[before,placed]:[mode==='before'?before:placed];
    const rowsFor=b=>bay?b.filter(e=>e.bay===bay):b;
    const bound=bounds(rowsFor(before.length?[...before,...after]:after));
    if(r.plant&&!r.local&&!bay){
      const rail=globalRailBounds;
      bound[0]=Math.min(bound[0],rail[0]);bound[1]=Math.min(bound[1],rail[1]);bound[2]=Math.max(bound[2],rail[2]);bound[3]=Math.max(bound[3],rail[3]);
    }
    const [x0,y0,x1,y1]=bound;const pw=W/panels.length;
    const padX=pw<450?15:26,padY=16;
    const scale=Math.min((pw-padX*2)/Math.max(x1-x0,1),(H-2*padY)/Math.max(y1-y0,1))*zoom;
    plot=[];
    panels.forEach((all,j)=>{
      const rows=rowsFor(all),left=j*pw,midX=left+pw/2,midY=H/2+1;
      const px=x=>midX+panX+(x-(x0+x1)/2)*scale,py=y=>midY+panY+(y-(y0+y1)/2)*scale;
      ctx.save();ctx.beginPath();ctx.rect(left+1,0,pw-2,H);ctx.clip();
      if(r.plant&&tracks){
        ctx.beginPath();ctx.strokeStyle='#aac1b4';ctx.lineWidth=zoom>2?.9:.63;
        for(const [name,ax,ay,bx,by] of DB.plant.rails){
          if((bay||r.local)&&name!==bay&&!(r.local&&!bay&&['A2','A3'].includes(name)))continue;
          const x=px(ax),y=py(ay),x2=px(bx),y2=py(by);
          if(Math.max(x,x2)<left||Math.min(x,x2)>left+pw||Math.max(y,y2)<0||Math.min(y,y2)>H)continue;
          ctx.moveTo(x,y);ctx.lineTo(x2,y2);
        }ctx.stroke();
        if(!r.local){
          ctx.fillStyle='#6a786d';
          for(const e of DB.plant.fixed)if(!bay||e.bay===bay){const x=px(e.x),y=py(e.y);ctx.fillRect(x-1.4,y-1.4,2.8,2.8);}
        }
      }
      const progressing=!!replay&&all===placed;
      if(progressing){
        ctx.strokeStyle='#cdd9d3';ctx.lineWidth=.55;
        for(const e of rowsFor(before))ctx.strokeRect(px(e.x)-scale*e.width/2,py(e.y)-scale*e.height/2,scale*e.width,scale*e.height);
      }
      const hits=[],bayPositions=new Map();
      let highlighted=null;
      for(const e of rows){
        const x=px(e.x),y=py(e.y),w=e.width?Math.max(2,scale*e.width):5,h=e.height?Math.max(2,scale*e.height):5;
        if(x+w/2<left||x-w/2>left+pw||y+h/2<0||y-h/2>H)continue;
        ctx.fillStyle=colorFor(e.area);
        if(e.width&&e.height){
          ctx.fillRect(x-w/2,y-h/2,w,h);
          if(w>5&&h>5){ctx.strokeStyle='#36524765';ctx.lineWidth=.55;ctx.strokeRect(x-w/2,y-h/2,w,h);}
        }else{ctx.beginPath();ctx.arc(x,y,3.2,0,Math.PI*2);ctx.fill();}
        const hit={x,y,w,h,e};hits.push(hit);if(e.equipment_name===selected)highlighted=hit;
        if(e.bay){const pos=bayPositions.get(e.bay)||[0,0,-Infinity];pos[0]+=x;pos[1]++;pos[2]=Math.max(pos[2],y+h/2);bayPositions.set(e.bay,pos);}
      }
      if(!bay&&r.plant&&pw>450&&zoom<1.8){
        ctx.font='9px Consolas, monospace';ctx.textAlign='center';ctx.fillStyle='#536b61';
        for(const [k,p] of bayPositions)if(p[1]>1)ctx.fillText(k,p[0]/p[1],Math.min(H-3,p[2]+10));
        ctx.textAlign='left';
      }
      if(highlighted){
        const a=maps[1].get(selected),b=maps[0].get(selected);
        if(a&&b&&mode!=='both'&&(!bay||a.bay===bay&&b.bay===bay)){
          const x=px(b.x),y=py(b.y),tx=px(a.x),ty=py(a.y),distance=Math.hypot(tx-x,ty-y);
          if(distance>11){
            const dx=(tx-x)/distance,dy=(ty-y)/distance;
            ctx.strokeStyle='#b65f1b';ctx.fillStyle='#b65f1b';ctx.lineWidth=2;ctx.setLineDash([6,4]);ctx.beginPath();ctx.moveTo(x,y);ctx.lineTo(tx,ty);ctx.stroke();ctx.setLineDash([]);
            ctx.beginPath();ctx.arc(x,y,4.4,0,Math.PI*2);ctx.stroke();ctx.beginPath();ctx.moveTo(tx,ty);ctx.lineTo(tx-dx*10-dy*4,ty-dy*10+dx*4);ctx.lineTo(tx-dx*10+dy*4,ty-dy*10-dx*4);ctx.closePath();ctx.fill();
          }
        }
        const {x,y,w,h}=highlighted;ctx.strokeStyle='#101f1e';ctx.lineWidth=2;ctx.strokeRect(x-w/2-2,y-h/2-2,w+4,h+4);
        ctx.strokeStyle='#f0a039';ctx.lineWidth=2;ctx.strokeRect(x-w/2-4,y-h/2-4,w+8,h+8);
      }
      const label=(progressing?'推理布局':mode==='both'?(j?'优化布局':'参考布局'):(r.before?(mode==='before'?'参考布局':'优化布局'):'导入布局'))+'   '+fmt(rows.length)+' 台';
      ctx.font='bold 11px "Microsoft YaHei", sans-serif';let tw=ctx.measureText(label).width;
      ctx.fillStyle='#f5f9f2ee';ctx.fillRect(left+7,5,tw+12,20);ctx.fillStyle='#345146';ctx.fillText(label,left+12,19);
      if(j>0){ctx.fillStyle='#c2d1c7';ctx.fillRect(left+1,0,1,H);}
      ctx.restore();plot.push({hits,left,pw,midX,midY});
    });details();
  }
  const point=e=>{const b=canvas.getBoundingClientRect();return [e.clientX-b.left,e.clientY-b.top];};
  canvas.addEventListener('wheel',e=>{e.preventDefault();const [x,y]=point(e),p=plot.find(p=>x>=p.left&&x<p.left+p.pw);if(p)scaleAt(e.deltaY<0?1.22:1/1.22,x-p.midX,y-p.midY);},{passive:false});
  canvas.onpointerdown=e=>{if(e.button!==0)return;const [x,y]=point(e);pan={x,y,px:panX,py:panY,drag:false};canvas.setPointerCapture(e.pointerId);};
  canvas.onpointermove=e=>{if(!pan)return;const [x,y]=point(e);if(Math.abs(x-pan.x)+Math.abs(y-pan.y)>4)pan.drag=true;if(pan.drag){panX=pan.px+x-pan.x;panY=pan.py+y-pan.y;queue();}};
  canvas.onpointerup=e=>{
    if(!pan)return;const [x,y]=point(e),p=plot.find(p=>x>=p.left&&x<p.left+p.pw);
    if(!pan.drag&&p){
      let hit=null,dist=Infinity;
      for(const h of p.hits)if(Math.abs(x-h.x)<=h.w/2+3&&Math.abs(y-h.y)<=h.h/2+3){const d=Math.hypot(x-h.x,y-h.y);if(d<dist){hit=h;dist=d;}}
      selected=hit?.e.equipment_name||'';
    }pan=null;queue();
  };
  canvas.onpointercancel=()=>{pan=null;};
  canvas.onkeydown=e=>{if(e.key==='+'||e.key==='=')scaleAt(1.5);else if(e.key==='-')scaleAt(1/1.5);else if(e.key==='0')reset();else return;e.preventDefault();};
  if(selected){const row=maps[1].get(selected)||maps[0].get(selected);if(row&&row.bay){bay=row.bay;select.value=bay;}}
  const obs=new ResizeObserver(queue);obs.observe(canvas);queue();
  return ()=>{disposed=true;clearInterval(timer);obs.disconnect();if(raf)cancelAnimationFrame(raf);};
}
const globalRailBounds=DB.plant.rails.reduce((b,s)=>[Math.min(b[0],s[1],s[3]),Math.min(b[1],s[2],s[4]),Math.max(b[2],s[1],s[3]),Math.max(b[3],s[2],s[4])],[Infinity,Infinity,-Infinity,-Infinity]);
