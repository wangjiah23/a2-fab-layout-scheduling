'use strict';
let liveJob=null, sending=false;
const finished=status=>['completed','failed'].includes(status);
function jobDownload(label,url){
  const a=el('a','toolbtn',label);a.href=url;a.download='';return a;
}
function jobDownloads(){
  const host=$('jobDownloads');host.replaceChildren();host.hidden=!result?.job;
  if(!host.hidden){
    const base='/api/jobs/'+result.job+'/';
    host.append(jobDownload('下载布局与回放 JSON',base+'result.json'),jobDownload('下载仿真输入 ZIP',base+'simulation-input.zip'));
  }
}
function onlineView(host){
  const s=chosen().scene,card=el('div','solve-pane');
  card.append(el('h2','','起始布局'),el('p','muted',s.initial_layout?'已载入 1,068 台设备的起始机位。':'使用 SMAT2022 全厂原始机位，可导入本厂的其他摆放方案。'));
  const input=el('input');input.type='file';input.accept='.csv,.json';input.hidden=true;
  const change=btn('导入起始布局',()=>input.click(),'outline');
  function setLayout(rows,label){
    const next={...s};if(rows)next.initial_layout=rows;else delete next.initial_layout;
    chooseCase(caseForScene(next,label),'rl');showTab('online');
  }
  input.onchange=async()=>{
    const f=input.files[0],seq=++importSerial;if(!f)return;
    try{const rows=await readInitial(f);if(seq===importSerial)setLayout(rows,f.name);}
    catch(err){message('导入起始布局失败。'+err.message,true);}
  };
  const layoutTools=el('div','tools');layoutTools.append(change,input);
  layoutTools.append(btn('下载原厂布局 CSV',()=>download('layout.csv',asCsv(DB.plant.slots,['equipment_name','tool_group','bay','x','y','width','height']),'text/csv;charset=utf-8')));
  if(s.initial_layout)layoutTools.append(btn('恢复原厂布局',()=>setLayout(null,chosen().label)));
  card.append(layoutTools,el('h2','','生产场景'),el('p','',chosen().label+' · '+s.demands.length+' 种产品 · '+fmt(sceneFacts(s).lots,3)+' lot'));
  card.append(el('p','muted','按相同订单评估起始布局和 PPO 候选布局，推荐运输成本较低的方案。'));
  const bar=el('div','tools'),start=btn('开始求解',submitJob,'primary');start.disabled=sending||!!(liveJob&&!finished(liveJob.status));
  bar.append(start);
  if(liveJob){
    const label=liveJob.status==='running'?(liveJob.phase||'正在准备模型'):liveJob.status==='queued'?'已进入求解队列':liveJob.status==='completed'?'上次求解已完成':(liveJob.detail||'求解失败');
    const status=el('span','run-status',label);status.setAttribute('role','status');bar.append(status);
    if(liveJob.status==='completed')bar.append(btn('查看上次求解',()=>openJobResult(liveJob,false)));
    if(liveJob.offline)bar.append(btn('重新连接任务',()=>watchJob(liveJob)));
  }
  card.append(bar);host.append(card);
}
function refreshRun(){
  if(tab==='online'){const host=$('view');host.replaceChildren();onlineView(host);}
}
async function apiJSON(response){
  if(!response.ok){
    let d;try{d=await response.json();}catch{}
    throw Error(typeof d?.detail==='string'?d.detail:'服务暂时不可用（HTTP '+response.status+'）。');
  }
  return response.json();
}
async function submitJob(){
  if(sending||liveJob&&!finished(liveJob.status))return;
  const s=chosen().scene; sending=true;refreshRun();message();
  try{
    const response=await fetch('/api/jobs',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({schema:'fab-job-v1',method:'rl',scenario:s})});
    const item=await apiJSON(response);
    history.replaceState(null,'','?job='+item.id);
    watchJob({id:item.id,key:sceneKey(s),status:'queued'});
  }catch(err){message(err.message,true);}finally{sending=false;refreshRun();}
}
async function openJobResult(job,auto){
  try{
    const response=await fetch('/api/jobs/'+job.id+'/result.json');
    if(!response.ok)throw Error('本次结果暂时无法读取。');
    const file=new File([await response.blob()],'PPO 求解结果.json',{type:'application/json'});
    if(auto&&(tab!=='online'||job.key&&job.key!==sceneKey(chosen().scene)))return;
    const res=await importResult(file,auto);
    if(res){res.job=job.id;jobDownloads();}
  }catch(err){message(err.message,true);}
}
async function restoreJob(id){
  const seq=importSerial;
  try{
    const request=await apiJSON(await fetch('/api/jobs/'+id+'/request.json',{cache:'no-store'}));
    if(seq!==importSerial||liveJob)return;
    const scene=normalizeScene(request.scenario);
    chooseCase(caseForScene(scene,'已提交场景'),'rl');
    watchJob({id,key:sceneKey(scene),status:'queued'});
  }catch(err){message('恢复任务失败。'+err.message,true);}
}
async function watchJob(job){
  liveJob=job;job.offline=false;refreshRun();
  try{
    while(liveJob===job){
      const next=await apiJSON(await fetch('/api/jobs/'+job.id,{cache:'no-store'}));
      const changed=job.status!==next.status||job.phase!==next.phase;
      Object.assign(job,next);if(changed)refreshRun();
      if(job.status==='completed'){await openJobResult(job,true);return;}
      if(job.status==='failed'){message(job.detail,true);return;}
      await new Promise(resolve=>setTimeout(resolve,1800));
    }
  }catch(err){job.offline=true;job.detail=err.message;message('暂时无法读取任务。可重新连接或重新打开本页。'+err.message,true);refreshRun();}
}
