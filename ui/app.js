const $ = id => document.getElementById(id);
const models = {text:'mlfoundations/open_clip/ViT-B-16-SigLIP2-256/webli',trace:'mlfoundations/open_clip/ViT-B-16-SigLIP2-256/webli',object:'transformers/owlv2/google/owlv2-base-patch16-ensemble',face:'deepinsight/insightface/buffalo_l/_unknown'};
models.mixed=models.text;
const samples = {trace:'jogging-OV-CGjSLNbg_000053_000063-preview.jpg',object:'backpack-independent.jpg',face:'coughing-6XvsLPDioVA_000000_000010-preview.jpg'};
samples.mixed=samples.object;
const descriptions = {
 action:['动作片段检索','Qwen3-VL 2B · 多帧视频语义','任意中文描述实时编码，检索已入库视频的动作片段；首次查询可能需要加载模型。'],
 mixed:['图文组合检索','SigLIP 2 · 图片与文字融合','参考图描述外观，文字补充场景；使用整帧语义索引排序。'],
 text:['中文语义检索','SigLIP 2 · 多语言视觉语义','直接编码中文查询，不依赖文件名关键词。'],
 trace:['截图溯源','语义召回 + 画面比对','先召回候选视频，再按约 0.1 秒间隔比对画面细化位置。'],
 object:['相似物体检索','OWLv2 · 目标区域特征','使用独立背包照片，检索视频中的相似物体区域。'],
 face:['按姓名查找人物','InsightFace · 姓名与人脸向量','姓名对应已登记的人脸特征，搜索结果按相似度排序。']
};
let mode='text', queryFile=null, previewURL=null, busy=false, lastComparison=null;
const setError = text => { $('error').hidden=!text; $('error').textContent=text; };
async function checked(response){if(!response.ok){let data=await response.json().catch(()=>({}));throw new Error(typeof data.detail==='string'?data.detail:JSON.stringify(data.detail||data));}return response;}
function lock(value){busy=value;$('rerank-enabled').disabled=value;$('rerank-candidates').disabled=value;$('order-original').disabled=value;$('order-reranked').disabled=value;$('search-button').disabled=value;$('search-button').textContent=value?'正在检索…':'开始检索 ↗';document.querySelectorAll('nav button[data-mode]').forEach(b=>b.disabled=value);}
function selectMode(next){
 if(busy)return;window.showSearch?.();mode=next;queryFile=null;
 $('image-query').value='';$('query-preview').hidden=true;$('query-name').textContent='尚未选择图片';
 document.querySelectorAll('nav button[data-mode]').forEach(b=>b.classList.toggle('active',b.dataset.mode===next));
 $('text-panel').hidden=!['text','mixed','action'].includes(next);
 $('image-panel').hidden=!['trace','object','mixed'].includes(next);
 $('face-panel').hidden=next!=='face';$('scene-panel').hidden=next!=='face';$('negative-panel').hidden=['trace','action'].includes(next);$('action-panel').hidden=next!=='action';document.querySelector('#text-panel .examples').hidden=next==='action';
 $('negative-query').value='';$('scene-query').value='';
 if(next==='mixed')$('text-query').value='在街道上';
 if(next==='action'){$('text-query').value='一个人失去平衡摔倒在地';loadActionStatus();}
 const d=descriptions[next];$('mode-title').textContent=d[0];$('model-label').textContent=d[1];$('mode-note').textContent=d[2];
 $('image-hint').textContent=next==='trace'?'上传视频截图':'上传参考图片（建议主体清晰）';
 $('sample-image').textContent=next==='trace'?'使用 2.25 秒演示截图':'使用独立背包照片';
 setError('');lastComparison=null;$('rerank-comparison').hidden=true;$('action-metrics').hidden=true;$('results').replaceChildren();$('result-status').textContent='等待查询';
}
document.querySelectorAll('nav button[data-mode]').forEach(b=>b.onclick=()=>selectMode(b.dataset.mode));
document.querySelectorAll('[data-text]').forEach(b=>b.onclick=()=>{$('text-query').value=b.dataset.text;});
function chooseFile(file){queryFile=file;if(previewURL)URL.revokeObjectURL(previewURL);previewURL=URL.createObjectURL(file);$('query-preview').src=previewURL;$('query-preview').hidden=false;$('query-name').textContent=file.name;}
$('image-query').onchange=e=>{if(e.target.files[0])chooseFile(e.target.files[0]);};
async function sampleFile(kind){const response=await checked(await fetch('/queries/'+samples[kind]));return new File([await response.blob()],samples[kind],{type:'image/jpeg'});}
$('sample-image').onclick=async()=>{try{chooseFile(await sampleFile(mode));}catch(e){setError(e.message);}};
async function loadPeople(){const names=await (await checked(await fetch('/mvp-api/people'))).json();$('people-list').replaceChildren(...names.map(n=>{const o=document.createElement('option');o.value=n;return o;}));return names;}
async function enroll(name,file){if(!name.trim()||!file)throw new Error('请填写姓名并选择一张单人照片');const form=new FormData();form.append('name',name.trim());form.append('photo',file);await checked(await fetch('/mvp-api/people',{method:'POST',body:form}));await loadPeople();$('person-name').value=name.trim();}
$('sample-person').onclick=async()=>{const b=$('sample-person');b.disabled=true;setError('');try{if(!(await loadPeople()).includes('演示人物A'))await enroll('演示人物A',await sampleFile('face'));$('person-name').value='演示人物A';$('result-status').textContent='演示人物A已就绪，点击开始检索';}catch(e){setError(e.message);}finally{b.disabled=false;}};
$('enroll').onclick=async()=>{const b=$('enroll');b.disabled=true;setError('');try{await enroll($('enroll-name').value,$('enroll-photo').files[0]);$('result-status').textContent='登记完成，可按姓名检索';}catch(e){setError(e.message);}finally{b.disabled=false;}};
$('delete-person').onclick=async()=>{setError('');try{const name=$('enroll-name').value.trim();if(!name)throw new Error('请在登记姓名栏输入要删除的姓名');await checked(await fetch('/mvp-api/people/'+encodeURIComponent(name),{method:'DELETE'}));await loadPeople();$('result-status').textContent='已删除登记：'+name;}catch(e){setError(e.message);}};
function clock(seconds){return Math.floor(seconds/60).toString().padStart(2,'0')+':'+(seconds%60).toFixed(1).padStart(4,'0');}
function mediaURL(link){return new URL(link,location.origin+'/demo/').href;}
function play(segment,video,action=false){const player=$('player');const start=action?segment.ts:(segment.thumbnail_ts??segment.ts);$('player-title').textContent=video.filename;$('player-caption').textContent=`${action?'命中片段 '+clock(segment.ts)+'–'+clock(segment.te):'命中时间 '+clock(start)} · ${segment.rerank_score!==undefined?'重排分':segment.face_similarity!==undefined?'融合排序分':'相似度'} ${Number(segment.rerank_score??segment.distance).toFixed(3)}${action?' · 从片段起点播放，可继续查看前后文':''}`;player.src=mediaURL(segment.link).split('#')[0];player.onloadedmetadata=()=>{player.currentTime=Math.min(start,player.duration);player.play().catch(()=>{});};$('player-dialog').showModal();}
$('close-player').onclick=()=>{$('player').pause();$('player-dialog').close();};$('player-dialog').addEventListener('close',()=>$('player').pause());
function showResults(data,elapsed,order='reranked'){
 const applied=data.rerank?.applied===true;
 $('rerank-comparison').hidden=!applied;
 $('action-metrics').hidden=data.score_kind!=='action';
 if(data.score_kind==='action'){
  const m=data.resources, rt=data.rerank?.timing, gib=n=>(n/1024**3).toFixed(2);
  const lines=[`本次查询：${data.query}`];
  if(m){lines.push(`服务器总耗时 ${m.total_seconds.toFixed(2)} 秒 · 召回 ${data.retrieval_seconds.toFixed(2)} 秒${data.query_embedding_cached?'（文本向量缓存）':''}${rt?` · 重排 ${rt.total_seconds.toFixed(2)} 秒（含模型加载 ${rt.load_seconds.toFixed(2)} 秒）`:''}`);lines.push(`服务进程 RSS 峰值 ${gib(m.sampled_peak.rss_bytes)} GiB · Metal 驱动峰值 ${gib(m.sampled_peak.mps_driver_bytes)} GiB · 系统可用内存最低 ${gib(m.min_system_available_bytes)} GiB`);lines.push('内存每 50ms 采样，包含服务的其他模型；RSS 与 Metal 有重叠，不可相加。');}
  if(data.rerank?.error)lines.push(data.rerank.error);
  $('action-metrics').replaceChildren(...lines.map(text=>{const p=document.createElement('p');p.textContent=text;return p;}));
 }
 if(applied){lastComparison={data,elapsed};$('comparison-label').textContent=`同一批 ${data.rerank.candidate_count} 个候选 · ${order==='original'?'原始排序':'重排排序'}`;$('order-original').setAttribute('aria-pressed',String(order==='original'));$('order-reranked').setAttribute('aria-pressed',String(order==='reranked'));}
 const r=data.video_results;$('results').replaceChildren();let list=r?.unmerged_windows||[];if(applied&&order==='original')list=[...list].sort((a,b)=>a.original_rank-b.original_rank);if(mode==='face'&&data.score_kind!=='fusion')list=list.filter(s=>s.distance>=Number($('face-threshold').value));
 // One strongest match per source video; preserve actual model ranking. No filenames influence retrieval.
 const seen=new Set();list=list.filter(s=>{if(data.score_kind==='action')return true;if(seen.has(s.media_id))return false;seen.add(s.media_id);return true;}).slice(0,data.score_kind==='action'?20:12);
 $('result-status').textContent=`${list.length} 个${data.score_kind==='action'?'片段':'视频'} · ${elapsed.toFixed(2)} 秒 · ${data.score_kind==='action'?`已索引 ${data.indexed_videos} 个视频 / ${data.indexed_segments} 个片段`:data.score_kind==='fusion'?'人物门槛已应用 · 场景参与排序':'每段显示最佳命中'}`;
 if(!list.length){const p=document.createElement('p');p.className='empty';p.textContent='没有符合当前条件的结果。请更换查询图片、描述或调整门槛。';$('results').append(p);return;}
 list.forEach((s,i)=>{const v=r.videos[s.media_id];const card=document.createElement('button');card.type='button';card.className='result';card.title=`播放 ${v.filename}，${clock(data.score_kind==='action'?s.ts:(s.thumbnail_ts??s.ts))}`;
 const thumb=document.createElement('div');thumb.className='thumb';const img=document.createElement('img');img.alt='命中画面';img.src=s.thumbnail.startsWith('data:')?s.thumbnail:mediaURL(s.thumbnail);thumb.append(img);
 if(s.bbox){img.onload=()=>{const box=document.createElement('span');box.className='bbox';const w=thumb.clientWidth,h=thumb.clientHeight,scale=Math.min(w/img.naturalWidth,h/img.naturalHeight),iw=img.naturalWidth*scale,ih=img.naturalHeight*scale;box.style.cssText=`left:${(w-iw)/2+s.bbox.x*iw}px;top:${(h-ih)/2+s.bbox.y*ih}px;width:${s.bbox.w*iw}px;height:${s.bbox.h*ih}px`;thumb.append(box);};}
 const rank=document.createElement('span');rank.className='rank';rank.textContent=String(i+1).padStart(2,'0');const time=document.createElement('span');time.className='time';time.textContent=data.score_kind==='action'?clock(s.ts)+'–'+clock(s.te):clock(s.thumbnail_ts??s.ts);thumb.append(rank,time);
 const info=document.createElement('div');info.className='result-info';const title=document.createElement('h3');title.textContent=v.filename;const line=document.createElement('p');const score=document.createElement('span');score.textContent=(s.rerank_score!==undefined?'重排分 ':data.score_kind==='fusion'?'融合排序分 ':'相似度 ')+Number(s.rerank_score??s.distance).toFixed(3);const cta=document.createElement('span');cta.textContent='播放片段 ↗';line.append(score,cta);info.append(title,line);if(s.face_similarity!==undefined){const detail=document.createElement('p');detail.className='score-evidence';detail.textContent=`人脸 ${s.face_similarity.toFixed(3)} · 场景 ${s.scene_similarity.toFixed(3)}`;info.append(detail);}if(s.original_rank!==undefined){const detail=document.createElement('p');detail.className='score-evidence';detail.textContent=`原第 ${s.original_rank} 名 · 向量相似度 ${s.embedding_score.toFixed(3)}`;info.append(detail);}card.append(thumb,info);card.onclick=()=>play(s,v,data.score_kind==='action');$('results').append(card);});
}
$('order-original').onclick=()=>{if(lastComparison)showResults(lastComparison.data,lastComparison.elapsed,'original');};
$('order-reranked').onclick=()=>{if(lastComparison)showResults(lastComparison.data,lastComparison.elapsed,'reranked');};
$('search-form').onsubmit=async e=>{
 e.preventDefault();if(busy)return;setError('');lock(true);const start=performance.now();const ticker=setInterval(()=>{$('result-status').textContent=`正在${mode==='action'&&$('rerank-enabled').checked?'检索并逐段重排':'检索'} · 已用 ${((performance.now()-start)/1000).toFixed(0)} 秒`;},1000);
 try{let response;const negative=$('negative-query').value.trim();
 if(mode==='action'){
  const q=$('text-query').value.trim();if(!q)throw new Error('请输入动作或场景描述');
  const form=new FormData();form.append('query',q);form.append('limit','20');form.append('rerank',String($('rerank-enabled').checked));form.append('rerank_candidates',$('rerank-candidates').value);
  response=await fetch('/mvp-api/actions/search',{method:'POST',body:form});
 }else if(mode==='face'){
  const name=$('person-name').value.trim();if(!name)throw new Error('请先选择或登记人物姓名');
  const form=new FormData();form.append('scene',$('scene-query').value.trim());form.append('negative',negative);form.append('threshold',$('face-threshold').value);
  response=await fetch('/mvp-api/people/'+encodeURIComponent(name)+'/search',{method:'POST',body:form});
 }else if(mode==='trace'){
  if(!queryFile)throw new Error('请上传截图或使用演示图片');const form=new FormData();form.append('photo',queryFile);
  response=await fetch('/mvp-api/trace',{method:'POST',body:form});
 }else{
  const params=new URLSearchParams({search_in:'video',feature_extractor_id:models[mode],start:'0',end:'100',thumbnails_to_send:'1',add_prefix:'false'});
  const form=new FormData();
  if(['text','mixed'].includes(mode)){const q=$('text-query').value.trim();if(!q)throw new Error('请输入场景描述');params.append('text_queries',q);}
  if(['object','mixed'].includes(mode)){if(!queryFile)throw new Error('请上传图片或使用演示图片');form.append('image_file_queries',queryFile);}
  if(negative)params.append('negative_text_queries',negative);
  response=await fetch('/demo/search?'+params,{method:'POST',body:form});
 }
 const data=await(await checked(response)).json();showResults(data,(performance.now()-start)/1000);
 }catch(e){setError(e.message);$('result-status').textContent='检索未完成';}finally{clearInterval(ticker);lock(false);}
};
fetch('/demo/info').then(checked).then(r=>r.json()).then(info=>{$('library').textContent=`本机视频库已连接 · ${info.num_media_files} 段研究素材 / ${info.num_vectors} 条特征 / 可定位播放`;}).catch(e=>{$('library').textContent='视频库连接失败';setError(e.message);});loadPeople().catch(()=>{});

let actionPollTimer=null, actionCoverageKey='';
async function loadActionStatus(){
 clearTimeout(actionPollTimer);
 let pending=false;
 try{
  const status=await(await checked(await fetch('/mvp-api/actions/status'))).json();
  pending=['pending','building'].includes(status.status);
  const summary=`动作索引：${status.indexed_videos} 个视频 / ${status.indexed_segments} 个片段 · ${status.message}`;
  $('action-coverage').textContent=summary;$('action-library-status').textContent=summary;
  $('action-progress').hidden=!pending;$('action-progress').value=status.progress;
  $('retry-action-index').hidden=status.status!=='failed';
  const key=status.indexed_videos+':'+status.indexed_segments;
  if(key!==actionCoverageKey&&!$('library-panel').hidden)window.refreshVideoLibrary?.();
  actionCoverageKey=key;
 }catch(e){$('action-coverage').textContent='动作索引状态暂不可用：'+e.message;}
 actionPollTimer=setTimeout(loadActionStatus,pending?2000:15000);
}
$('retry-action-index').onclick=async()=>{try{await checked(await fetch('/mvp-api/actions/retry',{method:'POST'}));await loadActionStatus();}catch(e){setError(e.message);}};
loadActionStatus();
