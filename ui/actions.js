(() => {
 const el=id=>document.getElementById(id);let report=null;
 const node=(tag,text,cls)=>{const e=document.createElement(tag);if(text!==undefined)e.textContent=text;if(cls)e.className=cls;return e;};
 const gb=n=>(n/1024**3).toFixed(2)+' GiB';
 function render(){
  const q=report.queries[Number(el('experiment-query').value)];if(!q)return;
  const grid=el('experiment-results');grid.replaceChildren();
  const first=q.ranking.findIndex(hit=>q.expected.includes(hit.id));
  el('experiment-status').textContent=`记录中的查询耗时 ${(q.encode_seconds+q.search_seconds).toFixed(3)} 秒 · 第一个符合人工标注的片段：${first<0?'未召回':`第 ${first+1} 名`}。${q.category==='no_helmet'?'注意：此查询存在将戴帽工人排在前面的误检，不能据此作出违规判定。':''}`;
  q.ranking.forEach((hit,i)=>{
   const item=report.clips.find(c=>c.id===hit.id);const card=node('article',undefined,'experiment-card');
   const video=node('video');video.controls=true;video.preload='none';video.playsInline=true;
   video.src='/action-media/'+encodeURIComponent(item.file.split('/').pop());
   video.poster=video.src.replace(/\.mp4$/,'.jpg');
   card.append(node('h3',`${String(i+1).padStart(2,'0')} · ${item.note}`),video);
   card.append(node('p',`相似度 ${hit.score.toFixed(4)} · ${item.duration.toFixed(2)} 秒片段`));
   const correct=q.expected.includes(hit.id);card.append(node('p',correct?'符合本次人工标注':'不属于该查询的人工正例',correct?'match':'miss'));
   const link=node('a','素材来源 ↗');link.href=item.source;link.target='_blank';link.rel='noopener noreferrer';card.append(link);grid.append(card);
  });
 }
 async function load(){
  const selectedText=el('experiment-query').selectedOptions[0]?.textContent;
  el('experiment-run').disabled=true;el('experiment-query').disabled=true;
  el('experiment-error').hidden=true;el('experiment-stats').replaceChildren();el('experiment-results').replaceChildren();
  el('experiment-status').textContent='正在读取实验记录…';
  try{
   const response=await fetch('/mvp-api/action-report?run='+el('experiment-run').value);
   if(!response.ok)throw new Error('本机没有此轮实验记录；实验数据不随源码交付，请返回检索台使用视频检索。');report=await response.json();
   if(report.status!=='completed')throw new Error('此轮实验尚未完成或未成功，请查看实验日志。');
   const peak=report.sampled_metal_peaks?.mps_driver_bytes;
   const observed=Math.max(...report.clips.map(c=>c.memory.mps_driver_bytes));
   for(const [label,value,detail] of [
    ['素材总时长',report.total_video_seconds.toFixed(1)+' 秒',`${report.clips.length} 个短片 / 每段 16 帧`],
    ['视频特征提取',report.total_embedding_seconds.toFixed(1)+' 秒','不含下载、转码和首次模型加载'],
    ['模型首次加载',report.load_seconds.toFixed(1)+' 秒','权重已缓存于本机'],
    [peak?'Metal 分配峰值':'Metal 阶段观测最大值',gb(peak||observed),peak?'约每 50ms 采样':'仅在每段结束后记录，非严格峰值']]){
    const card=node('div',undefined,'stat');card.append(node('span',label),node('strong',value),node('small',detail));el('experiment-stats').append(card);
   }
   el('experiment-query').replaceChildren(...report.queries.map((q,i)=>{const o=node('option',q.text);o.value=i;return o;}));
   const selectedIndex=report.queries.findIndex(q=>q.text===selectedText);
   if(selectedIndex>=0)el('experiment-query').value=selectedIndex;
   render();
  }catch(e){el('experiment-error').hidden=false;el('experiment-error').textContent=e.message;el('experiment-status').textContent='';}
  finally{el('experiment-run').disabled=false;el('experiment-query').disabled=!report;}
 }
 el('experiment-run').onchange=load;el('experiment-query').onchange=render;load();
})();
