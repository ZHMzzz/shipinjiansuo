import {node, api, formatTime, formatSize} from './api.js';

export const trashIcon = '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M3 6h18M9 6V3h6v3M5 6l1 15h12l1-15M10 10v7M14 10v7"/></svg>';

export function createVideoDeletion(onDeleted) {
  const dialog=node('dialog',undefined,'video-delete-dialog');dialog.setAttribute('aria-labelledby','video-delete-title');
  dialog.innerHTML=`<button class="video-delete-close" aria-label="关闭删除确认">×</button><div class="video-delete-icon">${trashIcon}</div><h2 id="video-delete-title">删除这个视频？</h2><p class="video-delete-copy">将删除整个视频，并清理它的物体、人脸、自然语言检索索引和缩略图。此操作无法撤销。</p><div class="video-delete-target"><img alt=""><div><strong></strong><span></span></div></div><p class="video-delete-feedback" role="status" hidden></p><div class="video-delete-footer"><button class="outline-button" type="button">取消</button><button class="video-delete-confirm" type="button">删除视频</button></div>`;
  document.body.append(dialog);
  let selected=null,busy=false;
  const confirm=dialog.querySelector('.video-delete-confirm'),cancel=dialog.querySelector('.outline-button'),close=dialog.querySelector('.video-delete-close'),message=dialog.querySelector('.video-delete-feedback');
  close.onclick=cancel.onclick=()=>{if(!busy)dialog.close();};
  dialog.addEventListener('cancel',event=>{if(busy)event.preventDefault();});
  confirm.onclick=async()=>{
    if(busy || !selected)return;
    const item=selected;busy=true;confirm.disabled=cancel.disabled=close.disabled=true;confirm.textContent='正在删除…';
    message.hidden=false;message.classList.remove('error');message.textContent='正在清理视频和检索索引，请稍候…';
    try {
      const result=await api(`/mvp-api/videos/${item.id}`,{method:'DELETE',headers:{'If-Match':item.revision}});
      await onDeleted(item,result);dialog.close();
    } catch(error) {
      message.classList.add('error');message.textContent=error.message;
    } finally {busy=false;confirm.disabled=cancel.disabled=close.disabled=false;confirm.textContent='删除视频';}
  };
  return item=>{
    selected=item;message.hidden=true;
    dialog.querySelector('.video-delete-target img').src=new URL(item.thumbnail,location.origin+'/demo/').href;
    dialog.querySelector('strong').textContent=item.name;
    dialog.querySelector('.video-delete-target span').textContent=`${formatTime(item.duration)} · ${item.width} × ${item.height} · ${formatSize(item.size_in_bytes)}`;
    dialog.showModal();cancel.focus();
  };
}
