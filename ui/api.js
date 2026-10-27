export const el = id => document.getElementById(id);
export function node(tag, text, cls) {
  const element = document.createElement(tag);
  if (text !== undefined) element.textContent = text;
  if (cls) element.className = cls;
  return element;
}
export function errorMessage(data, status) {
  return typeof data?.detail === 'string' ? data.detail : `请求失败（${status}），请重试`;
}
export async function api(url, options = {}) {
  const response = await fetch(url, {...options, cache: 'no-store'});
  const data = await response.json().catch(() => null);
  if (!response.ok) throw new Error(errorMessage(data, response.status));
  if (data === null) throw new Error('服务返回了无法读取的数据');
  return data;
}
export function feedback(id, text = '', error = false) {
  const target = el(id); target.textContent = text; target.hidden = !text;
  target.classList.toggle('error', error);
}
export const formatTime = n => `${Math.floor((n || 0) / 60).toString().padStart(2, '0')}:${Math.floor((n || 0) % 60).toString().padStart(2, '0')}`;
export const formatSize = n => n >= 1024 ** 3 ? `${(n / 1024 ** 3).toFixed(2)} GB` : `${((n || 0) / 1024 ** 2).toFixed(1)} MB`;
