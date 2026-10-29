import assert from 'node:assert/strict';
import {test} from 'node:test';
import {createVideoPlayer} from '../ui/video-player.js';

function setup() {
  const messages = [], loads = [];
  const video = {currentTime:0, duration:60, src:'', onerror:null, onloadedmetadata:null,
    pause() {}, removeAttribute() {this.src = '';},
    load() {loads.push(this.src); this.currentTime = 0;}, play() {return Promise.resolve();}};
  return {video, messages, loads, controller:createVideoPlayer(video, text => messages.push(text))};
}
const item = {url:'/original', fallback_url:'/compressed', start:12};

test('original first; a decode failure resumes compressed at current position once', () => {
  const {video, controller, messages, loads} = setup();
  controller.play(item); assert.equal(video.src, '/original');
  video.onloadedmetadata(); assert.equal(video.currentTime, 12);
  video.currentTime = 18; const oldError = video.onerror;
  video.onerror(); assert.equal(video.src, '/compressed');
  video.onloadedmetadata(); assert.equal(video.currentTime, 18);
  oldError(); assert.equal(messages.at(-1), '原视频无法播放，已切换兼容播放');
  video.onerror(); assert.equal(messages.at(-1), '视频播放失败，请关闭后重试');
  assert.equal(loads.filter(x => x === '/compressed').length, 1);
});

test('failure before metadata retains search start; files without originals never retry', () => {
  const {video, controller, loads} = setup();
  controller.play(item); video.onerror(); video.onloadedmetadata();
  assert.equal(video.currentTime, 12);
  controller.play({url:'/old', start:4}); video.onerror();
  assert.equal(loads.filter(x => x === '/old').length, 1);
});

test('fallback respects a user seeking backwards before the search hit', () => {
  const {video, controller} = setup();
  controller.play(item); video.onloadedmetadata(); video.currentTime = 3;
  video.onerror(); video.onloadedmetadata();
  assert.equal(video.currentTime, 3);
});

test('closing or switching invalidates previous errors and metadata callbacks', () => {
  const {video, controller} = setup();
  controller.play(item); const oldError = video.onerror, oldMetadata = video.onloadedmetadata;
  controller.stop(); oldError(); oldMetadata(); assert.equal(video.src, '');
  controller.play({url:'/second', start:2}); oldError(); oldMetadata();
  assert.equal(video.src, '/second'); assert.equal(video.currentTime, 0);
  video.onloadedmetadata(); assert.equal(video.currentTime, 2);
});

test('autoplay restrictions do not trigger fallback', async () => {
  const {video, controller} = setup();
  video.play = () => Promise.reject({name:'NotAllowedError'});
  controller.play(item); video.onloadedmetadata(); await Promise.resolve();
  assert.equal(video.src, '/original');
});

test('late unsupported play rejection cannot fail a successful fallback', async () => {
  const {video, controller, messages} = setup();
  let reject;
  video.play = () => new Promise((_, r) => {reject = r;});
  controller.play(item); video.onloadedmetadata(); video.onerror();
  reject({name:'NotSupportedError'}); await Promise.resolve();
  assert.equal(video.src, '/compressed');
  assert.equal(messages.at(-1), '原视频无法播放，已切换兼容播放');
});
