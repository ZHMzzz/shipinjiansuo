// Each opening owns its handlers; closing or reopening invalidates old work.
export function createVideoPlayer(player, report) {
  let generation = 0;
  function stop() {
    generation++;
    player.onerror = null;
    player.onloadedmetadata = null;
    player.pause();
    player.removeAttribute('src');
    player.load();
  }
  function play(item) {
    stop();
    const current = generation;
    let attempt = 0;
    let fallback = false;
    let metadataLoaded = false;
    let position = Math.max(0, item.start || 0);
    function fail(failedAttempt) {
      if (current !== generation || failedAttempt !== attempt) return;
      if (!fallback && item.fallback_url && item.fallback_url !== item.url) {
        if (metadataLoaded && Number.isFinite(player.currentTime)) position = Math.max(0, player.currentTime);
        fallback = true;
        report('原视频无法播放，已切换兼容播放');
        load(item.fallback_url);
      } else {
        report('视频播放失败，请关闭后重试');
      }
    }
    function load(url) {
      const currentAttempt = ++attempt;
      metadataLoaded = false;
      player.onloadedmetadata = () => {
        if (current !== generation || currentAttempt !== attempt) return;
        player.currentTime = Math.min(position, Number.isFinite(player.duration) ? player.duration : position);
        metadataLoaded = true;
        player.play().catch(error => {
          // Autoplay restrictions need user interaction, not a lower-quality file.
          if (error.name === 'NotSupportedError') fail(currentAttempt);
        });
      };
      player.onerror = () => fail(currentAttempt);
      player.src = url;
      player.load();
    }
    report('');
    load(item.url);
  }
  return {play, stop};
}
