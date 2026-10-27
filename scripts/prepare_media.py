"""Prepare a small real-video corpus. Query pictures remain outside the index."""
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FFMPEG = '/opt/homebrew/opt/ffmpeg@7/bin/ffmpeg'

def run(*args):
    subprocess.run([FFMPEG, '-hide_banner', '-loglevel', 'error', '-y', *map(str, args)], check=True)

sources = []
for folder in sorted((ROOT / 'data/downloads/kinetics').iterdir()):
    if folder.is_dir():
        for p in sorted(folder.glob('*.mp4'))[:2]:
            sources.append((p, f'{folder.name}-{p.name}', 'https://thor.robots.ox.ac.uk/wise/assets/test/Kinetics-6.tar.gz'))
for name in ['backpack-walking', 'backpack-street']:
    sources.append((ROOT / f'data/downloads/{name}.mp4', f'{name}.mp4',
                    'https://www.pexels.com/video/' + ('6326811/' if name.endswith('walking') else '3551155/')))

manifest = []
for source, name, url in sources:
    dest = ROOT / 'data/videos' / name
    if not dest.exists():
        run('-i', source, '-an', '-vf', "scale=640:640:force_original_aspect_ratio=decrease:force_divisible_by=2", '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', dest)
    preview = ROOT / 'data/queries' / (dest.stem + '-preview.jpg')
    run('-ss', '2.25', '-i', dest, '-frames:v', '1', preview)
    manifest.append({'file': name, 'source': url, 'query_preview_time': 2.25, 'transform': 'H264, max 640 px, audio removed; original timeline retained'})
    print('Prepared', name, flush=True)
(ROOT / 'data/manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))

from PIL import Image, ImageDraw
sheet = Image.new('RGB', (960, ((len(manifest)+3)//4)*210), '#e7e7e7')
draw = ImageDraw.Draw(sheet)
for i, m in enumerate(manifest):
    im = Image.open(ROOT/'data/queries'/(Path(m['file']).stem+'-preview.jpg'))
    im.thumbnail((235, 175))
    x,y = (i%4)*240, (i//4)*210
    sheet.paste(im, (x,y))
    draw.text((x+2,y+177), m['file'][:30], fill='black')
sheet.save(ROOT/'data/contact-sheet.jpg')
