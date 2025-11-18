import argparse, csv, os
from pathlib import Path
import cv2
from tqdm import tqdm

def list_videos(video_dir):
    exts={'.mp4','.mov','.avi','.mkv','.webm','.MP4','.MOV','.AVI','.MKV','.WEBM'}

    return [p for p in Path(video_dir).rglob('*') if p.suffix in exts]

def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def extract_video_frames(video_path: Path, out_dir: Path, target_fps: float=None, every_nth: int=None):
    cap=cv2.VideoCapture(str(video_path))
    if not cap.isOpened(): raise RuntimeError(f"Impossible d'ouvrir: {video_path}")
    orig_fps=cap.get(cv2.CAP_PROP_FPS) or 30.0
    total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if target_fps is not None:
        step=max(int(round((orig_fps or 30)/max(target_fps,1e-6))),1)
    elif every_nth is not None:
        step=max(int(every_nth),1)
    else:
        step=1
    out_sub=out_dir/video_path.stem; ensure_dir(out_sub)
    rows=[("frame_idx","filename","time_sec")]
    idx=saved=0
    pbar=tqdm(total=total, desc=video_path.stem)
    while True:
        ret,frame=cap.read()
        if not ret: break
        if idx % step==0:
            t=idx/max(orig_fps,1e-6); name=f"{video_path.stem}_f{idx:06d}.jpg"
            cv2.imwrite(str(out_sub/name), frame, [int(cv2.IMWRITE_JPEG_QUALITY),95])
            rows.append((idx,name,f"{t:.3f}")); saved+=1
        idx+=1; pbar.update(1)
    pbar.close(); cap.release()
    with open(out_sub/f"{video_path.stem}_frames_index.csv", 'w', newline='') as f:
        csv.writer(f).writerows(rows)
    return saved, total, orig_fps

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--video_dir', required=True)
    ap.add_argument('--out_dir', default='data/frames')
    g=ap.add_mutually_exclusive_group()
    g.add_argument('--target_fps', type=float)
    g.add_argument('--every_nth', type=int)
    args=ap.parse_args()
    vids=list_videos(args.video_dir)
    if not vids: return print('Aucune vidéo trouvée')
    ensure_dir(Path(args.out_dir))
    for vp in vids:
        s,t,fps=extract_video_frames(vp, Path(args.out_dir), args.target_fps, args.every_nth)
        print(f"OK: {vp.name} | fps≈{fps:.1f} | frames {s}/{t}")

if __name__=='__main__':
    main()