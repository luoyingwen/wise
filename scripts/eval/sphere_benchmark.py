"""Run the four Sphere releases with WISE's visual model and exact FAISS search."""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import traceback
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

DATASETS = [
    ('sphere-benchmark-20261028-r02', 'sphere-cards-20261028-r01-proxy-media-47-old.zip'),
    ('sphere-public-video-jonghwa-12v-32q-20261104-r01', 'SP_eval_set_provided_by_Jonghwa.zip'),
    ('sphere-public-tsv360-cross-video-agreement-full-queries-20261116-r01', 'videos_fix_format_74_cross_video_agreement.zip'),
    ('sphere-public-qvhighlights-test-s50-20261113-r01', 'qvhighlights-test-s50-20261113-r01.zip'),
]
MODEL = 'mlfoundations/open_clip/ViT-B-16-SigLIP2-512/webli'
SIZES = (2, 4, 8, 16, 32, 64)


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def align(archive, media):
    by_size = {m['size'] for m in media}
    members = defaultdict(list)
    for entry in archive.infolist():
        if Path(entry.filename).suffix.lower() == '.mp4' and entry.file_size in by_size:
            with archive.open(entry) as stream:
                members[hashlib.file_digest(stream, 'sha256').hexdigest()].append(entry)
    mapping = {}
    failures = []
    for m in media:
        matches = members[m['sha256'].lower()]
        if len(matches) != 1:
            failures.append(f"{m['path']} ({len(matches)} SHA-256 matches)")
        else:
            mapping[m['asset_id']] = matches[0]
    return mapping, failures


def encode_video(path, model, ffmpeg, np):
    from PIL import Image
    cmd = [ffmpeg, '-nostdin', '-v', 'error', '-i', str(path), '-map', '0:v:0',
           '-vf', 'fps=2,scale=512:512:force_original_aspect_ratio=increase,crop=512:512,setsar=1',
           '-pix_fmt', 'rgb24', '-threads', '1', '-f', 'rawvideo', '-']
    vectors, batch = [], []
    with subprocess.Popen(cmd, stdout=subprocess.PIPE) as proc:
        try:
            while data := proc.stdout.read(512 * 512 * 3):
                batch.append(Image.frombytes('RGB', (512, 512), data))
                if len(batch) == 24:
                    vectors.extend(f.vectors[0] for f in model.extract_image_features(model.preprocess_image(batch)))
                    batch.clear()
            if proc.wait() or not (vectors or batch):
                raise RuntimeError(f'FFmpeg failed or no frames: {path.name}')
            if batch:
                vectors.extend(f.vectors[0] for f in model.extract_image_features(model.preprocess_image(batch)))
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
    return np.asarray(vectors, dtype='float32')


def evaluate(archive, mapping, cards, queries, model, ffmpeg, scratch):
    import faiss
    import numpy as np
    import torch
    from sphere_eval.eval.metrics import score_query, summarize

    asset_vectors, asset_ids, moment_vectors, moments = [], [], [], []
    for i, (asset, entry) in enumerate(mapping.items(), 1):
        print(f'  [{i}/{len(mapping)}] {entry.filename}', flush=True)
        path = scratch / 'video.mp4'
        with archive.open(entry) as src, path.open('wb') as dst:
            shutil.copyfileobj(src, dst)
        with torch.autocast('cuda', dtype=torch.float16, enabled=torch.cuda.is_available()):
            frames = encode_video(path, model, ffmpeg, np)
        path.unlink()
        asset_ids.extend([asset] * len(frames))
        asset_vectors.extend(frames)
        duration = min(float(cards[asset]['duration_s']), len(frames) / 2)
        for seconds in SIZES:
            for start in range(0, int(duration * 2), seconds):
                end = min(start / 2 + seconds, duration)
                if end - start / 2 < seconds / 2:
                    continue
                moment_vectors.append(frames[start:round(end * 2)].mean(axis=0))
                moments.append(dict(asset_id=asset, t_start=start / 2, t_end=end))
    indexes = []
    for vectors in (asset_vectors, moment_vectors):
        matrix = np.ascontiguousarray(vectors, dtype='float32')
        faiss.normalize_L2(matrix)
        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)
        indexes.append(index)
    rows = []
    for offset in range(0, len(queries), 32):
        batch = queries[offset:offset + 32]
        text = np.ascontiguousarray(model.extract_text_features([q['request']['query'] for q in batch]), dtype='float32')
        for q, vector in zip(batch, text):
            hits = []
            is_asset = q['grain'] == 'asset'
            index = indexes[0 if is_asset else 1]
            limit = min(q['request'].get('top_k', 10), 10)
            scores, ranks = index.search(vector[None], index.ntotal)
            seen = set()
            for score, rank in zip(scores[0], ranks[0]):
                if is_asset:
                    asset = asset_ids[rank]
                    if asset in seen:
                        continue
                    seen.add(asset)
                    hit = dict(asset_id=asset, t_start=0.0, t_end=cards[asset]['duration_s'])
                else:
                    hit = moments[rank]
                    if any(old['asset_id'] == hit['asset_id'] and
                           max(0, min(old['t_end'], hit['t_end']) - max(old['t_start'], hit['t_start'])) /
                           (max(old['t_end'], hit['t_end']) - min(old['t_start'], hit['t_start'])) >= .5 for old in hits):
                        continue
                hits.append({**hit, 'score': float(score)})
                if len(hits) >= limit:
                    break
            rows.append({**q, **score_query(q, hits)})
    return {key: summarize([r for r in rows if key == 'all' or r['grain'] == key])
            for key in ('all', 'asset', 'moment') if key == 'all' or any(r['grain'] == key for r in rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sphere-eval', type=Path, required=True)
    parser.add_argument('--archives', type=Path, required=True)
    parser.add_argument('--ffmpeg', default='ffmpeg')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--align-only', action='store_true')
    args = parser.parse_args()
    sys.path.insert(0, str(args.sphere_eval.resolve()))
    releases = args.sphere_eval / 'datasets' / 'releases'
    lines = ['# WISE × Sphere 视觉检索基准', '',
             f'- WISE 模型：`{MODEL}`；FFmpeg 2 fps，CUDA FP16 帧推理（若可用），全库 FAISS 精确余弦搜索。',
             '- 视频级：按最高帧分数给视频排序。片段级：2/4/8/16/32/64 秒窗口，半窗步长，归一化均值特征，同视频 tIoU ≥ 0.5 去重。',
             '- 直接采用 Sphere score_query/summarize（匹配 tIoU ≥ 0.5），按原始 top_k 返回、最多 10 项；R@10 为 Sphere 归一化 Recall。',
             '- 不启用音频、元数据、拒答阈值或 WISE Web API；指标衡量 WISE 视觉模型的离线基线。',
             '- 各集严格校验整个候选库的 SHA-256；每集独立临时目录和内存索引，不读取或删除现有 WISE 项目。', '']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text('\n'.join(lines), encoding='utf-8')
    model = None
    model_error = None
    failed = False
    for release, zipname in DATASETS:
        print(release, flush=True)
        section = [f'## {release}', '']
        try:
            benchmark = load(releases / release / 'benchmark.json')
            queries = benchmark['queries']
            pinned = releases / release / 'card-set.json'
            if not pinned.exists():
                pinned = releases / benchmark['card_set_id'] / 'card-set.json'
            card_set = load(pinned)
            cards = {c['asset_id']: c for c in card_set['cards']}
            media_path = releases / release / 'media.json'
            media = load(media_path) if media_path.exists() else card_set['media']
            if isinstance(media, dict):
                media = media['media']
            with ZipFile(args.archives / zipname) as archive:
                mapping, failures = align(archive, media)
                expected = {h['asset_id'] for q in queries for h in q.get('expected_hits', [])}
                absent = (set(cards) | expected) - mapping.keys()
                section.extend([f'查询数：{len(queries)}；SHA-256 对齐：{len(mapping)}/{len(cards)}。', ''])
                if failures or absent or mapping.keys() - cards.keys():
                    section += ['**跳过：归档媒体与发布标注不能完整对齐。**', '']
                    section.extend(f'- {detail}' for detail in failures)
                    if absent:
                        section.append(f'- 缺少 {len(absent)} 个发布资产。')
                elif args.align_only:
                    section += ['**媒体对齐通过；未运行精度测试。**']
                else:
                    if model_error:
                        raise RuntimeError(model_error)
                    if model is None:
                        try:
                            from wise.feature.feature_extractor_factory import FeatureExtractorFactory
                            model = FeatureExtractorFactory(MODEL, {MODEL: {'compile': False}})
                        except Exception as exc:
                            model_error = f'WISE 模型加载失败：{exc}'
                            raise
                    with tempfile.TemporaryDirectory(prefix='wise-sphere-') as tmp:
                        summaries = evaluate(archive, mapping, cards, queries, model, args.ffmpeg, Path(tmp))
                    section += ['| 粒度 | 评分查询数 | R@1 | R@10 | nDCG@10 | 平均 tIoU | 无结果误报 |',
                                '|---|---:|---:|---:|---:|---:|---:|']
                    for grain, s in summaries.items():
                        vals = [s[k] for k in ('recall_at_1', 'recall_at_k', 'ndcg_at_k', 'mean_tiou')]
                        fields = ' | '.join('—' if v is None else f'{v:.4f}' for v in vals)
                        nr = s['no_result']
                        section.append(f"| {grain} | {s['n']} | {fields} | {nr['false_positives']}/{nr['n']} |")
        except Exception as exc:
            traceback.print_exc()
            failed = True
            section += [f'**运行失败：** {type(exc).__name__}: {exc}']
        with args.output.open('a', encoding='utf-8') as stream:
            stream.write('\n' + '\n'.join(section) + '\n\n')
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())
