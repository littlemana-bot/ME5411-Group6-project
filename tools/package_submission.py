"""Create an explicitly labeled review package with source and result hashes."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    files = []
    for name in ['README.md', 'requirements.txt', 'requirements-tested.txt',
                 'segment_characters.py', 'run_task7.py', 'dataset_2026_updated.zip',
                 'ME5411_Project_AY2627.pdf']:
        path = ROOT/name
        if not path.is_file():
            raise FileNotFoundError(path)
        files.append(path)
    for folder in ['task7', 'tools', 'docs', 'output/segmentation', 'output/task7', 'output/report']:
        for path in (ROOT/folder).rglob('*'):
            if path.is_file() and not {'__pycache__', 'cache'} & set(path.parts) and path.suffix != '.pyc':
                files.append(path)
    files = sorted(set(files))
    pending = ['Four member names and student numbers', 'Part 2 lecture evidence for PCA and SVM',
               'Group-authored discussion rationale and conclusions', 'Actual contribution confirmation',
               'Final group review and Canvas submission']
    manifest = {'package_type': 'review draft, not final submission', 'group': '06',
                'pending': pending,
                'files': {str(p.relative_to(ROOT)).replace('\\','/'): {
                    'bytes': p.stat().st_size,
                    'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in files}}
    out = ROOT/'output/submission'
    out.mkdir(parents=True, exist_ok=True)
    target = out/'Group_06_review.zip'
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, 'Group_06/'+p.relative_to(ROOT).as_posix())
        z.writestr('Group_06/MANIFEST.json', json.dumps(manifest, indent=2))
        z.writestr('Group_06/REVIEW_STATUS.txt',
                   'REVIEW DRAFT. Do not submit without resolving the following items.\n\n'+
                   '\n'.join('- '+item for item in pending)+'\n')
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
        for name, entry in manifest['files'].items():
            assert hashlib.sha256(z.read('Group_06/'+name)).hexdigest() == entry['sha256']
    record = {'archive': target.name, 'sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
              'bytes': target.stat().st_size, 'files_verified': len(files), 'crc_and_sha256_verified': True,
              'package_type': manifest['package_type']}
    (out/'package_integrity.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(json.dumps(record,indent=2))


if __name__ == '__main__':
    main()
