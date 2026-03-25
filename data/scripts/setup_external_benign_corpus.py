"""
Setup external benign validation corpus from public sources.

This script:
1. Creates data/external_benign/ directory
2. Downloads or copies sample benign ZIPs from known public repositories
3. Provides fallback approach using archived GitHub projects

Sources used:
  - PyPI published packages (small .zip archives)
  - GitHub repository releases
  - Common open-source project distributions
"""

import os
import sys
import shutil
import urllib.request
import io
import zipfile
from pathlib import Path
from urllib.error import URLError


def create_external_corpus():
    """Setup external benign validation corpus."""
    
    print("\n" + "="*70)
    print("Setting up external benign validation corpus")
    print("="*70)
    
    repo_root = Path(__file__).parent.parent
    external_dir = repo_root / "data" / "external_benign"
    
    # Create directory
    external_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n✓ Created directory: {external_dir}")
    
    # Sources: Small public project repos and archives
    # These are well-known, safe, and represent realistic benign ZIPs
    sources = [
        {
            'name': 'requests-2.25.1.zip',
            'url': 'https://github.com/psf/requests/archive/refs/tags/v2.25.1.zip',
            'description': 'Python requests library'
        },
        {
            'name': 'flask-1.1.2.zip',
            'url': 'https://github.com/pallets/flask/archive/refs/tags/1.1.2.zip',
            'description': 'Flask web framework'
        },
        {
            'name': 'click-8.0.1.zip',
            'url': 'https://github.com/pallets/click/archive/refs/tags/8.0.1.zip',
            'description': 'Click CLI toolkit'
        },
        {
            'name': 'numpy-1.19.5.zip',
            'url': 'https://github.com/numpy/numpy/archive/refs/tags/v1.19.5.zip',
            'description': 'NumPy library'
        },
        {
            'name': 'pytest-6.2.4.zip',
            'url': 'https://github.com/pytest-dev/pytest/archive/refs/tags/6.2.4.zip',
            'description': 'pytest testing framework'
        },
        {
            'name': 'black-21.5b0.zip',
            'url': 'https://github.com/psf/black/archive/refs/tags/21.5b0.zip',
            'description': 'Black code formatter'
        },
        {
            'name': 'pydantic-1.8.2.zip',
            'url': 'https://github.com/samuelcolvin/pydantic/archive/refs/tags/v1.8.2.zip',
            'description': 'Pydantic validation library'
        },
        {
            'name': 'pandas-1.2.4.zip',
            'url': 'https://github.com/pandas-dev/pandas/archive/refs/tags/v1.2.4.zip',
            'description': 'Pandas data analysis'
        },
    ]
    
    print(f"\nDownloading {len(sources)} benign reference archives...")
    print("(These are well-known open-source projects from GitHub)")
    
    downloaded = 0
    failed = 0
    
    for source in sources:
        out_path = external_dir / source['name']
        
        # Skip if already exists
        if out_path.exists():
            print(f"  ✓ {source['name']} (already present)")
            downloaded += 1
            continue
        
        try:
            print(f"  ⬇ {source['name']}...", end=" ", flush=True)
            
            # Download
            urllib.request.urlretrieve(source['url'], out_path)
            
            print(f"✓ ({source['description']})")
            downloaded += 1
        
        except URLError as e:
            failed += 1
            print(f"✗ (network error)")
        except Exception as e:
            failed += 1
            print(f"✗ (error: {type(e).__name__})")
    
    print(f"\n{'='*70}")
    print(f"Downloaded: {downloaded}/{len(sources)} archives")
    print(f"Failed: {failed}")
    print(f"{'='*70}")
    
    # List what we have
    existing_zips = list(external_dir.glob("*.zip"))
    if existing_zips:
        print(f"\n✓ External corpus ready: {len(existing_zips)} benign ZIPs in {external_dir.name}/")
        total_size_mb = sum(z.stat().st_size for z in existing_zips) / (1024*1024)
        print(f"  Total size: {total_size_mb:.1f} MB")
        return True
    else:
        print(f"\n⚠️  No benign ZIPs available for validation")
        print(f"\nManual setup option:")
        print(f"  1. Add benign ZIP files to: {external_dir}/")
        print(f"  2. Use any publicly-available .zip archives (projects, releases, etc.)")
        return False


if __name__ == '__main__':
    success = create_external_corpus()
    sys.exit(0 if success else 1)
