"""Colab notebook helpers — extract the loop-heavy bits (data download, run streaming)
so notebook cells stay readable. Git push stays inline in the notebook (short, no loop).

Usage (after cloning the repo and os.chdir into it):
    from utils.colab import gh_token, download_cifar100, run_training
    GH_TOKEN = gh_token()
    download_cifar100()
    run_training(cmd)   # streams [Epoch] lines live
"""
import os
import tarfile
import subprocess
import urllib.request


def gh_token():
    """Return GH_TOKEN from Colab Secrets (fallback: env)."""
    try:
        from google.colab import userdata
        t = userdata.get('GH_TOKEN')
    except Exception:
        t = os.environ.get('GH_TOKEN', '')
    assert t, 'GH_TOKEN not set — add it to Colab Secrets (repo scope)'
    print('GH_TOKEN OK')
    return t


def download_cifar100(data, urls):
    """Download+extract CIFAR-100 (minimal MB progress). `urls` = mirror list (from notebook)."""
    os.makedirs(data, exist_ok=True)
    if os.path.exists(f'{data}/cifar-100-python'):
        print('CIFAR-100 ready.')
        return
    tar = f'{data}/cifar-100-python.tar.gz'
    for url in urls:
        try:
            print('downloading from', url)
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=60) as r, open(tar, 'wb') as f:
                total = int(r.headers.get('Content-Length', 0))
                done = 0
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)
                    if total:
                        print(f'\r  {done // 1048576}/{total // 1048576} MB ({done * 100 // total}%)',
                              end='', flush=True)
            print()
            tarfile.open(tar).extractall(data)
            print('ready via', url)
            return
        except Exception as e:
            print('fail', url, e)
    raise RuntimeError('CIFAR-100 download failed (all mirrors)')


def run_training(cmd, log='/content/run.log'):
    """Run cmd, stream stdout in the MAIN thread, print epoch lines live, tee to log.

    Robust vs the old daemon-thread tailer (which silently died -> stuck at ep0).
    """
    print('training... (epoch lines below; full log in', log, ')')
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    with open(log, 'w') as lf:
        for line in proc.stdout:
            lf.write(line)
            lf.flush()
            if '[Epoch' in line and ('[train]' in line or '[val]' in line):
                print(line.strip(), flush=True)
    proc.wait()
    print('done rc', proc.returncode)
    return proc.returncode
