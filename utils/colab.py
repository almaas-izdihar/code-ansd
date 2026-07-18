"""Colab notebook helpers — keep run-notebook cells thin.

Usage (after cloning the repo and os.chdir into it):
    from utils.colab import gh_token, download_cifar100, run_training, push_results
    GH_TOKEN = gh_token()
    download_cifar100()
    run_training(cmd)                                   # streams [Epoch] lines live
    push_results('code-xxx', DIR, MODEL, EXP, GH_TOKEN) # copy log -> results/ + git push
"""
import os
import glob
import shutil
import tarfile
import datetime
import subprocess
import urllib.request

USER = 'almaas-izdihar'
EMAIL = 'almaasizdihar@gmail.com'
CIFAR100_URLS = [
    'https://data.brainchip.com/dataset-mirror/cifar100/cifar-100-python.tar.gz',
    'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz',
]


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


def setup_repo(repo_url, work_dir):
    """Clone if absent, pull main, chdir into it."""
    if not os.path.exists(work_dir):
        subprocess.run(f'git clone {repo_url} {work_dir}', shell=True, check=True)
    subprocess.run(f'git -C {work_dir} pull origin main', shell=True, check=True)
    os.chdir(work_dir)
    print(subprocess.check_output('git log --oneline -1', shell=True).decode().strip())


def download_cifar100(data='/content/data'):
    """Download+extract CIFAR-100 (fallback mirror, minimal MB progress)."""
    os.makedirs(data, exist_ok=True)
    if os.path.exists(f'{data}/cifar-100-python'):
        print('CIFAR-100 ready.')
        return
    tar = f'{data}/cifar-100-python.tar.gz'
    for url in CIFAR100_URLS:
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


def push_results(repo, work_dir, model, exp_glob, token):
    """Copy models/*{exp_glob}*/log/log.txt -> results/{model}/ and git-push."""
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    dest = f'{work_dir}/results/{model}'
    os.makedirs(dest, exist_ok=True)
    logs = glob.glob(f'models/*{exp_glob}*/log/log.txt')
    if not logs:
        print('WARNING: no log found for', exp_glob)
    for lg in logs:
        et = lg.split('/')[-3]
        shutil.copy(lg, f'{dest}/{ts}_{et}.txt')
        print('->', et)
    remote = f'https://oauth2:{token}@github.com/{USER}/{repo}'

    def run(c):
        r = subprocess.run(c, shell=True, cwd=work_dir, capture_output=True, text=True)
        if r.stdout.strip():
            print(r.stdout.strip()[:200])
        if r.returncode:
            raise RuntimeError(r.stderr.strip())

    run(f"git config user.email '{EMAIL}'")
    run(f"git config user.name '{USER}'")
    run('git pull --rebase origin main')
    run(f'git add results/{model}')
    run(f"git commit -m 'results: {model} {ts}'")
    run(f'git push {remote} HEAD:main')
    print('pushed results/' + model)
