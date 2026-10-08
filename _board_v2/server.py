"""多线程静态文件服务（避免单线程 http.server 在大文件并发下被阻塞导致 502）
启动：python3 server.py [port]
根目录自动探测：cwd / 脚本所在目录 / 常见上传目录，取第一个含 index.html 的。
"""
import os
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 3000

CANDIDATES = [
    os.getcwd(),
    os.path.dirname(os.path.abspath(__file__)),
    os.path.join(os.getcwd(), 'dist'),
    os.path.join(os.getcwd(), 'public'),
    '/app',
]

ROOT = os.getcwd()
for c in CANDIDATES:
    try:
        if os.path.isfile(os.path.join(c, 'index.html')):
            ROOT = c
            break
    except Exception:
        pass


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=ROOT, **kwargs)

    def log_message(self, *args):  # 静音访问日志
        pass


if __name__ == '__main__':
    print('serving %s on port %d' % (ROOT, PORT), flush=True)
    ThreadingHTTPServer(('0.0.0.0', PORT), Handler).serve_forever()
