"""多线程静态文件服务（看板专用）

要点：
1) 多线程 —— 避免单线程 http.server 在 3MB data.js 并发下载时被阻塞导致 502。
2) HTML/JS/CSS 一律 no-store —— 保证「重新部署后刷新页面就能看到新版本」，
   不会出现浏览器拿旧 JS + 新 data.js 的混搭状态（会导致侧栏点击异常等怪问题）。
3) 端口读 PORT 环境变量，兜底 3000；绑定 0.0.0.0 以便反代访问。
4) 「⟳ 重建数据」按钮：提供 /__rebuild 接口，点一下在本机执行 build.py 重建数据。
   浏览器本身不能运行本地脚本，必须由这个本地服务代跑。

启动：python3 server.py   （或 python3 server.py 3000）
根目录自动探测：cwd / 脚本所在目录 / 常见上传目录，取第一个含 index.html 的。
"""
import os
import re
import sys
import json
import subprocess
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

PORT = int(os.environ.get('PORT') or (sys.argv[1] if len(sys.argv) > 1 else 3000))

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

# 这些类型不缓存：内容一变，刷新即可见
NO_STORE_EXT = ('.html', '.htm', '.js', '.css', '.json', '.map')
# 这些类型可长缓存（几乎不变）：图标等
LONG_CACHE_EXT = ('.svg', '.png', '.jpg', '.jpeg', '.gif', '.webp', '.ico', '.woff', '.woff2')

# ---------------------------------------------------------------- 本地重建能力
HERE = os.path.dirname(os.path.abspath(__file__))
BUILD_PY = os.path.join(HERE, 'build.py')
SCAN_EXCLUDE = {'.git', '_board', '_board_pub', '_board_v2', 'node_modules',
                '.workbuddy', 'assets', '__pycache__'}


def rebuild_available():
    """只在「本机源笔记目录 + build.py 都在」时才允许重建。

    安全兜底：对外部署的副本（云端沙箱）不含源笔记目录，绝不能执行构建，
    否则会扫到空目录、把好的 data.js 覆盖成空数据。判定依据（需同时满足）：
      1) build.py 就在脚本同目录；
      2) 脚本目录名是 _board（本地源看板的结构特征，部署副本不会叫这个）；
      3) 上级内容根里真的能扫到笔记（>=5 篇 .md）。
    """
    try:
        if not os.path.isfile(BUILD_PY):
            return False
        if os.path.basename(HERE) != '_board':
            return False
        root = os.path.dirname(HERE)
        n = 0
        for dp, dirs, files in os.walk(root):
            dirs[:] = [d for d in dirs if d not in SCAN_EXCLUDE]
            n += sum(1 for f in files if f.lower().endswith('.md'))
            if n >= 5:
                return True
        return False
    except Exception:
        return False


class Handler(SimpleHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=ROOT, **kwargs)

    def end_headers(self):
        path = self.path.split('?')[0].split('#')[0].lower()
        if path.endswith(NO_STORE_EXT) or path in ('/', ''):
            self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Expires', '0')
        elif path.endswith(LONG_CACHE_EXT):
            self.send_header('Cache-Control', 'public, max-age=86400')
        super().end_headers()

    def log_message(self, *args):  # 静音访问日志
        pass

    # ---------------- /__rebuild ----------------
    def _send_json(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        try:
            self.wfile.write(body)
        except Exception:
            pass

    def _is_rebuild(self):
        return self.path.split('?')[0].rstrip('/') == '/__rebuild'

    def do_GET(self):
        if self._is_rebuild():
            return self._send_json(200, {'available': rebuild_available()})
        return super().do_GET()

    def do_POST(self):
        if not self._is_rebuild():
            return self._send_json(404, {'ok': False, 'message': 'not found'})
        if not rebuild_available():
            return self._send_json(403, {
                'ok': False,
                'message': '仅本地可用：未找到源笔记目录或 build.py（云端不含源笔记）',
            })

        out = os.path.join(HERE, 'assets', 'data.js')
        backup = None
        try:
            if os.path.isfile(out):
                with open(out, 'rb') as f:
                    backup = f.read()

            res = subprocess.run(
                [sys.executable, BUILD_PY],
                cwd=HERE, capture_output=True, text=True, timeout=180,
            )
            if res.returncode != 0:
                if backup is not None:
                    with open(out, 'wb') as f:
                        f.write(backup)
                msg = (res.stderr or res.stdout or ('exit %d' % res.returncode)).strip()
                return self._send_json(500, {'ok': False, 'message': msg[-400:]})

            head = ''
            try:
                with open(out, 'r', encoding='utf-8', errors='replace') as f:
                    head = f.read(4000)
            except Exception:
                pass
            m_total = re.search(r'"total":\s*(\d+)', head)
            m_gen = re.search(r'"generatedAt":\s*"([^"]+)"', head)
            total = int(m_total.group(1)) if m_total else 0
            if not total:
                if backup is not None:
                    with open(out, 'wb') as f:
                        f.write(backup)
                return self._send_json(500, {'ok': False, 'message': '构建结果为空，已回滚到上一版'})

            return self._send_json(200, {
                'ok': True,
                'total': total,
                'generatedAt': m_gen.group(1) if m_gen else '',
            })
        except Exception as e:
            if backup is not None:
                try:
                    with open(out, 'wb') as f:
                        f.write(backup)
                except Exception:
                    pass
            return self._send_json(500, {'ok': False, 'message': str(e)[-400:]})


if __name__ == '__main__':
    print('serving %s on port %d' % (ROOT, PORT), flush=True)
    print('rebuild endpoint %s' % ('ENABLED (local)' if rebuild_available() else 'disabled'), flush=True)
    ThreadingHTTPServer.daemon_threads = True
    ThreadingHTTPServer.allow_reuse_address = True
    ThreadingHTTPServer(('0.0.0.0', PORT), Handler).serve_forever()
