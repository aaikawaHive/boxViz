from flask import Flask
try:
    from urllib.parse import unquote  # PY3
except ImportError:
    from urllib import unquote  # PY2

app = Flask(__name__)

@app.route('/v1/<path:param>')  # NOTE: <path:param> is required to match /
def f(param=''):
    return (
        f"param: {param}\ndecoded param: {unquote(param)}\n",
        200,
        {'content-type': 'text/plain'}
    )

if __name__ == '__main__':
   app.run(debug=True, host='0.0.0.0', port=8888)
