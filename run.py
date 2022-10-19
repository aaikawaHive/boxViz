from boxViz.app import app

import boxViz.views

if __name__ == '__main__':
    exit(app.run(host='0.0.0.0', port=8888, debug=True))
