# BoxViz

Web-based image viewer for viewing bounding boxes.

It is a tiny [flask](http://flask.pocoo.org/) app, built with [bootstrap](http://getbootstrap.com/),
which will display images and their meta information from a table with pagination. It could be a starting point for a more complex application.

**WARNING:** This app was built to run locally for my own convenience and
to be shared with a small number of collaborators.

- [this stackoverflow question](http://stackoverflow.com/questions/28141454/flask-using-a-global-variable-to-load-data-files-into-memory)
- flask documentation on databases


## Test out

You can either run in a new conda environment or use docker.

1. Create a new conda environment to install required packages locally:

```sh
cd imageviewer
conda env create    # will create 'viewer' environment specified in environment.yml
source activate viewer
flask --app run.py run --host '0.0.0.0' --port 8888
```

2. Use docker

```
cd imageviewer
docker-compose up --build
flask --app run.py run --host '0.0.0.0' --port 8888
```

There are other dependencies you might want to install like `torch` depending on the format of your groundtruths/predictions. Install as needed.

Configure static file paths in `__init__.py` and modify templates in `templates/`
according to your needs. [Flask](http://flask.pocoo.org/) uses the [jinja](http://jinja.pocoo.org/) template engine.
