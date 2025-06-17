# wsgi.py
from main import app
from mangum import Mangum

handler = Mangum(app)
