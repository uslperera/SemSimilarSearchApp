# ----------------------------------------------------------------------------#
# Imports
# ----------------------------------------------------------------------------#

from flask import Flask, render_template, request
# from flask.ext.sqlalchemy import SQLAlchemy
import logging
from logging import Formatter, FileHandler
import os, json
from semsimilar.core.textprocessor.tokenize import CodeTokenizer
from semsimilar.core.similarity.corpus.hal import HAL
from semsimilar.core.similarity.main import similarity
from semsimilar.core.model.document import Document

# ----------------------------------------------------------------------------#
# App Config.
# ----------------------------------------------------------------------------#

app = Flask(__name__)
app.config.from_object('config')
# db = SQLAlchemy(app)

documents = []
hal_model = None

# Automatically tear down SQLAlchemy.
'''
@app.teardown_request
def shutdown_session(exception=None):
    db_session.remove()
'''

# Login required decorator.
'''
def login_required(test):
    @wraps(test)
    def wrap(*args, **kwargs):
        if 'logged_in' in session:
            return test(*args, **kwargs)
        else:
            flash('You need to login first.')
            return redirect(url_for('login'))
    return wrap
'''


# ----------------------------------------------------------------------------#
# Controllers.
# ----------------------------------------------------------------------------#


@app.route('/')
def index():
    return render_template('pages/home.html')


@app.route('/search/')
@app.route('/search/<query>')
def search(query=None):
    if query is None:
        return ""
    new_document = Document(0, query, None, None)
    results = similarity(documents, new_document, app.hal_model, 5)
    query_results = []
    for top_doc, score in results:
        query_results.append(top_doc.title)
    # sample = ["Cras justo odio", "Dapibus ac facilisis in", "Morbi leo risus", "Porta ac consectetur ac"]
    return json.dumps(query_results)


# Error handlers.

@app.errorhandler(500)
def internal_error(error):
    # db_session.rollback()
    return render_template('errors/500.html'), 500


@app.errorhandler(404)
def not_found_error(error):
    return render_template('errors/404.html'), 404


if not app.debug:
    file_handler = FileHandler('error.log')
    file_handler.setFormatter(
            Formatter('%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]')
    )
    app.logger.setLevel(logging.INFO)
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    app.logger.info('errors')


def initialize_corpus(count):
    Document.set_tokenizer(CodeTokenizer())
    with open(
            '/Users/shamal/Documents/IIT/Project/Development/SemSimilar/semsimilar/tests/data/100posts.json') as posts_file:
        posts = json.loads(posts_file.read())

    for i, post in enumerate(posts):
        if i == count:
            break
        documents.append(Document(post['Id'], post['Title'], post['Body'], post['Tags']))

    with open('/Users/shamal/Documents/IIT/Project/Development/SemSimilar/semsimilar/tests/data/100duplicates.json') as posts_file:
        duplicate_posts = json.loads(posts_file.read())

    for i, post in enumerate(duplicate_posts):
        if i == count:
            break
        documents.append(Document(post['Id'], post['Title'], post['Body'], post['Tags']))

    texts = []
    for doc in documents:
        texts.append(" ".join(doc.get_stemmed_tokens()))

    app.hal_model = HAL(documents=texts)
    print("---corpus created---")


initialize_corpus(100)

# ----------------------------------------------------------------------------#
# Launch.
# ----------------------------------------------------------------------------#

# Default port:
if __name__ == '__main__':
    app.run()

# Or specify port manually:
'''
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
'''