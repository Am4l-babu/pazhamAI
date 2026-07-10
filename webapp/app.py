import os
import uuid
from datetime import datetime

from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from werkzeug.utils import secure_filename
from model_utils import predict_from_image

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-only-secret-key-change-me')

# Vercel's filesystem is read-only at runtime except /tmp, so uploads have to
# live there rather than under static/.
app.config['UPLOAD_FOLDER'] = '/tmp/pazham_uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload size

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp', 'bmp'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.context_processor
def inject_current_year():
    return {'current_year': datetime.now().year}


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/team')
def team():
    return render_template('team.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        flash('No image was submitted. Please choose a file or use the camera.')
        return redirect(url_for('index'))
    file = request.files['image']
    if file.filename == '':
        flash('No image was submitted. Please choose a file or use the camera.')
        return redirect(url_for('index'))
    if not allowed_file(file.filename):
        flash('Unsupported file type. Please upload an image file.')
        return redirect(url_for('index'))

    original_name = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex}_{original_name}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
    file.save(filepath)

    result = predict_from_image(filepath)
    return render_template('result.html', result=result, filename=unique_name)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.errorhandler(413)
def file_too_large(e):
    flash('That image is too large. Please upload a file under 16 MB.')
    return redirect(url_for('index'))


if __name__ == '__main__':
    debug_mode = os.environ.get('FLASK_DEBUG', '0') == '1'
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=debug_mode, threaded=True)
