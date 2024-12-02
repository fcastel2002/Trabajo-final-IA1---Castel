from flask import Flask, render_template_string, request
import os

app = Flask(__name__)

UPLOAD_FOLDER = '../anexos/img_uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def index():
    return render_template_string('''
        <!doctype html>
        <html lang="es">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Subir Imagen</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 0;
                    background-color: #f4f4f4;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                }
                .container {
                    background: white;
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
                    max-width: 400px;
                    width: 100%;
                    text-align: center;
                }
                h1 {
                    font-size: 24px;
                    color: #333;
                    margin-bottom: 20px;
                }
                form {
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                }
                input[type="file"] {
                    margin: 10px 0;
                    padding: 10px;
                    background-color: #f0f0f0;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    font-size: 16px;
                }
                input[type="submit"] {
                    padding: 12px 20px;
                    background-color: #4CAF50;
                    color: white;
                    border: none;
                    border-radius: 5px;
                    font-size: 16px;
                    cursor: pointer;
                    transition: background-color 0.3s;
                }
                input[type="submit"]:hover {
                    background-color: #45a049;
                }
                .message {
                    padding: 10px;
                    margin-top: 20px;
                    border-radius: 5px;
                    font-size: 16px;
                    color: white;
                }
                .success {
                    background-color: #4CAF50;
                }
                .error {
                    background-color: #f44336;
                }
                @media (max-width: 480px) {
                    h1 {
                        font-size: 20px;
                    }
                    .container {
                        padding: 15px;
                    }
                    input[type="file"] {
                        font-size: 14px;
                    }
                    input[type="submit"] {
                        font-size: 14px;
                    }
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Sube una Imagen</h1>
                <form method="POST" action="/upload" enctype="multipart/form-data">
                    <input type="file" name="file" accept="image/*">
                    <input type="submit" value="Subir">
                </form>
                {% if message %}
                <div class="message {{ message_type }}">
                    {{ message }}
                </div>
                {% endif %}
            </div>
        </body>
        </html>
    ''')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return render_template_string('''
            <!doctype html>
            <html lang="es">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Subir Imagen</title>
            </head>
            <body>
                <h1>Subir Imagen</h1>
                <form method="POST" action="/upload" enctype="multipart/form-data">
                    <input type="file" name="file" accept="image/*">
                    <input type="submit" value="Subir">
                </form>
                <div class="message error">
                    No se ha seleccionado ningún archivo.
                </div>
            </body>
            </html>
        ''')

    file = request.files['file']
    if file.filename == '':
        return render_template_string('''
            <!doctype html>
            <html lang="es">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Subir Imagen</title>
            </head>
            <body>
                <h1>Subir Imagen</h1>
                <form method="POST" action="/upload" enctype="multipart/form-data">
                    <input type="file" name="file" accept="image/*">
                    <input type="submit" value="Subir">
                </form>
                <div class="message error">
                    No se seleccionó ningún archivo.
                </div>
            </body>
            </html>
        ''')

    if file and allowed_file(file.filename):
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        return render_template_string('''
            <!doctype html>
            <html lang="es">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Subir Imagen</title>
            </head>
            <body>
                <h1>Subir Imagen</h1>
                <form method="POST" action="/upload" enctype="multipart/form-data">
                    <input type="file" name="file" accept="image/*">
                    <input type="submit" value="Subir">
                </form>
                <div class="message success">
                    Imagen subida con éxito: {{ filename }}
                </div>
            </body>
            </html>
        ''', filename=filename)

    return render_template_string('''
        <!doctype html>
        <html lang="es">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Subir Imagen</title>
        </head>
        <body>
            <h1>Subir Imagen</h1>
            <form method="POST" action="/upload" enctype="multipart/form-data">
                <input type="file" name="file" accept="image/*">
                <input type="submit" value="Subir">
            </form>
            <div class="message error">
                El archivo no es permitido.
            </div>
        </body>
        </html>
    ''')
