from flask import Flask, render_template, request, jsonify, Blueprint, flash, redirect, url_for, session, send_from_directory, make_response
import os
import search
import subprocess
from urllib.parse import unquote
from jinja2 import FileSystemLoader, Environment
import clickhouse_connect
client = clickhouse_connect.get_client(host=os.environ.get('CLICKHOUSE_HOST', 'localhost'),
                                           username=os.environ.get('CLICKHOUSE_USERNAME', 'default'),
                                           password=os.environ.get('CLICKHOUSE_PASSWORD', ''),
                                           port=os.environ.get('CLICKHOUSE_PORT', 8123))

basedir = os.path.abspath(os.path.dirname(__file__))
app = Flask(__name__, static_folder='/Volumes/T7/photos_from_icloud')
app.secret_key = "secret"
app.jinja_env.globals.update(unquote=unquote)
# @app.route("/clear_cookies")
# def clear_cookies():
#     response = make_response(redirect(url_for("home")))
#     for cookie in request.cookies:
#         response.delete_cookie(cookie)
#     return response
@app.route('/files/<path:filename>')
def serve_file(filename):
    decoded_filename = unquote(filename)
    decoded_filename = decoded_filename.replace('Volumes/T7/photos_from_icloud/', '')
    print(decoded_filename)
    return send_from_directory('/Volumes/T7/photos_from_icloud', decoded_filename)

@app.route('/delete_photo', methods=['POST'])
def delete_photo():
    image_path = request.form.get('image_path') 
    search_text = request.form.get("search") or request.args.get("search") or ""

    try:
        # Delete from DB
        query = f"DELETE FROM photos_db WHERE path = '{image_path}'"
        client.command(query)

        # Delete from filesystem
        if os.path.exists(image_path):
            os.remove(image_path)
            print(f"Deleted file: {image_path}")
    except Exception as e:
        print(f"Error deleting photo: {e}")

    print(search_text)
    return redirect(url_for("home", search_text=search_text))
@app.route("/", methods=['GET','POST'])
def home(search_text=""):
    
    uploaded_image = None
    saved_image_path = None
    try: search_text = search_text
    except: search_text = request.form.get("search_text")
    limit = request.form.get('limit', 50, type=int)
    if request.method == 'POST':
        session["search_text"] = request.form.get("search_text", "")
        uploaded_image = request.files.get('image')

        if uploaded_image and uploaded_image.filename != '':
            try:
                saved_image_path = os.path.join("/Volumes/T7/photos_from_icloud/tmp", uploaded_image.filename)
                uploaded_image.save(saved_image_path)
                print(f"Saved image to {saved_image_path}")
            except Exception as e:
                print(f"Error saving image: {e}")
    search_text = session.get("search_text", "")
    environment = Environment(loader=FileSystemLoader("templates/"))
    template = environment.get_template("results.html")

    try:
        if search_text and search_text != 'reset':
            context = search.return_file(
                'search', text=search_text, image=None,
                table='photos_db', limit=limit, filter=None
            )
            return render_template("results.html", **context)

        if saved_image_path:
            context = search.return_file(
                'search', text=None, image=saved_image_path,
                table='photos_db', limit=limit, filter=None
            )
            return render_template("results.html", **context)
    except Exception as e:
        print(f"Error during search: {e}")
        return render_template("results.html")

    return render_template("results.html")
if __name__ == '__main__':
    app.run(debug=True)
    print(app.url_map)