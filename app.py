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
def delete_photo(file_id):
    pass
    # print(f"delete_file called with file_id: {file_id}")
    # Logic to delete the file
# def delete_photo():
#     text = request.form.get('search', '')
#     image_path = request.form.get('image_path')  
#     uploaded_image = request.files.get('image', '')
#     try:
#         query = f"DELETE FROM photos_db WHERE path = '{image_path}'"
#         client.command(query)
#         # full_path = os.path.join('/Volumes/T7/photos_from_icloud', image_path)
#         if os.path.exists(image_path):
#             print('1')
#             print(image_path)
#             os.remove(image_path)
#             print(f"Deleted file: {image_path}")
#     except Exception as e:
#         print(f"Error deleting photo: {e}")
#     return redirect(url_for("home", image=uploaded_image.filename if uploaded_image else None))

@app.route("/", methods=['GET','POST'])
def home(text=None, image=None):
    new_file = None
    text = 'city busy centre'
    uploaded_image = 'hhdehh'
    text = request.form.get('search', '')
    uploaded_image = request.files.get('image', '')
    if uploaded_image != '':
        try:
            temp_path = "/Volumes/T7/photos_from_icloud/tmp/" + uploaded_image.filename
            uploaded_image.save(temp_path)
            print(f"Saved image to {temp_path}")
        except Exception as e:
            print(f"Error: {e}")
    environment = Environment(loader=FileSystemLoader("templates/"))
    template = environment.get_template("results.html")
    # if to_delete != '' and text not in ['reset', '']:
    #     try:
    #         os.remove(to_delete)
    #         print(f"Deleted {to_delete}")
    #     except Exception as e:
    #         print(f"Error deleting file: {e}")
    #         flash(f"Error deleting file: {e}", 'error')
    #     to_delete = ''
    try:
        if text != 'reset' and text != '':
            context = search.return_file('search', text=text, image=None, table='photos_db', limit=200, filter=None)
            return render_template("results.html", **context)
        # elif text == 'reset' or text is None:
        #     return render_template("results.html")

        elif uploaded_image != '' and uploaded_image.filename:
            context = search.return_file('search', text=None, image=temp_path, table='photos_db', limit=200, filter=None)
            return render_template("results.html", **context)
    except Exception as e:   
        print(e)
        print('1')
        return render_template("results.html")
    return render_template("results.html")

if __name__ == '__main__':
    app.run(debug=True)
    print(app.url_map)