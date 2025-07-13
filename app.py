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
def delete_photo():
    image_path = request.form.get('image_path')  
    search = request.form.get('search', '')

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

    # Redirect with current search parameter
    return redirect(url_for("home", search=search))
@app.route("/", methods=['GET','POST'])
def home(text=None, image=None):
    # new_file = None
    # text = 'city busy centre'
    # uploaded_image = 'hhdehh'
    # text = request.form.get('search', '')
    # uploaded_image = request.files.get('image', '')
    # if uploaded_image != '':
    #     try:
    #         temp_path = "/Volumes/T7/photos_from_icloud/tmp/" + uploaded_image.filename
    #         uploaded_image.save(temp_path)
    #         print(f"Saved image to {temp_path}")
    #     except Exception as e:
    #         print(f"Error: {e}")
    search = request.args.get('search', '')
    image = request.args.get('image', '')

    if request.method == 'POST':
        uploaded_image = request.files.get('image')
        if uploaded_image and uploaded_image.filename != '':
            try:
                temp_path = os.path.join("/Volumes/T7/photos_from_icloud/tmp", uploaded_image.filename)
                uploaded_image.save(temp_path)
                print(f"Saved image to {temp_path}")
                image = uploaded_image.filename
            except Exception as e:
                print(f"Error: {e}")
    else:
        uploaded_image = None
    environment = Environment(loader=FileSystemLoader("templates/"))
    template = environment.get_template("results.html")
    try:
        if text != 'reset' and text != '':
            context = search.return_file('search', text=text, image=None, table='photos_db', limit=50, filter=None)
            return render_template("results.html", **context)
        # elif text == 'reset' or text is None:
        #     return render_template("results.html")

        elif uploaded_image != '' and uploaded_image.filename:
            context = search.return_file('search', text=None, image=temp_path, table='photos_db', limit=50, filter=None)
            return render_template("results.html", **context)
    except Exception as e:   
        print(e)
        print('1')
        return render_template("results.html")
    return render_template("results.html")

if __name__ == '__main__':
    app.run(debug=True)
    print(app.url_map)