import os
from urllib.parse import unquote

import clickhouse_connect
from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory, abort, \
    make_response, flash

import search

client = clickhouse_connect.get_client(host=os.environ.get('CLICKHOUSE_HOST', 'localhost'),
            username=os.environ.get('CLICKHOUSE_USERNAME', 'default'),
            password=os.environ.get('CLICKHOUSE_PASSWORD', ''),
            port=os.environ.get('CLICKHOUSE_PORT', 8123))

basedir = os.path.abspath(os.path.dirname(__file__))
BASE_DIR = '/Volumes/T7/photos_from_icloud'
app = Flask(__name__, static_folder='/Volumes/T7/photos_from_icloud')
app.secret_key = "secret"
app.jinja_env.globals.update(unquote=unquote)

@app.route('/files/<path:filename>')
def serve_file(filename):
    filename = unquote(filename)
    if filename.startswith(BASE_DIR):
        safe_path = os.path.relpath(filename, BASE_DIR)
    else:
        safe_path = filename

    safe_path = os.path.normpath(safe_path)
    if safe_path.startswith("..") or os.path.isabs(safe_path):
        abort(404)

    response = make_response(send_from_directory(BASE_DIR, safe_path))
    # Force no-cache so browser fetches the file every time
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route('/delete_photo', methods=['POST'])
def delete_photo():
    # request.form.getlist('image_paths') will get all values from checked checkboxes
    image_paths_to_delete = request.form.getlist('image_paths')

    if not image_paths_to_delete:
        flash('No images selected for deletion.', 'warning')
        return redirect(url_for('index'))

    deleted_count = 0
    for image_path in image_paths_to_delete:
        try:
            # Construct the full path to the image file
            # IMPORTANT: Adjust this path to match where your images are actually stored
            full_image_path = os.path.join('/Volumes/T7/photos_from_icloud/', image_path)  # Example path
            if os.path.exists(full_image_path):
                os.remove(full_image_path)
                # You might also want to delete the image's entry from your database here
                # e.g., db.session.query(Image).filter_by(url=image_path).delete()
                # db.session.commit()
                deleted_count += 1
            else:
                print(f"Warning: Image not found at {full_image_path}")
        except Exception as e:
            print(f"Error deleting {image_path}: {e}")
            flash(f"Error deleting {image_path}: {e}", 'error')

    if deleted_count > 0:
        flash(f'Successfully deleted {deleted_count} image(s).', 'success')
    else:
        flash('No images were deleted.', 'info')

    return redirect(url_for('index'))
@app.route("/", methods=['GET','POST'])
def home(search_text=''):
    uploaded_image = None
    saved_image_path = None
    if request.method == 'POST':
        uploaded_image = request.files.get('image')
        form_search_text = request.form.get("search_text", "")
        limit = request.form.get('limit', 50, type=int)
        end_date = request.form.get('end_date', '')
        start_date = request.form.get('start_date', '')
        print('start_date', start_date)

        if uploaded_image and uploaded_image.filename != '':
            saved_image_path = os.path.join("/Volumes/T7/photos_from_icloud/tmp", uploaded_image.filename)
            uploaded_image.save(saved_image_path)
            context = search.return_file(
                'search', text='', image=saved_image_path,
                table='photos_db', limit=limit, filter_expr='', start_date=start_date,
                end_date=end_date
            )
            return render_template("results.html", **context)
        elif form_search_text:
            session["search_text"] = form_search_text
            context = search.return_file(
                'search', text=form_search_text, image='',
                table='photos_db', limit=limit, filter_expr='', start_date=start_date, end_date=''
            )
            return render_template("results.html", **context)
        elif start_date:
            print('executing startdate search')
            context = search.return_file('search', text='', image='', table='photos_db', limit=limit, filter_expr='',
                                         start_date=start_date, end_date=end_date)
            return render_template("results.html", **context)

    # For GET, use session only if no new POST
    search_text = session.get("search_text", "")
    if search_text:
        context = search.return_file(
            'search', text=search_text, image='',
            table='photos_db', limit=50, filter_expr=''
        )
        return render_template("results.html", **context)
    # except Exception as e:
    #     print(f"Error during search: {e}")
    #     return render_template("results.html")

    return render_template("results.html")
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
    print(app.url_map)
