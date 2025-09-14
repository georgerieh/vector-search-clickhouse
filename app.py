import os
from time import sleep
from urllib.parse import unquote

import clickhouse_connect
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    send_from_directory,
    abort,
    make_response,
    flash,
)

import search

client = clickhouse_connect.get_client(
    host=os.environ.get("CLICKHOUSE_HOST", "localhost"),
    username=os.environ.get("CLICKHOUSE_USERNAME", "default"),
    password=os.environ.get("CLICKHOUSE_PASSWORD", ""),
    port=os.environ.get("CLICKHOUSE_PORT", 8123),
)

basedir = os.path.abspath(os.path.dirname(__file__))
BASE_DIR = "/Volumes/T7/photos_from_icloud"
app = Flask(__name__, static_folder="/Volumes/T7/photos_from_icloud")
app.secret_key = "secret"
app.jinja_env.globals.update(unquote=unquote)


@app.route("/files/<path:filename>")
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
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.route("/delete_photo", methods=["POST"])
def delete_photo():
    # Save form/session values
    session["search_text"] = request.form.get(
        "search_text", session.get("search_text", "")
    )
    session["start_date"] = session.get(
        "start_date", request.form.get("start_date", "")
    )
    session["end_date"] = session.get("end_date", request.form.get("end_date", ""))
    session["limit"] = session.get("limit", request.form.get("limit", 50, type=int))

    image_paths_to_delete = request.form.getlist("image_paths")
    if not image_paths_to_delete:
        flash("No images selected for deletion.", "warning")
        return redirect(url_for("home"))

    deleted_count = 0
    for path in image_paths_to_delete:
        full_path = os.path.join(BASE_DIR, path)
        try:
            if os.path.exists(full_path):
                os.remove(full_path)
                deleted_count += 1
        except Exception as e:
            flash(f"Error deleting file {path}: {e}", "error")
        try:
            client.query(f"ALTER TABLE photos_db DELETE WHERE path = '{full_path}'")
        except Exception as e:
            print(f"Error deleting from ClickHouse: {e}")

    flash(
        f"Successfully deleted {deleted_count} image(s)."
        if deleted_count
        else "No images deleted.",
        "info",
    )
    sleep(1)
    # Instead of redirecting, call home() to render the results page
    return redirect(url_for("home"))


@app.route("/", methods=["GET", "POST"])
def home():
    # --- Initialize session and local variables ---
    session.setdefault("search_text", "")
    search_text = session.get("search_text", request.form.get("search_text", ""))
    start_date = session.get("start_date", request.form.get("start_date", ""))
    end_date = session.get("end_date", request.form.get("end_date", ""))
    limit = session.get("limit", request.form.get("limit", 50, type=int))
    # session.setdefault('start_date', '')
    # session.setdefault('end_date', '')
    # session.setdefault('limit', 50)

    uploaded_image = request.files.get("image")

    saved_image_path = None

    # --- Handle POST requests ---
    if request.method == "POST":
        # Update session with current search/filter inputs
        session["search_text"] = request.form.get("search_text", session["search_text"])
        session["start_date"] = request.form.get("start_date", "")
        session["end_date"] = request.form.get("end_date", "")
        session["limit"] = request.form.get("limit", 50, type=int)

        search_text = session["search_text"]
        start_date = session["start_date"]
        end_date = session["end_date"]
        limit = session["limit"]
        print(search_text, start_date, end_date, limit)
        # --- Handle uploaded image ---
        if uploaded_image and uploaded_image.filename != "":
            saved_image_path = os.path.join(BASE_DIR, "tmp", uploaded_image.filename)
            uploaded_image.save(saved_image_path)
            context = search.return_file(
                "search",
                text="",
                image=saved_image_path,
                table="photos_db",
                limit=limit,
                filter_expr="",
                start_date=start_date,
                end_date=end_date,
            )
            return render_template("results.html", **context)

        # --- Handle text search ---
        if search_text:
            context = search.return_file(
                "search",
                text=search_text,
                image="",
                table="photos_db",
                limit=limit,
                filter_expr="",
                start_date=start_date,
                end_date=end_date,
            )
            return render_template("results.html", **context)

        # --- Handle date filter only ---
        if start_date or end_date:
            context = search.return_file(
                "search",
                text="",
                image="",
                table="photos_db",
                limit=limit,
                filter_expr="",
                start_date=start_date,
                end_date=end_date,
            )
            print(context)
            return render_template("results.html", **context)

    # --- Handle GET requests ---
    search_text = session.get("search_text", "")
    start_date = session.get("start_date", "")
    end_date = session.get("end_date", "")
    limit = session.get("limit", 50)

    if search_text or start_date or end_date:
        context = search.return_file(
            "search",
            text=search_text,
            image="",
            table="photos_db",
            limit=limit,
            filter_expr="",
            start_date=start_date,
            end_date=end_date,
        )
        return render_template("results.html", **context)

    # --- Default render if no search/filter ---
    return render_template("results.html")


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
    print(app.url_map)
