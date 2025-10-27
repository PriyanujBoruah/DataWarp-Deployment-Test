# run.py

from app import create_app

app = create_app()

if __name__ == '__main__':
    # The debug=True flag allows you to see errors in the browser
    # and automatically reloads the server when you make changes.
    # The host='0.0.0.0' makes it accessible on your network.
    # This block is ONLY used for local development, not by Gunicorn in Docker.
    app.run(debug=True, host='0.0.0.0')