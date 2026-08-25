from flask import Flask
import redis

app = Flask(__name__)

database = redis.Redis(
    host="redis",
    port=6379,
    decode_responses=True
)

@app.route("/")
def home():
    count = database.incr("visits")
    return f"""
    <h1>Docker Compose Practical</h1>
    <p>This page has been visited {count} times.</p>
    """

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
