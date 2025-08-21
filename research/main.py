from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Hello, World!"}

@app.get("/posts")
def get_posts():
    return {"data": "this is your post"}

@app.post("/createposts")
def create_post():
    return {"message": "Post created successfully"}

