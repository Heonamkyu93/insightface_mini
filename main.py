import cv2
import numpy as np
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
from fastapi import FastAPI, File, UploadFile, Form ,WebSocket ,WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
app = FastAPI()
face = FaceAnalysis(allowed_modules=['detection', 'recognition'], providers=['CPUExecutionProvider'])

face.prepare(ctx_id=0, det_size=(640, 640))
import io
from PIL import Image
import time
from starlette.requests import Request
from fastapi.responses import StreamingResponse
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import shutil
from starlette.responses import RedirectResponse
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import StreamingResponse



templates = Jinja2Templates(directory="templates")

# 웹캠 스트리밍 인식 함수
def webcam_stream2():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if(ret) :
            gray = cv2.cvtColor(frame,  cv2.COLOR_BGR2GRAY)    # 입력 받은 화면 Gray로 변환

            face2_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face2_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            # 검출된 얼굴 주위에 사각형을 그립니다.
        
            if cv2.waitKey(1) != -1:
                cv2.imwrite('photo.jpg', frame)
                break

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)   
                _, jpeg = cv2.imencode('.jpg', frame)
                frame_bytes = jpeg.tobytes()
                yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')



        if not ret:
            break
      # 웹캠 스트리밍 함수
def webcam_stream():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        _, jpeg = cv2.imencode('.jpg', frame)
        frame_bytes = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(webcam_stream(), media_type="multipart/x-mixed-replace;boundary=frame")
@app.get("/video_feed2")
async def video_feed2():
    return StreamingResponse(webcam_stream2(), media_type="multipart/x-mixed-replace;boundary=frame")

@app.get("/insight", response_class=HTMLResponse)
async def insight(request: Request):
    # 템플릿에 전달할 데이터
    context = {"request": request, "message": "Hello, FastAPI with Jinja2Templates!"}
    
    # templates 디렉토리에서 root.html을 렌더링하여 반환
    return templates.TemplateResponse("insight.html", {"request": request, **context})



@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    # 템플릿에 전달할 데이터
    context = {"request": request, "message": "Hello, FastAPI with Jinja2Templates!"}
    
    # templates 디렉토리에서 root.html을 렌더링하여 반환
    return templates.TemplateResponse("index.html", {"request": request, **context})
  
@app.post("/up")
async def predict_api(image_file1: UploadFile):
     img = cv2.imread('img/join.jpg')
     faces1 = face.get(img)
     contents1 = await image_file1.read()
     buffer1 = io.BytesIO(contents1)
     pil_img1 = Image.open(buffer1)
     cv_img1=np.array(pil_img1)
     cv2.cvtColor(cv_img1,cv2.COLOR_RGB2BGR)
     faces2 = face.get(cv_img1)
     feat1 = np.array(faces1[0].normed_embedding, dtype=np.float32)
     feats2 = np.array(faces2[0].normed_embedding, dtype=np.float32)
     sims = np.dot(feat1,feats2)
     if sims > 0.55:
        return "yes"
     else :
        return "no"

@app.get("/test")
def test():
    img = cv2.imread('img/join.jpg')
    faces1 = face.get(img)
    cap = cv2.VideoCapture(0)
    feat1 = np.array(faces1[0].normed_embedding, dtype=np.float32)
    print('width :%d, height : %d' % (cap.get(3), cap.get(4)))
    while(True):
      ret, frame = cap.read()    # Read 결과와 frame

      if(ret) :
        gray = cv2.cvtColor(frame,  cv2.COLOR_BGR2GRAY)    # 입력 받은 화면 Gray로 변환

        face2_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face2_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        # 검출된 얼굴 주위에 사각형을 그립니다.
        
        if cv2.waitKey(1) != -1:
                cv2.imwrite('photo.jpg', frame)
                break

        for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)   

        # 결과 영상을 화면에 출력합니다.
        cv2.imshow('Face Detection', frame)
        
        
        #cv2.imshow('frame_color', frame)    # 컬러 화면 출력
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

UPLOAD_FOLDER = "upimg"

    # 이미지 업로드를 처리하는 엔드포인트
@app.post("/uploadfile")
async def create_upload_file(file: UploadFile):
     print('adsasasdad')
     img = cv2.imread('img/join.jpg')
     faces1 = face.get(img)
     contents1 = await file.read()
     buffer1 = io.BytesIO(contents1)
     pil_img1 = Image.open(buffer1)
     cv_img1=np.array(pil_img1)
     cv2.cvtColor(cv_img1,cv2.COLOR_RGB2BGR)
     faces2 = face.get(cv_img1)
     feat1 = np.array(faces1[0].normed_embedding, dtype=np.float32)
     feats2 = np.array(faces2[0].normed_embedding, dtype=np.float32)
     sims = np.dot(feat1,feats2)
     if sims > 0.55:
        print('yes')
        return "yes"
     else :
        print('no')
        return "no"

     