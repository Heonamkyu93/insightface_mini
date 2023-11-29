import cv2
import numpy as np
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
from fastapi import FastAPI, File, UploadFile, Form ,WebSocket ,WebSocketDisconnect ,Depends
from fastapi.staticfiles import StaticFiles
app = FastAPI()
face = FaceAnalysis(allowed_modules=['detection', 'recognition'], providers=['CPUExecutionProvider'])
from sqlalchemy.orm import Session
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
from pydantic import BaseModel
import base64
from fastapi.responses import JSONResponse
import os
from database import get_db
from typing import List
templates = Jinja2Templates(directory="templates")
from repository import get_img
from orm import Img
from pydantic import BaseModel
from database import engine
from PIL import Image
from io import BytesIO
from database import SessionFactory
from repository import create_img ,delete_img
import uuid

# 웹캠 안면 인식하는 함수
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
      # 웹캠 스트리밍만 하는 함수
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


#  index html 에 스트리밍만 해주는 함수
@app.get("/video_feed")  
async def video_feed():
    return StreamingResponse(webcam_stream(), media_type="multipart/x-mixed-replace;boundary=frame")
# 얼굴등록 save html에 스트리밍만 해주는 함수
@app.get("/video_feed3")
async def video_feed():
    return StreamingResponse(webcam_stream(), media_type="multipart/x-mixed-replace;boundary=frame")
# insight html 에 얼굴인식해주는 함수
@app.get("/video_feed2")
async def video_feed2():
    return StreamingResponse(webcam_stream2(), media_type="multipart/x-mixed-replace;boundary=frame")

# insight html 불러오는 함수
@app.get("/insight", response_class=HTMLResponse)
async def insight(request: Request):
    # 템플릿에 전달할 데이터
    context = {"request": request, "message": "Hello, FastAPI with Jinja2Templates!"}
    
    # templates 디렉토리에서 root.html을 렌더링하여 반환
    return templates.TemplateResponse("insight.html", {"request": request, **context})


# index html 불러오는 함수
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    # 템플릿에 전달할 데이터
    context = {"request": request, "message": "Hello, FastAPI with Jinja2Templates!"}
    
    # templates 디렉토리에서 root.html을 렌더링하여 반환
    return templates.TemplateResponse("index.html", {"request": request, **context})
# 얼굴 등록 하는 save html 불러오는 함수
@app.get("/save", response_class=HTMLResponse)
async def read_root(request: Request):
    # 템플릿에 전달할 데이터
    context = {"request": request, "message": "Hello, FastAPI with Jinja2Templates!"}
    
    # templates 디렉토리에서 root.html을 렌더링하여 반환
    return templates.TemplateResponse("save.html", {"request": request, **context})


# 사진비교 테스트 함수 필요없음
@app.post("/up")
async def predict_api(image_file1: UploadFile):
     img = cv2.imread('img/join.jpg')
     faces1 = face.get(img)
     contents1 = await image_file1.read()
     buffer1 = io.BytesIO(contents1)
     pil_img1 = Image.open(buffer1)
     cv_img1=np.array(pil_img1)
     cv2.cvtColor(cv_img1,cv2.COLOR_RGB2BGR)
     cv2.imwrite("test/i1_output.jpg")
     faces2 = face.get(cv_img1)
     feat1 = np.array(faces1[0].normed_embedding, dtype=np.float32)
     feats2 = np.array(faces2[0].normed_embedding, dtype=np.float32)
     sims = np.dot(feat1,feats2)
     if sims > 0.55:
        return "yes"
     else :
        return "no"
# 테스트용 필요없음
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
class ImageData(BaseModel):
    image: str

# 디비에서 가져온 파일이름 경로를 이용해서 폴더에서 파일을 긁어오는 함수    
def read_images(img_instances: List[Img]) -> List:
    images = []
    for img_instance in img_instances:
        img_path = img_instance.img_path
        img_name = img_instance.img_name
        image = cv2.imread(img_path+"/"+img_name)
        images.append(image)
    return images
# index 에서 axios 를 사용해서 이미지를 보내면 디비에서 가져온 정보로 파일을 긁어오고 비교후 출입 가능여부를 반환하는 함수
@app.post("/uploadfile")
async def create_upload_file(image_data: ImageData,session: Session = Depends(get_db)):
    img2: List[Img] = get_img(session=session)
    images = read_images(img2)
  #  img = cv2.imread('img/join.jpg')
  #  faces1 = face.get(img)
  #  feat1 = np.array(faces1[0].normed_embedding, dtype=np.float32)
    try:
        # base64 데이터를 이미지로 디코딩
        encoded_image = image_data.image.split(",")[1]
        decoded_image = base64.b64decode(encoded_image)
        realImg = Image.open(BytesIO(decoded_image))
       # realImg.save("upimg/image.jpg")
        image_np = np.frombuffer(decoded_image, dtype=np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        faces2 = face.get(image)
        feats2 = np.array(faces2[0].normed_embedding, dtype=np.float32)
        result = False
        for idx, image in enumerate(images):
                
                faces1 = face.get(image)
                feat1 = np.array(faces1[0].normed_embedding, dtype=np.float32)

                sims = np.dot(feat1,feats2)
                if sims > 0.55:
                    result = True     
                
        if result:
                    return JSONResponse(content={"message": "출입가능"})
        elif not result:
                    return JSONResponse(content={"message": "접근금지"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

# save html에서 axios로 이미지 업로드하면 파일 정보를 디비에 저장하고 저장소에 저장하는 메소드    
@app.post("/img_save")
async def create_upload_file(image_data: ImageData,session: Session = Depends(get_db)):
    try:
        # base64 데이터를 이미지로 디코딩
        encoded_image = image_data.image.split(",")[1]
        decoded_image = base64.b64decode(encoded_image)
        realImg = Image.open(BytesIO(decoded_image))
        new_uuid = uuid.uuid4()
        uuid_str = str(new_uuid)
        file_name = f"{uuid_str}.jpg"
        img_path="upimg"
        realImg.save(img_path+"/"+file_name)
        img: List[Img] = get_img(session=session)
        db = SessionFactory()
        try:
            img = create_img(db, img_name=file_name, img_path=img_path)
        finally:
            db.close()

        return JSONResponse(content={"message": "등록완료 출근하세요"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
# 디비 테스트 필요없음    
@app.get("/db")
async def root(session: Session = Depends(get_db)):
    img: List[Img] = get_img(session=session)
    db = SessionFactory()
    name='hi'
    path='ai'
    try:
        img = create_img(db, img_name=name, img_path=path)
        return {"img_name": name, "img_path": path}
    finally:
        db.close()
# 디비 테스트 필요없음
@app.get("/db2")
async def root(session: Session = Depends(get_db)):
    img: List[Img] = get_img(session=session)
    return img
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 이 부분을 필요에 따라 실제 도메인으로 변경하세요.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/upimg", StaticFiles(directory="upimg"), name="upimg")
@app.get("/admin",response_class=HTMLResponse)
async def admin(request: Request,session: Session = Depends(get_db)):
     img: List[Img] = get_img(session=session)
     context = {"request": request, "message": img}
     return templates.TemplateResponse("admin.html", {"request": request, **context})


class Del_name(BaseModel):
    del_id: int
    del_name: str
@app.post("/delete_img")   # 해야할일 db에서 데이터가 지워지니깐 로컬에 저장소에있는 파일을 직접 지울것
async def del_img(Del_name: Del_name,session: Session = Depends(get_db) ):
    delete_img(session, Del_name.del_id)
    file_path='upimg/'+Del_name.del_name
    if os.path.exists(file_path):
        os.remove(file_path)
    return JSONResponse(content={"message": "test"})







     