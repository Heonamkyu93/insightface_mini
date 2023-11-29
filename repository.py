from typing import List
from sqlalchemy import select
from sqlalchemy.orm import Session
from orm import Img

def get_img(session: Session) -> List[Img]:
    return session.query(Img).all()
   # return session.execute(select(Img)).scalars().all()


def create_img(db, img_name: str, img_path: str):
    db_img = Img(img_name=img_name, img_path=img_path)
    db.add(db_img)
    db.commit()
    db.refresh(db_img)
    return db_img


def delete_img(db,id:int):
    data = db.query(Img).filter(Img.id == id).first()
    db.delete(data)
    db.commit()