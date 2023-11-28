from sqlalchemy import Boolean,Column , Integer ,String
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class Img(Base):
    __tablename__="img"

    id=Column(Integer,primary_key=True,autoincrement=True,index=True)
    img_name=Column(String(256))
    img_path=Column(String(256))


    def __repr__(self):
        return f"Img(id={self.id}, img_name={self.img_name}, img_path={self.img_path})"